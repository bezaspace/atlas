"""OpenTelemetry middleware for atlascore."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Union

from ..types import AgentEvent
from ._base import BaseMiddleware, MiddlewareContext

logger = logging.getLogger(__name__)

try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    if TYPE_CHECKING:
        from opentelemetry import metrics, trace  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
        from opentelemetry.sdk.metrics import MeterProvider  # type: ignore
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
        from opentelemetry.trace import Status, StatusCode  # type: ignore


def _is_enabled() -> bool:
    return os.getenv("ATLAS_ENABLE_OTEL", "false").lower() in ("true", "1", "yes")


def _should_capture_content() -> bool:
    return os.getenv("ATLAS_OTEL_CAPTURE_CONTENT", "false").lower() in ("true", "1", "yes")


def _setup_telemetry() -> tuple:
    if not OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry enabled but libraries not installed. "
            'Install with: pip install "atlascore[otel]"'
        )
        return None, None

    try:
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        service_name = os.getenv("OTEL_SERVICE_NAME", "atlascore")
        metrics_enabled = os.getenv("OTEL_METRICS_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        resource = Resource.create({"service.name": service_name})
        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
        trace.set_tracer_provider(trace_provider)
        tracer = trace.get_tracer("atlascore")

        meter = None
        if metrics_enabled:
            try:
                metric_reader = PeriodicExportingMetricReader(
                    OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics")
                )
                meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
                metrics.set_meter_provider(meter_provider)
                meter = metrics.get_meter("atlascore")
                logger.info("OpenTelemetry initialized: service=%s, endpoint=%s", service_name, endpoint)
            except Exception as e:
                logger.debug("Metrics setup failed: %s", e)
        else:
            logger.info("OpenTelemetry initialized (metrics disabled): service=%s", service_name)

        return tracer, meter
    except Exception as e:
        logger.error("Failed to initialize OpenTelemetry: %s", e)
        return None, None


class OTelMiddleware(BaseMiddleware):
    """Emits OpenTelemetry spans for agent, model, and tool calls."""

    def __init__(self) -> None:
        self._enabled = _is_enabled()
        self._capture_content = _should_capture_content()

        if not self._enabled:
            self._tracer = None
            self._meter = None
            return

        self._tracer, self._meter = _setup_telemetry()
        if not self._tracer:
            self._enabled = False
            return

        if self._meter:
            self._token_histogram = self._meter.create_histogram(
                name="gen_ai.client.token.usage",
                unit="{token}",
                description="Number of tokens used in operation",
            )
            self._duration_histogram = self._meter.create_histogram(
                name="gen_ai.client.operation.duration",
                unit="s",
                description="Duration of AI operation",
            )
        else:
            self._token_histogram = None
            self._duration_histogram = None

    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Union[MiddlewareContext, AgentEvent], None]:
        if not self._enabled or not self._tracer:
            yield context
            return

        from opentelemetry import context as otel_context
        from opentelemetry import trace

        span_name = self._span_name(context)
        span = self._tracer.start_span(span_name)
        ctx_token = otel_context.attach(trace.set_span_in_context(span))

        span.set_attribute("gen_ai.system", "atlascore")
        span.set_attribute("gen_ai.operation.name", context.operation)
        span.set_attribute("gen_ai.agent.name", context.agent_name)
        if context.agent_context.session_id:
            span.set_attribute("gen_ai.session.id", context.agent_context.session_id)

        if context.operation == "model_call":
            span.set_attribute("gen_ai.request.model", self._get_model_name(context))
            if self._capture_content and isinstance(context.data, list):
                try:
                    messages = self._format_input_messages(context.data)
                    span.set_attribute("gen_ai.input.messages", json.dumps(messages))
                except Exception as e:
                    logger.debug("Failed to capture input messages: %s", e)
        elif context.operation == "tool_call":
            span.set_attribute("gen_ai.tool.name", self._get_tool_name(context))
            if self._capture_content and hasattr(context.data, "parameters"):
                try:
                    span.set_attribute(
                        "gen_ai.tool.parameters", json.dumps(context.data.parameters)
                    )
                except Exception as e:
                    logger.debug("Failed to capture tool parameters: %s", e)

        context.metadata["_otel_span"] = span
        context.metadata["_otel_token"] = ctx_token
        context.metadata["_otel_start"] = time.time()
        yield context

    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Union[Any, AgentEvent], None]:
        if not self._enabled:
            yield result
            return

        span = context.metadata.get("_otel_span")
        if not span:
            yield result
            return

        try:
            start_time = context.metadata.get("_otel_start", time.time())
            duration_s = time.time() - start_time
            if self._duration_histogram:
                self._duration_histogram.record(
                    duration_s, {"gen_ai.operation.name": context.operation}
                )

            if context.operation == "model_call" and hasattr(result, "usage"):
                if hasattr(result.usage, "tokens_input"):
                    tokens_in = result.usage.tokens_input
                    span.set_attribute("gen_ai.usage.input_tokens", tokens_in)
                    if self._token_histogram:
                        self._token_histogram.record(
                            tokens_in,
                            {"gen_ai.token.type": "input", "gen_ai.operation.name": context.operation},
                        )
                if hasattr(result.usage, "tokens_output"):
                    tokens_out = result.usage.tokens_output
                    span.set_attribute("gen_ai.usage.output_tokens", tokens_out)
                    if self._token_histogram:
                        self._token_histogram.record(
                            tokens_out,
                            {"gen_ai.token.type": "output", "gen_ai.operation.name": context.operation},
                        )

                if self._capture_content and hasattr(result, "message"):
                    try:
                        output = self._format_output_message(result.message)
                        span.set_attribute("gen_ai.output.messages", json.dumps([output]))
                    except Exception as e:
                        logger.debug("Failed to capture output messages: %s", e)
            elif context.operation == "tool_call" and hasattr(result, "success"):
                span.set_attribute("gen_ai.tool.success", result.success)
                if self._capture_content and hasattr(result, "result"):
                    try:
                        span.set_attribute("gen_ai.tool.result", str(result.result))
                    except Exception as e:
                        logger.debug("Failed to capture tool result: %s", e)

            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            logger.debug("Error recording telemetry: %s", e)
        finally:
            from opentelemetry import context as otel_context

            ctx_token = context.metadata.get("_otel_token")
            if ctx_token:
                otel_context.detach(ctx_token)
            span.end()

        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Union[Any, AgentEvent], None]:
        if not self._enabled:
            if False:
                yield
            raise error

        span = context.metadata.get("_otel_span")
        if span:
            try:
                span.set_status(Status(StatusCode.ERROR, str(error)))
                span.set_attribute("error.type", type(error).__name__)
                span.set_attribute("error.message", str(error))
            except Exception:
                pass
            finally:
                from opentelemetry import context as otel_context

                ctx_token = context.metadata.get("_otel_token")
                if ctx_token:
                    otel_context.detach(ctx_token)
                span.end()
        if False:
            yield
        raise error

    def _span_name(self, context: MiddlewareContext) -> str:
        if context.operation == "model_call":
            return f"chat {self._get_model_name(context)}"
        if context.operation == "tool_call":
            return f"tool {self._get_tool_name(context)}"
        return f"{context.operation} {context.agent_name}"

    def _get_model_name(self, context: MiddlewareContext) -> str:
        return context.metadata.get("model", "unknown")

    def _get_tool_name(self, context: MiddlewareContext) -> str:
        if hasattr(context.data, "tool_name"):
            return context.data.tool_name
        return "unknown"

    def _format_input_messages(self, messages: list) -> list:
        formatted = []
        for msg in messages:
            role = "user"
            if hasattr(msg, "source") and msg.source in ("assistant", "system"):
                role = msg.source
            parts = []
            if hasattr(msg, "content") and msg.content:
                parts.append({"type": "text", "content": msg.content})
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(
                        {
                            "type": "tool_call",
                            "id": getattr(tc, "call_id", ""),
                            "name": getattr(tc, "tool_name", ""),
                            "arguments": getattr(tc, "parameters", {}),
                        }
                    )
            if parts:
                formatted.append({"role": role, "parts": parts})
        return formatted

    def _format_output_message(self, message) -> dict:
        parts = []
        if hasattr(message, "content") and message.content:
            parts.append({"type": "text", "content": message.content})
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                parts.append(
                    {
                        "type": "tool_call",
                        "id": getattr(tc, "call_id", ""),
                        "name": getattr(tc, "tool_name", ""),
                        "arguments": getattr(tc, "parameters", {}),
                    }
                )
        return {"role": "assistant", "parts": parts}


def auto_instrument() -> None:
    """Auto-instrument Agent.__init__ with OTelMiddleware when enabled."""
    if not _is_enabled():
        return

    try:
        from ..agents._agent import Agent

        original_init = Agent.__init__

        def instrumented_init(self, *args, middlewares=None, **kwargs):
            middlewares = list(middlewares or [])
            middlewares.insert(0, OTelMiddleware())
            original_init(self, *args, middlewares=middlewares, **kwargs)

        Agent.__init__ = instrumented_init  # type: ignore
        logger.info("OpenTelemetry auto-instrumentation enabled")
    except Exception as e:
        logger.error("Failed to auto-instrument: %s", e)
