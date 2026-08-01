"""Benchmark tests for Phase 15 — framework comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlascore.research import ResearchPipeline
from atlascore.research_schemas import ResearchReport
from benchmarks.benchmark import BenchmarkHarness, BenchmarkReport
from benchmarks.fakes import (
    FakeBenchmarkFetchTool,
    FakeBenchmarkModelClient,
    FakeBenchmarkSearchTool,
    make_fake_outputs,
)
from benchmarks.langgraph import LangGraphResearchPipeline


@pytest.fixture
def fake_model():
    return FakeBenchmarkModelClient(make_fake_outputs())


@pytest.fixture
def fake_search():
    return FakeBenchmarkSearchTool()


@pytest.fixture
def fake_fetch():
    return FakeBenchmarkFetchTool()


@pytest.mark.asyncio
async def test_both_pipelines_produce_comparable_briefs(fake_model, fake_search, fake_fetch, tmp_path):
    atlascore_pipeline = ResearchPipeline(
        model_client=fake_model,
        search_tool=fake_search,
        fetch_tool=fake_fetch,
        persist_dir=str(tmp_path / "atlascore"),
    )
    langgraph_pipeline = LangGraphResearchPipeline(
        model_client=fake_model,
        search_tool=fake_search,
        fetch_tool=fake_fetch,
    )

    ac_report = await atlascore_pipeline.run("What is Atlas?")
    lg_report = await langgraph_pipeline.run("What is Atlas?")

    assert isinstance(ac_report, ResearchReport)
    assert isinstance(lg_report, ResearchReport)
    assert ac_report.brief.title
    assert lg_report.brief.title
    assert "Atlas" in ac_report.brief.summary
    assert "Atlas" in lg_report.brief.summary


@pytest.mark.asyncio
async def test_benchmark_harness_runs_and_reports_metrics(fake_model, fake_search, fake_fetch, tmp_path):
    atlascore_pipeline = ResearchPipeline(
        model_client=fake_model,
        search_tool=fake_search,
        fetch_tool=fake_fetch,
        persist_dir=str(tmp_path / "atlascore"),
    )
    langgraph_pipeline = LangGraphResearchPipeline(
        model_client=fake_model,
        search_tool=fake_search,
        fetch_tool=fake_fetch,
    )

    harness = BenchmarkHarness(atlascore_pipeline, langgraph_pipeline)
    report = await harness.run(
        queries=["What is Atlas?"],
        expected_outputs=["Atlas is a multi-agent research platform."],
    )

    assert isinstance(report, BenchmarkReport)
    assert len(report.queries) == 1
    q = report.queries[0]
    assert q.atlascore_quality > 0.5
    assert q.langgraph_quality > 0.5
    assert q.atlascore_duration_ms >= 0
    assert q.langgraph_duration_ms >= 0
    assert "queries" in report.summary
    assert report.summary["queries"] == 1

    report_path = tmp_path / "benchmark_report.md"
    report.save(report_path)
    assert Path(report_path).exists()
    assert "Atlas Framework Comparison Benchmark" in Path(report_path).read_text(encoding="utf-8")
