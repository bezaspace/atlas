"""Smoke test for the full Phase 7 research pipeline.

Usage:
    export LLM_API_KEY="..."
    export LLM_BASE_URL="https://api.openai.com/v1"  # optional
    export LLM_MODEL="gpt-4o-mini"
    export TAVILY_API_KEY="..."  # or GOOGLE_API_KEY + GOOGLE_CSE_ID
    python examples/research_pipeline.py "Your research question"
"""

import argparse
import asyncio
import os
import sys

from atlascore import OpenAIChatCompletionClient
from atlascore.research import ResearchPipeline
from atlascore.tools import WebFetchTool, WebSearchTool


def get_env_or_exit(name: str) -> str:
    value = os.getenv(name)
    if not value:
        print(f"❌ Environment variable {name} is required")
        sys.exit(1)
    return value


async def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas Research Pipeline")
    parser.add_argument(
        "query", nargs="?", default="What are the latest advances in multi-agent systems?"
    )
    parser.add_argument(
        "--persist-dir",
        default="data/research",
        help="Directory to save the research brief and sources",
    )
    args = parser.parse_args()

    api_key = get_env_or_exit("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL")

    search_api_key = os.getenv("TAVILY_API_KEY") or os.getenv("GOOGLE_API_KEY")
    search_provider = "tavily" if os.getenv("TAVILY_API_KEY") else "google"
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not search_api_key:
        print("❌ Set TAVILY_API_KEY or GOOGLE_API_KEY + GOOGLE_CSE_ID for web search")
        sys.exit(1)

    model_client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    search_tool = WebSearchTool(
        api_key=search_api_key,
        provider=search_provider,
        cse_id=cse_id,
    )
    fetch_tool = WebFetchTool()

    pipeline = ResearchPipeline(
        model_client=model_client,
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        persist_dir=args.persist_dir,
    )

    print(f"🔬 Researching: {args.query}")
    report = await pipeline.run(args.query)

    print("\n✅ Research complete")
    print(f"\nTitle: {report.brief.title}")
    print(f"Summary: {report.brief.summary}")
    print(f"Confidence: {report.brief.confidence:.0%}")
    print(f"Sources: {len(report.sources)}")
    print(f"Critic requested revisions: {report.critic_review.revisions_required if report.critic_review else 'N/A'}")
    print(f"Artifacts: {', '.join(report.paths)}")


if __name__ == "__main__":
    asyncio.run(main())
