"""Generate ``docs/benchmark_report.md`` from a demo benchmark run."""

from __future__ import annotations


def main() -> None:
    import asyncio
    import sys
    import tempfile
    from pathlib import Path

    # When this script is invoked directly the ``benchmarks/langgraph`` package can
    # shadow the installed ``langgraph`` library, so we remove the script directory
    # from ``sys.path`` and ensure the repo root is present.
    script_dir = str(Path(__file__).parent)
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    repo_root = str(Path(__file__).parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from atlascore.research import ResearchPipeline
    from benchmarks.benchmark import BenchmarkHarness
    from benchmarks.fakes import (
        FakeBenchmarkFetchTool,
        FakeBenchmarkModelClient,
        FakeBenchmarkSearchTool,
        make_fake_outputs,
    )
    from benchmarks.langgraph import LangGraphResearchPipeline

    model = FakeBenchmarkModelClient(make_fake_outputs())
    search = FakeBenchmarkSearchTool()
    fetch = FakeBenchmarkFetchTool()

    with tempfile.TemporaryDirectory() as tmp:
        atlascore_pipeline = ResearchPipeline(
            model_client=model,
            search_tool=search,
            fetch_tool=fetch,
            persist_dir=tmp,
        )
        langgraph_pipeline = LangGraphResearchPipeline(
            model_client=model,
            search_tool=search,
            fetch_tool=fetch,
        )

        harness = BenchmarkHarness(atlascore_pipeline, langgraph_pipeline)
        report = asyncio.run(
            harness.run(
                queries=["What is Atlas?", "How does Atlas optimize research cost?"],
                expected_outputs=[
                    "Atlas is a multi-agent research platform.",
                    "Atlas uses a cheap triage model and tracks cost estimates per LLM call.",
                ],
            )
        )

    repo_root_path = Path(__file__).parents[1]
    docs_dir = repo_root_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    report.save(docs_dir / "benchmark_report.md")
    print(f"Report written to {docs_dir / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
