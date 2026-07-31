import pytest

from atlascore import Citation, Evidence, ResearchBrief, VerificationResult


@pytest.mark.asyncio
async def test_research_brief_markdown():
    brief = ResearchBrief(
        title="Python",
        summary="Python is great.",
        sections=[{"heading": "Intro", "content": "Python intro."}],
        citations=[
            Citation(
                source_title="Python.org",
                source_url="https://python.org",
                quote="Python is awesome",
                index=1,
            )
        ],
        confidence=0.95,
    )
    markdown = brief.to_markdown()
    assert "# Python" in markdown
    assert "Python is awesome" in markdown
    assert "95%" in markdown


@pytest.mark.asyncio
async def test_verification_result():
    result = VerificationResult(
        overall_confidence=0.8,
        evidence=[
            Evidence(
                claim="Python supports async",
                assessment="supported",
                confidence=0.99,
                citations=[],
            )
        ],
    )
    assert result.overall_confidence == 0.8
    assert len(result.evidence) == 1
    assert result.evidence[0].confidence == 0.99
