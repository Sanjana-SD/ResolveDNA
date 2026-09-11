"""Unit tests for full ResolveDNA pipeline."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import ResolveDNAPipeline


@pytest.fixture(scope="module")
def pipeline():
    return ResolveDNAPipeline(memory_path="data/processed/resolution_memory_sample.pkl")


def test_pipeline_execution_structure(pipeline):
    result = pipeline.process("Where is my order? It was supposed to arrive yesterday.")
    assert "customer_message" in result
    assert "intent" in result
    assert "decision" in result
    assert "resolution_confidence" in result
    assert "resolution_consistency" in result
    assert "next_best_action" in result
    assert "draft_reply" in result
    assert "verification" in result
    assert "retrieved_cases" in result
    assert result["decision"] in ["AUTO_HANDLE", "ESCALATE"]


def test_pipeline_escalation_on_ambiguity(pipeline):
    result = pipeline.process("Help")
    assert result["decision"] == "ESCALATE"
