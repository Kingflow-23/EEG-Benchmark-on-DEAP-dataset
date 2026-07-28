"""Tests for the structured workflow tracker."""

import json

from src.workflow import WorkflowTracker


def test_workflow_tracker_persists_events(tmp_path):
    tracker = WorkflowTracker(tmp_path, run_name="demo")
    tracker.log(20, "workflow_started", split="subject")
    with tracker.stage("benchmark", split="subject"):
        tracker.log(20, "experiment_started", model="mlp")
    tracker.finish(best_model={"Valence": "mlp"})

    workflow = json.loads((tmp_path / "workflow.json").read_text(encoding="utf-8"))
    events = (
        (tmp_path / "workflow.jsonl").read_text(encoding="utf-8").strip().splitlines()
    )

    assert workflow["run_name"] == "demo"
    assert workflow["stages"][0]["name"] == "benchmark"
    assert workflow["stages"][0]["status"] == "ok"
    assert any(event["message"] == "workflow_completed" for event in workflow["events"])
    assert len(events) == len(workflow["events"])
