"""
Uses fastAPI to interact with the tool

Currently has a nice set of start APIs, like health to see
availability, runs to get a list of runs, create a run, or
see a specific run's details.
"""

from fastapi import FastAPI, HTTPException
from core.models import AttackConfig, RunResult, MetricSnapshot
from datetime import datetime

TITLE = "Negev"
VERSION = "0.1.0"

app = FastAPI(title=TITLE, version=VERSION)

_runs: dict[str, RunResult] = {} # storing this in memory for now. will use DBs later


def _make_fake_result(config: AttackConfig) -> RunResult:
    """
    Creates a fake result so i could debug stuff with it and test UI later. (AI generated function)

    Args:
        config (AttackConfig): Configuration of the attack

    Returns:
        RunResult: Result of the attack
    """
    return RunResult(
        config=config,
        status="complete",
        baseline_metrics=MetricSnapshot(accuracy=0.91, perplexity=12.3),
        post_attack_metrics=MetricSnapshot(
            accuracy=0.91 - (config.poison_rate * 0.6),
            perplexity=12.3 + (config.poison_rate * 20),
            attack_success_rate=None
        ),
        duration_seconds=42.7,
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.get("/runs")
def list_runs() -> list[RunResult]:
    """
    Sends a get request to /runs in order to get a list of ALL available runs

    Returns:
        list[RunResult]: A list of available results stored in the server
    """
    return list(_runs.values())


@app.post("/runs", status_code=201)
def create_run(config: AttackConfig) -> RunResult:
    """
    Sends a post request to /runs in order to create a new run

    Args:
        config (AttackConfig): Configuration of the attack

    Returns:
        RunResult: A result instance
    """
    result = _make_fake_result(config)
    _runs[result.run_id] = result
    return result


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> RunResult:
    """
    Sends a get request to /runs with an ID param to retrieve a SPECIFIC run

    Args:
        run_id (str): The ID of the target run

    Returns:
        RunResult: The retrieved return instance
    """
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return _runs[run_id]