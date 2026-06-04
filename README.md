# Negev

A modular benchmarking framework for **LLM data poisoning attacks and defenses**, focused on OWASP LLM03 (Training Data Poisoning).

> **Early stage project.** This is an active work in progress, not production ready software. The architecture and core pipeline work end to end, but many planned attacks, defenses, and features are not yet implemented. Expect rough edges.

## What it is

Negev lets you simulate data poisoning attacks against language models, measure how much damage they cause, then run defensive data sanitization pipelines and measure how much of that damage they recover.

A single run does this:

```
load model + dataset
* train a clean baseline, measure it
* poison the data, train on it, measure the degradation
* run a defense to clean the data, retrain, measure the recovery
* return a full result with all three metric snapshots
```

## Why

Training data poisoning is one of the least tooled up areas of LLM security. Most existing work is either academic one off scripts or buried inside larger ML robustness libraries. The goal here is a focused, self hostable tool a red teamer can point at a model and dataset to answer two questions: *how badly can this attack hurt the model, and how well does this defense stop it?*

## Current state

So far tested end to end on DistilBERT + IMDB as the development target:

  **Attacks**: untargeted label flipping (random, multi class, seeded/reproducible)
  **Training**: fine tuning loop wrapping HuggingFace `Trainer` with explicit train/eval splits
  **Evaluation**: accuracy capture, with a metrics snapshot data model
  **Defenses**: `cleanlab` based label noise filtering (confident learning), verified to measurably recover accuracy on poisoned data
  **Orchestration**: a runner that ties the full attack -> train  -> defend -> retrain pipeline into a single call *(in progress)*
  **Interfaces**: a FastAPI backend and a Textual terminal UI, decoupled so the core logic never depends on either

Verified result so far: at a 30% label flip poison rate on 2,000 samples, the cleanlab defense flagged ~537 rows and recovered roughly +0.05 accuracy versus training on the poisoned data directly.

## Architecture

Pipeline goes through three layers: `TUI -> API -> Core`:

```
core/        pure logic, no interface dependencies
  attacks/     poisoning attacks (Attack base class + implementations)
  defenses/    data sanitization pipelines
  evaluation/  metric capture
  training/    fine tuning loop
  models/      Pydantic schemas (AttackConfig, RunResult, MetricSnapshot, FilterReport)
  runner.py    full pipeline orchestrator
api/         FastAPI layer that exposes the core
tui/         Textual terminal dashboard
```

The decoupling is deliberate: the core knows nothing about how it's called, so the same code runs from a notebook, the API, the TUI, or a future web frontend without changes.

## A note on how this is built

Everything in `core/`, `api/`, and the architecture itself is written by hand, that's the entire point of the project for me. I'm using it to learn ML and security tooling in depth, so the engine is built deliberately and slowly rather than generated.

The **terminal UI is the exception**: frontend/design isn't where my learning focus is, so the TUI is AI assisted while i concentrate the hand written effort on the parts i actually want to understand deeply. The interface is intentionally treated as a thin, replaceable shell over the core.

## Roadmap (rough)

More attacks: targeted label flip, backdoor triggers, LoRA weight poisoning, feature collision
More defenses: embedding based outlier detection (`pyod`)
The full pipeline view (attacks x defenses)
Containerization for self hosted deployment

## Stack

Python, HuggingFace (`transformers`, `datasets`), `cleanlab`, FastAPI, Textual, Pydantic. (see all dependancies in requirements.txt)

   

*Built by a solo developer as a learning driven security research project. Not affiliated with any organization.*