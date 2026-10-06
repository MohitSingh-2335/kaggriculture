# Kaggriculture Agent

An autonomous agent for the [Kaggriculture Kaggle competition](https://www.kaggle.com/competitions/kaggriculture). The agent manages a virtual farm by planting crops, hiring labor, buying land, caring for animals, and trading on a dynamic market.

This repository contains the agent implementation and local tests only. Competition replays, leaderboard exports, generated notebooks, credentials, and submission archives are intentionally excluded.

## Overview

- **Competition:** Kaggle Kaggriculture (`kaggle_environments` framework)
- **Goal:** Maximize final bank against the opponent
- **Approach:** Typed observation parsing, market-price modeling, heuristic strategy, and multi-unit scheduling

## Repo structure

```
kaggriculture/
├── agent/          # Submission entrypoint (main.py) — what actually ships to Kaggle
├── src/            # Core agent logic: scoring, heuristics, market model, etc.
├── notebooks/       # Local research only; ignored and not part of the public source
├── data/            # Local competition data only; ignored and not part of the public source
├── tests/           # Local dry-run harness and unit tests
├── docs/            # Competition overview and local usage notes
└── scripts/         # Bundling, benchmark, and smoke-test utilities
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Development workflow

This project follows a two-tier workflow:
- **Architecture & review** — high-level design, schema/API contracts, PR audits
- **Implementation** — codebase scaffolding, multi-file writes, tests, debugging

See [`docs/competition-compliance.md`](docs/competition-compliance.md) for the publication boundary and the files intentionally excluded from version control.

## Tech stack

Python · `kaggle-environments` · heuristic strategy

## Build a submission archive

```bash
python scripts/bundle.py -o submission.tar.gz
```

The archive contains only `main.py` and the `src/` package. The generated archive is ignored by Git.
