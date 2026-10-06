# Kaggriculture Agent

An autonomous farming agent created for the
[Kaggriculture Kaggle competition](https://www.kaggle.com/competitions/kaggriculture).
The competition is a two-player farming simulation: each agent manages land,
crops, animals, labor, and a dynamic market over a fixed season. The winner is
the agent with the most money at the end of the season.

This repository presents the final agent implementation and the engineering
behind it. It is intended as a project showcase and reproducible source
repository, not as an installable application.

## What I built

The agent receives the game observation each turn and returns a valid Kaggle
action dictionary. Its main components are:

- **Typed game-state parsing** for farms, crops, animals, inventories, markets,
  town demand, and private observations.
- **Dynamic market modeling** for price curves, inventory impact, marginal
  revenue, buying costs, and sell-order sizing.
- **Phase-based strategy** that adapts between early production, mid-season
  expansion, and late-season liquidation.
- **Multi-unit scheduling** for the farmer and hired hands, including movement,
  watering, feeding, harvesting, planting, construction, and shed logistics.
- **Economic safeguards** for feed availability, cash reserves, land purchases,
  labor costs, market timing, and terminal selling.
- **Kaggle-compatible packaging** with a standalone submission entrypoint.

## How the submission works

Kaggle calls [`agent/main.py`](agent/main.py) once per turn. The entrypoint
parses the raw observation, builds a typed [`GameState`](src/game_state.py),
generates market orders and farm tasks, assigns actions to available units, and
returns:

```python
{
    "farmer": [...],
    "hands": [[...], ...],
    "market": [[...], ...],
}
```

The submission bundle contains only `main.py` and the `src/` package. It does
not depend on local notebooks, replay files, credentials, or private datasets.

## Repository structure

```text
kaggriculture/
├── agent/          # Kaggle submission entrypoint
├── src/            # State model, strategy, scheduler, market, and actions
├── tests/          # Unit tests for the agent components
├── docs/           # Competition overview and compliance guidance
├── scripts/        # Bundling, local-run, benchmark, and smoke-test utilities
├── requirements.txt
├── LICENSE
└── README.md
```

Competition replays, leaderboard exports, downloaded data, exploratory
notebooks, generated reports, credentials, and submission archives are
intentionally excluded from the public repository. See
[`docs/competition-compliance.md`](docs/competition-compliance.md).

## Optional local development

You do not need to install this repository to read or review the project.
Installation is only needed if you want to run the tests, use the local
simulation tools, or build a submission archive:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Build a Kaggle submission

After installing the optional development dependencies:

```bash
python scripts/bundle.py -o submission.tar.gz
```

Upload the generated archive directly to Kaggle. The archive is ignored by
Git and should not be committed.

## License

This project is released under the MIT License. Competition data and
competition-specific materials remain subject to the Kaggle competition rules
and are not included here.
