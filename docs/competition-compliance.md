# Public repository and competition compliance

This project is published as source code for a Kaggle competition. Before making
the repository public, review the current competition page and terms:

<https://www.kaggle.com/competitions/kaggriculture>

## Publication boundary

The public repository may contain:

- original agent source code;
- tests and small synthetic fixtures;
- dependency declarations;
- scripts that operate on a locally installed Kaggle environment;
- documentation that describes the implementation at a high level;
- an OSI-approved license for original work.

The repository must not contain:

- Kaggle credentials, API tokens, `.env` files, or local machine settings;
- competition replay files, downloaded competition data, leaderboard exports,
  opponent traces, or derived datasets;
- notebooks with embedded replay/data outputs;
- generated submission archives or Python caches;
- internal agent handoff logs or private research reports.

The competition-specific rules state that competition data must not be
transmitted, duplicated, published, redistributed, or made available to people
who have not agreed to the rules. A public Git repository is therefore not an
appropriate place for replay files or data-derived artifacts, even when the
competition page itself is public.

## Fresh-clone verification

After cloning, install dependencies and run the tests:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
```

To create a Kaggle submission, run `python scripts/bundle.py`. Do not commit
the resulting archive; upload it directly to Kaggle.

## Before publishing

1. Check `git status --short` and inspect every staged file.
2. Confirm that no `kaggle.json`, `.env`, token, replay, leaderboard export,
   notebook output, or archive is staged.
3. Confirm that the current Kaggle rules still permit the intended use of any
   external data, models, or tools.
4. If a previous Git history contained a secret or restricted data, remove it
   from history or publish a new clean repository; deleting it from the latest
   commit is not sufficient.
