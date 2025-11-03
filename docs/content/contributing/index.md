# Contributing

Development and documentation contribution guidelines.

## Environment Setup

- **Python**: create a virtual environment and install the package in editable mode: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]`
- **Node**: install dependencies inside the docs workspace with `npm --prefix docs install`
- **Tooling**: run `pre-commit run --all-files`, `pytest`, and `npm --prefix docs run build` before opening a PR

## Workflow Overview

1. Branch from `main` and keep changes focused (docs versus code versus tooling)
2. Run linters/tests locally before pushing
3. Open a PR with a clear summary, test notes, and any follow-up TODOs
4. Address CI feedback—red checks block review

## Documentation Specifics

- Preview docs locally: `npm --prefix docs run start`
- Regenerate API markdown when docstrings change: `npm --prefix docs run generate:api`
- Add new guides under `docs/content/` and update `docs/sidebars.js`

## Commit & Review Etiquette

- Prefer `area: short summary` commit messages (for example `docs: add forecasting tutorial`)
- Avoid committing generated artefacts (for example `docs/content/api/generated/*`) unless they are intended changes
- Rebase (don’t merge) when syncing from `main`
- Respond to every review comment; clarify disagreements rather than ignoring them

## Getting Help

- Open a draft PR early for directional feedback
- Use GitHub Issues/Discussions for larger proposals

## NXAI Contributor License Agreement

Read the full CLA for Individual Contributors here: [CLA](https://github.com/NX-AI/CLA/blob/main/CLA.md)

### Contact
If you have any question about the CLA, feel free to reach out to [contact@nx-ai.com](mailto:contact@nx-ai.com)
