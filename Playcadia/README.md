# Playcadia — OpenAI enrichment for US listings (English)

## Requirements

- Python 3.10+
- Environment variable: `OPENAI_API_KEY` (never commit it).

```bash
pip install -r requirements.txt
```

## Files

| File | Description |
|------|-------------|
| `games.csv` | **Input** (you maintain it): columns `id`, `console-name`, `product-name`. UTF-8 with BOM is supported. |
| `results.csv` | **Output**: valid rows only — `id`, `description`, `recommended_age_group`, `game_style` (English copy for US marketplace). |
| `pending.csv` | Same schema as `games.csv`: rows not yet completed in `results.csv`. |
| `enrich_errors.log` | Append-only log of failed IDs (timestamp, id, message). Use this instead of scrolling the notebook. |
| `last_run_report.json` | Last aggregate counts and paths. |
| `enrich_games.ipynb` | Notebook to run. |

## Large lists (~40k+)

- Progress prints **one line every `PROGRESS_EVERY` rows** (default 250), not per title.
- Per-title success is silent; failures go to **`enrich_errors.log`** (and a short sample is printed at the end of the run).
- `load_completed_ids_from_results` uses vectorized pandas checks so restarts stay fast.

## Flow

1. Ensure `games.csv` is present under `Playcadia/` (or repo root — see `resolve_data_dir` in the notebook).
2. Run all cells.
3. Read `last_run_report.json` and/or the printed JSON for totals vs pending.
4. **Re-run:** keep the same `games.csv`; IDs already valid in `results.csv` are skipped. Or point `GAMES_CSV` at `pending.csv` if you prefer a minimal input file.

Interrupted runs: rows already appended to `results.csv` are kept; the next run only processes remaining IDs.

## Git

Generated files (`results.csv`, `pending.csv`, logs, report) are listed in the repo `.gitignore`. Add `Playcadia/games.csv` to `.gitignore` locally if the catalog is large and should not be pushed.
