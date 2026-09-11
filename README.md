# Deck Optimizer — Web UI

A small, good-looking local web UI for the Clash Royale deck optimizer. It runs
entirely on your machine, opens in your browser, and never changes any of the
optimizer logic — it only *reads* from it.

## Run it

From the project root:

```bash
python UI/server.py
```

That's it. The terminal prints a local address (e.g. `http://localhost:8000`)
and your browser opens automatically. Press **Ctrl+C** in the terminal to stop.

> Uses only the Python standard library — nothing to install. If you use the
> conda env, run it with that interpreter, e.g.
> `C:/Users/jerem/miniconda3/envs/hackathon/python.exe UI/server.py`.

## What you can do

- **Optimizer tab** — set population, generations and (optionally) a random
  seed, then hit **Optimize Deck**. Watch the generations tick by live, then see
  the evolved 8-card deck with elixir, evolution, hero and champion badges plus deck
  stats. Use the same seed to reproduce a run.
- **Card Pool tab** — browse and search all cards; filter by rarity, type and
  elixir; click a card to see its combat stats.

## Notes

- Decks are scored by `optimizer/heuristic.py`: a weighted average of average
  elixir, air troops, buildings, spells, total HP, total DPS and win conditions.
  Tweak its weights and targets, then re-run from the UI — no UI changes needed.
- Card artwork is pulled from the public community asset repo
  [`cr-api-assets`](https://github.com/RoyaleAPI/cr-api-assets). If you're
  offline or an image is missing, the card falls back to a clean styled tile.

## Project layout

```
main.py              CLI entry point: python main.py
optimizer/           the optimizer logic
  config.py            paths, deck rules, GA settings
  models.py            Card, CardPool, Deck
  cr_api.py            fetch cards from the API / load them from data/cards.csv
  ga.py                genetic algorithm
  heuristic.py         deck scoring
  dev_sample.py        offline sample data: python -m optimizer.dev_sample
UI/                  web UI: python UI/server.py
  server.py            tiny local HTTP server; serves the page + JSON/stream APIs
  index.html           page structure
  style.css            the Clash Royale theme
  app.js               front-end logic (runs the optimizer, renders cards, filters)
data/                cards.csv (source of truth), card_attributes.csv, scrape cache
notebooks/           fetch_cards.ipynb, scrape.ipynb (rebuild data/cards.csv)
token.txt            your API token (gitignored; see token.txt.example)
```
