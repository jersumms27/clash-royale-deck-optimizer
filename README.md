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
  the evolved 8-card deck with elixir, evolution and champion badges plus deck
  stats. Use the same seed to reproduce a run.
- **Card Pool tab** — browse and search all cards; filter by rarity, type and
  elixir; click a card to see its combat stats.

## Notes

- The scoring heuristic in `heuristic.py` currently returns a flat score, so the
  produced deck is **valid but not yet ranked**. Once you fill in `heuristic.py`,
  re-run from the UI — no UI changes needed.
- Card artwork is pulled from the public community asset repo
  [`cr-api-assets`](https://github.com/RoyaleAPI/cr-api-assets). If you're
  offline or an image is missing, the card falls back to a clean styled tile.

## Files (all UI-only)

| File         | Purpose                                                        |
|--------------|----------------------------------------------------------------|
| `server.py`  | Tiny local HTTP server; serves the page + JSON/stream APIs.    |
| `index.html` | Page structure.                                                |
| `style.css`  | The Clash Royale theme.                                        |
| `app.js`     | Front-end logic (runs the optimizer, renders cards, filters).  |
