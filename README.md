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

- **Optimizer tab** — pick how decks are scored (the **learned matchup model**
  or the **heuristic** baseline), set population, generations and (optionally) a
  random seed, then hit **Optimize Deck**. Watch the generations tick by live
  (or **Stop** early and keep the best deck so far), then see the evolved 8-card
  deck with elixir, evolution, hero and champion badges plus deck stats. With
  the model selected, fitness is shown as an **expected win rate** and a
  **Matchups vs the meta** panel breaks it down into P(win) against every meta
  deck. Use the same seed to reproduce a run.
- **Matchups tab** — head to head: build two decks (click a slot to add a card,
  click the form pill to cycle base / EVO / HERO), load the optimizer's best deck
  or any meta deck into either side, and the model tells you **P(A beats B)**
  plus each deck's expected win rate vs the meta. The meta deck set the fitness
  averages over is listed below with its usage weights.
- **Card Pool tab** — browse and search all cards; filter by rarity, type and
  elixir; click a card to see its combat stats or add it to Deck A / B.

## Notes

- Two scorers are offered. **Learned matchup model** (`optimizer/matchup.py`):
  a neural net trained on real 1v1 battles that predicts P(deck A beats deck B);
  fitness is the usage-weighted expected win rate against `data/meta_decks.csv`.
  **Heuristic** (`optimizer/heuristic.py`): the hand-tuned baseline. The UI
  discovers them through `UI/scorers.py`; if the model can't load (no
  checkpoint yet, PyTorch missing) it's listed as unavailable with the reason and
  the heuristic is used. After training a model, hit **↻ rescan** — no restart.
- The scorer contract is duck-typed and documented at the top of
  `UI/scorers.py`: `score(deck)` is all the optimizer needs; `predict(a, b)`,
  `meta_decks()`, `matchups(deck)` and `info()` light up the matchup features.
- Starting a run streams progress over Server-Sent Events; closing the tab or
  pressing Stop aborts the run on the server too, so the GPU isn't left busy.
- When Supercell releases new cards, rebuild the card data with
  `python -m optimizer.build_dataset` (or `python main.py --refresh`). It needs an
  API token in `token.txt` and the `hackathon` conda env (pandas/requests/lxml).
- Card artwork is pulled from the public community asset repo
  [`cr-api-assets`](https://github.com/RoyaleAPI/cr-api-assets). If you're
  offline or an image is missing, the card falls back to a clean styled tile.

## Collecting battle data

The matchup model trains on `data/battles.csv` — one row per real 1v1 battle
with both decks and who won (schema in `data/README.md`). The `collector/`
package builds it from the official API's battle logs (standard library only;
needs `token.txt`):

```bash
python -m collector.probe                       # optional: inspect one battle log's fields
python -m collector.crawl                       # crawl until Ctrl+C (resumable; ~400k battles/hour)
python -m collector.crawl --max-minutes 120     # or a bounded run
python -m collector.flatten                     # raw logs -> data/battles.csv
python -m collector.stats                       # label balance, modes, unknown cards, ...
```

- `crawl` seeds from the Path of Legend rankings and snowballs through
  opponents, storing raw battle JSON in `data/battles/raw.jsonl` (gitignored)
  and its progress in `data/battles/state.json`. Re-running resumes; players are
  re-polled after `--recrawl-after` hours (default 6), so leaving it running or
  re-running it daily keeps the dataset growing. Run **one** crawler per data
  folder. Ctrl+C once stops cleanly after in-flight requests.
- `flatten` keeps normal-rules 1v1 modes (`pathOfLegend`, `trail` = ladder,
  `riverRacePvP` = clan-war 1v1) and drops event variants (rage, triple elixir,
  …). Which player is "A" is a deterministic coin flip, so labels are ~50/50 and
  re-running yields an identical file. `--types pathOfLegend` restricts to
  ranked; `--maxed-only` keeps only battles where all 16 cards are max level
  (see the `a_levels`/`b_levels` columns).
- Card forms come from the API's `evolutionLevel`: `1` = evolved, `2` = hero
  form (there is no separate hero field). Champions carry neither and are
  implicit heroes, so `a_hero`/`b_hero` list non-champions only and
  `hero_known` is `True`.
- Tokens are IP-locked. If your IP changes, either re-create the token or use
  RoyaleAPI's proxy: whitelist `45.79.218.79` and set
  `CR_API_BASE=https://proxy.royaleapi.dev/v1`.
- Offline tests: `python -m unittest tests.test_collector`.

## Training the matchup model

The learned scorer is a small Transformer over all 16 cards of a matchup:

```
card token  = Linear([learned embedding | stats from cards.csv]) + form (base/evo/hero) + side
encoder     = self-attention over both decks (a deck is a *set* — slot position is irrelevant)
logit(A,B)  = f(A,B) − f(B,A)          # antisymmetric: P(A,B) + P(B,A) = 1 exactly
fitness(D)  = Σ weight_i · P(D beats meta_deck_i)
```

1. `conda env update -f environment.yml --prune` — adds PyTorch (CUDA 12.8) and matplotlib.
2. Get `data/battles.csv` (section above).
3. Open `notebooks/train_model.ipynb` with the **Python (hackathon)** kernel and run it top
   to bottom. It loads the battles (time-ordered, last 10 % held out), writes
   `data/meta_decks.csv` (the top-300 most-played decks, usage-weighted — edit it by hand and
   set `REBUILD_META = False` to keep your edits; `config.META_DECK_COUNT` sets the size, and GA
   cost per generation grows linearly with it), trains with early stopping on validation
   log-loss (`EPOCHS` is only a cap), plots the learning curve and calibration, probes
   counters, and saves `data/matchup_model.pt` (the previous checkpoint is kept as
   `matchup_model.prev.pt`). The last section evolves the best deck against **one specific
   opponent** — type the deck as names (`Goblin Barrel*, Tombstone^, …`; `*` = evo, `^` = hero);
   from code that's `optimizer.matchup.fitness_against(...)` passed to `GeneticAlgorithm`.
   On the RTX 4050 it trains at ~35k battles/s, so a few hundred thousand battles take a
   couple of minutes. Without `battles.csv` it trains on synthetic data so the pipeline can
   be exercised — that model knows nothing about real Clash Royale.
4. `python main.py --fitness model` (or pick the model in the web UI; hit **↻ rescan** after
   retraining). Fitness is then the expected win rate against the meta.

Notes:
- The GA scores a whole generation in one batched GPU pass (`GeneticAlgorithm` looks for a
  `score_batch` on the fitness function) — ~200k matchups per generation at the defaults.
- Hero form is only modelled when `battles.csv` carries it (`hero_known`); otherwise hero
  cards are scored as base form, so the hero slot doesn't affect fitness until data exists.
- Cards released after training (not in the checkpoint's vocabulary) are scored as a neutral
  masked card; `main.py --fitness model` reports how many. Retrain to give them embeddings.
- `python -m unittest discover -s tests` covers the model (antisymmetry, permutation
  invariance, forms, learning a planted truth), the data files and the GA batch path.

## Project layout

```
main.py              CLI entry point: python main.py
optimizer/           the optimizer logic
  config.py            paths, deck rules, GA settings
  models.py            Card, CardPool, Deck
  cr_api.py            fetch cards from the API / read + write data/cards.csv
  build_dataset.py     rebuild data/cards.csv from scratch: python -m optimizer.build_dataset
  ga.py                genetic algorithm (batched fitness when the scorer supports it)
  heuristic.py         hand-tuned deck scoring (baseline)
  fitness.py           pick a fitness by name: "heuristic" | "model"
  model.py             MatchupModel (PyTorch): card features, P(A beats B), fit/evaluate
  learned_fitness.py   LearnedFitness: expected win rate vs the meta, batched on the GPU
  matchup.py           the model as plain functions (score / predict / meta_decks / matchups)
  battles.py           battles.csv loader + synthetic battle generator
  meta.py              meta_decks.csv: build from battles, load, save
  dev_sample.py        offline sample data: python -m optimizer.dev_sample
tests/               python -m unittest discover -s tests
UI/                  web UI: python UI/server.py
  server.py            tiny local HTTP server; serves the page + JSON/stream APIs
  scorers.py           scorer registry: finds the learned model / heuristic for the UI
  index.html           page structure
  style.css            the Clash Royale theme
  app.js               front-end logic (optimizer, matchups + deck builder, card pool)
data/                cards.csv (source of truth), battles.csv, meta_decks.csv, matchup_model.pt
                     (formats in data/README.md)
notebooks/           build_cards.ipynb (rebuild data/cards.csv), train_model.ipynb (train the model)
token.txt            your API token (gitignored; see token.txt.example)
```
