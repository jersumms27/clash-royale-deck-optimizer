"""Local web UI for the Clash Royale deck optimizer.

Run:   python UI/server.py     (works from any directory)
Then:  your browser opens automatically at the printed URL.

This is a UI-ONLY layer. It imports the logic modules in optimizer/
(config / cr_api / ga / models) read-only and never modifies them. Which
fitness function rates a deck is decided in UI/scorers.py: the learned matchup
model when it's importable, the heuristic otherwise (or on request).
Standard library only -- no extra installs, no build step.

Endpoints (all GET, all JSON unless noted):
  /api/config                 deck rules, GA defaults, the scorer list
  /api/scorers?rescan=1       re-probe for the learned model without a restart
  /api/scorer_info?scorer=..  the model's own info() (device, meta size, metrics)
  /api/optimize?...           Server-Sent Events stream of the GA run
  /api/evaluate?cards=..      score one deck (+ its matchups vs the meta decks)
  /api/matchup?a=..&b=..      P(deck A beats deck B), both ways
  /api/meta                   the meta deck set the fitness averages over
Decks are passed as  <p>=id,id,...  <p>_evo=id,...  <p>_hero=id,...
"""

from __future__ import annotations

import csv
import json
import math
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# --- Make the optimizer package importable when run as `python UI/server.py`. ---
UI_DIR = Path(__file__).resolve().parent
ROOT = UI_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UI_DIR))

from optimizer import config  # noqa: E402  (import after sys.path tweak, on purpose)
from optimizer.cr_api import load_card_pool  # noqa: E402
from optimizer.ga import GeneticAlgorithm  # noqa: E402
from optimizer.models import CardPool, Deck  # noqa: E402

import scorers  # noqa: E402  (UI/scorers.py)

CARD_ATTRS_CSV = config.CARD_ATTRIBUTES_CSV

# Static assets we serve out of the UI/ folder.
_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}

# Safety rails so the UI sliders can never ask the engine for something absurd.
POP_MIN, POP_MAX = 2, 10000
GEN_MIN, GEN_MAX = 1, 10000


class ClientGone(Exception):
    """Raised inside the GA callback once the browser closed the stream."""


class BadRequest(ValueError):
    """A malformed deck / unknown scorer; reported to the client as 400."""


# --------------------------------------------------------------------------- #
# Data helpers (all read-only; never touch the logic modules' files)          #
# --------------------------------------------------------------------------- #
_pool_cache: dict[str, object] = {}
_pool_lock = threading.Lock()


def get_pool() -> CardPool:
    """The card pool, re-read only when cards.csv changes on disk."""
    try:
        stamp = config.CARDS_CSV.stat().st_mtime_ns
    except OSError:
        stamp = None
    with _pool_lock:
        if _pool_cache.get("stamp") != stamp or "pool" not in _pool_cache:
            _pool_cache["pool"] = load_card_pool()
            _pool_cache["stamp"] = stamp
        return _pool_cache["pool"]  # type: ignore[return-value]


def load_card_attributes() -> dict[str, dict]:
    """name -> {stat: value} parsed from card_attributes.csv (blanks dropped)."""
    if not CARD_ATTRS_CSV.exists():
        return {}
    out: dict[str, dict] = {}
    with open(CARD_ATTRS_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("name") or "").strip()
            if not name:
                continue
            stats: dict[str, object] = {}
            for key, value in row.items():
                if key == "name" or value is None or value == "":
                    continue
                try:
                    num = float(value)
                    stats[key] = int(num) if num.is_integer() else num
                except ValueError:
                    stats[key] = value
            out[name] = stats
    return out


def cards_payload() -> list[dict]:
    """Every card in the pool + its merged stats, for the Card Pool browser."""
    pool = get_pool()
    attrs = load_card_attributes()
    cards = []
    for card in sorted(pool.cards, key=lambda c: (c.elixir, c.name)):
        cards.append(
            {
                "id": card.id,
                "name": card.name,
                "elixir": card.elixir,
                "rarity": card.rarity,
                "type": card.type,
                "has_evolution": card.has_evolution,
                "is_champion": card.is_champion,
                "is_champion_hero": card.is_champion_hero,
                "stats": attrs.get(card.name, {}),
            }
        )
    return cards


def scorers_payload(rescan: bool = False) -> dict:
    listed = scorers.list_scorers(rescan=rescan)
    default = scorers.default_scorer()
    return {
        "scorers": [s.to_json() for s in listed],
        "default_scorer": default.id if default else None,
    }


def config_payload() -> dict:
    """Deck rules + GA defaults so the UI matches config.py exactly."""
    return {
        "deck_size": config.DECK_SIZE,
        "max_evolutions": config.MAX_EVOLUTIONS,
        "max_champions": config.MAX_CHAMPIONS,
        # The slot model, so the deck builder can mirror config.slots_ok().
        "slots": {
            "base_evolution": config.BASE_EVOLUTION_SLOTS,
            "base_champion": config.BASE_CHAMPION_SLOTS,
            "wild": config.WILD_SLOTS,
        },
        "defaults": {
            "population": config.POPULATION_SIZE,
            "generations": config.GENERATIONS,
        },
        "limits": {
            "population": [POP_MIN, POP_MAX],
            "generations": [GEN_MIN, GEN_MAX],
        },
        **scorers_payload(),
    }


def _num(x) -> float | None:
    """JSON can't carry NaN/inf; the UI treats null as 'no value'."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def deck_payload(deck: Deck, fitness=None, kind: str | None = None) -> dict:
    """Serialize a Deck for the client (cards sorted like main.print_deck)."""
    ok, reason = deck.is_valid()
    cards = []
    for card in sorted(deck.cards, key=lambda c: (c.elixir, c.name)):
        cards.append(
            {
                "id": card.id,
                "name": card.name,
                "elixir": card.elixir,
                "rarity": card.rarity,
                "type": card.type,
                "is_champion": card.is_champion,
                "has_evolution": card.has_evolution,
                "is_champion_hero": card.is_champion_hero,
                # form: "evo", "hero", or "base" -- decides the badge/slot in the UI.
                "form": deck.form_of(card),
                "is_evolved": card.id in deck.evolved,
                "is_hero": card.id in deck.hero,
            }
        )
    payload = {
        "cards": cards,
        "avg_elixir": round(deck.avg_elixir, 2),
        "num_evolutions": len(deck.evolved_cards),
        "num_heroes": len(deck.hero_cards),
        "num_champions": len(deck.champions),
        "valid": ok,
        "valid_reason": reason,
    }
    if fitness is not None:
        payload["fitness"] = _num(fitness)
        payload["fitness_kind"] = kind
    return payload


def matchups_payload(scorer: scorers.Scorer, deck: Deck) -> list[dict] | None:
    """P(deck beats each meta deck), with the meta decks' usage shares."""
    if scorer.matchups is None:
        return None
    rows = scorer.matchups(deck)
    shares = scorers.normalized_shares([m.meta for m in rows])
    out = []
    for m, share in zip(rows, shares):
        out.append(
            {
                "name": m.meta.name,
                "weight": _num(m.meta.weight),
                "share": share,
                "p_win": _num(m.p_win),
                "deck": deck_payload(m.meta.deck),
            }
        )
    out.sort(key=lambda r: -r["share"])
    return out


def meta_payload(scorer: scorers.Scorer) -> list[dict] | None:
    if scorer.meta is None:
        return None
    metas = scorer.meta()
    shares = scorers.normalized_shares(metas)
    rows = [
        {"name": m.name, "weight": _num(m.weight), "share": s, "deck": deck_payload(m.deck)}
        for m, s in zip(metas, shares)
    ]
    rows.sort(key=lambda r: -r["share"])
    return rows


def _jsonable(obj):
    """Keep only what JSON can carry (the model's info() may hold anything)."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, int, str)) or obj is None:
        return obj
    if isinstance(obj, float):
        return _num(obj)
    return str(obj)


def _clamp_int(raw, default: int, lo: int, hi: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def _qs_ids(qs: dict, key: str) -> list[int]:
    raw = (qs.get(key, [""])[0] or "").strip()
    if not raw:
        return []
    try:
        return [int(tok) for tok in raw.split(",") if tok.strip()]
    except ValueError:
        raise BadRequest(f"'{key}' must be a comma-separated list of card ids") from None


def parse_deck(qs: dict, prefix: str, pool: CardPool) -> Deck:
    """Build a Deck from  <prefix>=ids  <prefix>_evo=ids  <prefix>_hero=ids.

    Champions are folded into the hero set automatically (they have no base
    form). Only *structural* problems are rejected here: size, duplicates,
    unknown ids, one card claiming both forms. The slot budget and form
    eligibility are deliberately not enforced: decks in the meta set come from
    real (or synthetic) battles and may predate the current slot rules or a
    refreshed cards.csv, and the model scores them fine. The payload's
    `valid` / `valid_reason` (from Deck.is_valid) tells the UI to flag them.
    """
    ids = _qs_ids(qs, prefix)
    if not ids:
        raise BadRequest(f"deck '{prefix}' is empty")
    if len(ids) != config.DECK_SIZE:
        raise BadRequest(f"deck '{prefix}' has {len(ids)} cards, expected {config.DECK_SIZE}")
    if len(set(ids)) != len(ids):
        raise BadRequest(f"deck '{prefix}' has duplicate cards")
    unknown = [i for i in ids if i not in pool.by_id]
    if unknown:
        raise BadRequest(f"deck '{prefix}': unknown card id(s) {unknown}")
    in_deck = set(ids)
    evo = {i for i in _qs_ids(qs, prefix + "_evo") if i in in_deck}
    hero = {i for i in _qs_ids(qs, prefix + "_hero") if i in in_deck}
    hero |= {i for i in ids if i in pool.champion_set}
    if evo & hero:
        both = pool.get(next(iter(evo & hero))).name
        raise BadRequest(f"deck '{prefix}': {both} can't be both evolved and hero")
    return Deck(cards=tuple(pool.get(i) for i in ids), evolved=frozenset(evo), hero=frozenset(hero))


def resolve_scorer(qs: dict, need: str = "score") -> scorers.Scorer:
    """The scorer named in ?scorer= (or the default), checked for `need`."""
    wanted = (qs.get("scorer", [""])[0] or "").strip() or None
    scorer = scorers.get_scorer(wanted)
    if scorer is None:
        raise BadRequest(
            f"unknown scorer '{wanted}'" if wanted else "no scorer is available"
        )
    if not scorer.available:
        raise BadRequest(f"{scorer.label} is unavailable: {scorer.reason}")
    if getattr(scorer, need) is None:
        raise BadRequest(f"{scorer.label} doesn't support '{need}'")
    return scorer


# --------------------------------------------------------------------------- #
# HTTP handler                                                                 #
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "CRDeckUI/1.1"

    def log_message(self, *args) -> None:  # keep the console quiet
        pass

    # -- routing ----------------------------------------------------------- #
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send_file(UI_DIR / "index.html")
            elif path == "/api/config":
                self._send_json(config_payload())
            elif path == "/api/scorers":
                self._send_json(scorers_payload(rescan=qs.get("rescan", ["0"])[0] == "1"))
            elif path == "/api/scorer_info":
                scorer = resolve_scorer(qs, need="info")
                self._send_json({"scorer": scorer.id, "info": _jsonable(scorer.info())})
            elif path == "/api/cards":
                self._send_json(cards_payload())
            elif path == "/api/optimize":
                self._optimize(qs)
            elif path == "/api/evaluate":
                self._evaluate(qs)
            elif path == "/api/matchup":
                self._matchup(qs)
            elif path == "/api/meta":
                self._meta(qs)
            else:
                self._send_static(path)
        except BadRequest as exc:
            self._send_json({"error": str(exc)}, status=400)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-response
        except Exception as exc:  # never let one bad request kill the server
            try:
                if path.startswith("/api/"):
                    self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
                else:
                    self.send_error(500, "Server error", str(exc))
            except OSError:
                pass

    # -- static files ------------------------------------------------------ #
    def _send_static(self, path: str) -> None:
        target = (UI_DIR / path.lstrip("/")).resolve()
        try:
            target.relative_to(UI_DIR)  # block path traversal
        except ValueError:
            self.send_error(403, "Forbidden")
            return
        if not target.is_file() or target.suffix.lower() == ".py":
            self.send_error(404, "Not found")
            return
        self._send_file(target)

    def _send_file(self, target: Path) -> None:
        data = target.read_bytes()
        ctype = _MIME.get(target.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, status: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # -- evaluate / matchup / meta ---------------------------------------- #
    def _evaluate(self, qs: dict) -> None:
        scorer = resolve_scorer(qs)
        deck = parse_deck(qs, "cards", get_pool())
        payload = deck_payload(deck, scorer.score(deck), scorer.kind)
        payload["scorer"] = scorer.id
        payload["matchups"] = matchups_payload(scorer, deck)
        self._send_json(payload)

    def _matchup(self, qs: dict) -> None:
        scorer = resolve_scorer(qs, need="predict")
        pool = get_pool()
        a = parse_deck(qs, "a", pool)
        b = parse_deck(qs, "b", pool)
        p_ab = _num(scorer.predict(a, b))
        p_ba = _num(scorer.predict(b, a))  # reverse order: a symmetry check for the model
        fit_a = scorer.score(a) if scorer.score else None
        fit_b = scorer.score(b) if scorer.score else None
        self._send_json(
            {
                "scorer": scorer.id,
                "kind": scorer.kind,
                "p_ab": p_ab,
                "p_ba": p_ba,
                "a": deck_payload(a, fit_a, scorer.kind),
                "b": deck_payload(b, fit_b, scorer.kind),
            }
        )

    def _meta(self, qs: dict) -> None:
        scorer = resolve_scorer(qs, need="meta")
        self._send_json({"scorer": scorer.id, "kind": scorer.kind, "decks": meta_payload(scorer)})

    # -- optimize (Server-Sent Events) ------------------------------------- #
    def _optimize(self, qs: dict) -> None:
        population = _clamp_int(qs.get("population", [None])[0],
                                config.POPULATION_SIZE, POP_MIN, POP_MAX)
        generations = _clamp_int(qs.get("generations", [None])[0],
                                 config.GENERATIONS, GEN_MIN, GEN_MAX)
        seed_raw = (qs.get("seed", [""])[0] or "").strip()
        try:
            seed = int(seed_raw) if seed_raw else None
        except ValueError:
            seed = None

        # HTTP/1.0-style stream: no Content-Length, body runs until we close.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        alive = {"ok": True}

        def emit(event: str, payload: dict) -> None:
            if not alive["ok"]:
                return
            try:
                msg = f"event: {event}\ndata: {json.dumps(payload)}\n\n"
                self.wfile.write(msg.encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                alive["ok"] = False  # client closed the EventSource

        try:
            scorer = resolve_scorer(qs)
        except BadRequest as exc:
            emit("failed", {"message": str(exc)})
            return

        emit("start", {"total": generations, "population": population,
                       "generations": generations, "seed": seed,
                       "scorer": scorer.to_json()})

        pool = get_pool()
        ga = GeneticAlgorithm(
            pool, scorer.score,
            population_size=population,
            generations=generations,
            seed=seed,
        )

        last_key = {"value": None}

        def on_generation(gen: int, ranked) -> None:
            # ranked is already sorted best-first (ga.run sorts before callback);
            # all fitnesses are cached from that sort, so this is just lookups.
            fits = [ga.fitness(d) for d in ranked]
            best_deck = ranked[0]
            payload = {
                "gen": gen + 1,
                "total": generations,
                "best_fitness": _num(fits[0]),
                "avg_fitness": _num(sum(fits) / len(fits)),
                "worst_fitness": _num(fits[-1]),
                "diversity": len({d.key for d in ranked}),
                "pop_size": len(ranked),
                "best_avg_elixir": round(best_deck.avg_elixir, 2),
            }
            # Ship the full deck only when the best actually changes (saves bytes
            # and lets the UI animate real mutations instead of every tick).
            if best_deck.key != last_key["value"]:
                payload["deck"] = deck_payload(best_deck, fits[0], scorer.kind)
                last_key["value"] = best_deck.key
            emit("progress", payload)
            if not alive["ok"]:
                # The browser hit Stop (or left). Don't burn the GPU on a run
                # nobody is watching.
                raise ClientGone()

        try:
            best = ga.run(on_generation=on_generation)
        except ClientGone:
            return
        except Exception as exc:  # surface engine errors to the UI, don't crash
            emit("failed", {"message": f"{type(exc).__name__}: {exc}"})
            return

        final = deck_payload(best, ga.fitness(best), scorer.kind)
        final["scorer"] = scorer.id
        try:
            final["matchups"] = matchups_payload(scorer, best)
        except Exception as exc:  # a breakdown failure shouldn't hide the deck
            final["matchups"] = None
            final["matchups_error"] = f"{type(exc).__name__}: {exc}"
        emit("done", final)


# --------------------------------------------------------------------------- #
# Server bootstrap                                                             #
# --------------------------------------------------------------------------- #
def make_server(preferred: int = 8000) -> tuple[ThreadingHTTPServer, int]:
    last_err: OSError | None = None
    for port in range(preferred, preferred + 25):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            httpd.daemon_threads = True
            return httpd, port
        except OSError as exc:
            last_err = exc
    raise last_err if last_err else OSError("no free port found")


def main() -> None:
    if not (UI_DIR / "index.html").exists():
        sys.exit(f"index.html not found next to server.py (looked in {UI_DIR}).")

    httpd, port = make_server()
    url = f"http://localhost:{port}"
    print("\n  Clash Royale Deck Optimizer  -  web UI")
    print(f"  -> {url}")
    for s in scorers.list_scorers():
        state = "ready" if s.available else f"unavailable ({s.reason})"
        print(f"  scorer {s.id:<10} {state}")
    print("  Press Ctrl+C to stop.\n")
    # Open the browser a beat after the server starts listening.
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down. Bye!")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    main()
