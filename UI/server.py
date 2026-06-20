"""Local web UI for the Clash Royale deck optimizer.

Run:   python UI/server.py     (works from any directory)
Then:  your browser opens automatically at the printed URL.

This is a UI-ONLY layer. It imports the existing logic modules
(config / cr_api / ga / heuristic / models) read-only and never modifies them.
Standard library only -- no extra installs, no build step.
"""

from __future__ import annotations

import csv
import json
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# --- Make the project-root logic modules importable without changing them. ----
UI_DIR = Path(__file__).resolve().parent
ROOT = UI_DIR.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402  (import after sys.path tweak, on purpose)
from cr_api import load_card_pool  # noqa: E402
from ga import GeneticAlgorithm  # noqa: E402
from heuristic import score  # noqa: E402

CARD_ATTRS_CSV = ROOT / "card_attributes.csv"

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


# --------------------------------------------------------------------------- #
# Data helpers (all read-only; never touch the logic modules' files)          #
# --------------------------------------------------------------------------- #
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
    pool = load_card_pool()
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


def config_payload() -> dict:
    """Deck rules + GA defaults so the UI matches config.py exactly."""
    return {
        "deck_size": config.DECK_SIZE,
        "max_evolutions": config.MAX_EVOLUTIONS,
        "max_champions": config.MAX_CHAMPIONS,
        "defaults": {
            "population": config.POPULATION_SIZE,
            "generations": config.GENERATIONS,
        },
        "limits": {
            "population": [POP_MIN, POP_MAX],
            "generations": [GEN_MIN, GEN_MAX],
        },
    }


def deck_payload(deck, fitness: float) -> dict:
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
                # form: "evo", "hero", or "base" -- decides the badge/slot in the UI.
                "form": deck.form_of(card),
                "is_evolved": card.id in deck.evolved,
                "is_hero": card.id in deck.hero,
            }
        )
    return {
        "cards": cards,
        "avg_elixir": round(deck.avg_elixir, 2),
        "num_evolutions": len(deck.evolved_cards),
        "num_heroes": len(deck.hero_cards),
        "num_champions": len(deck.champions),
        "fitness": fitness,
        "valid": ok,
        "valid_reason": reason,
    }


def _clamp_int(raw, default: int, lo: int, hi: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------- #
# HTTP handler                                                                 #
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "CRDeckUI/1.0"

    def log_message(self, *args) -> None:  # keep the console quiet
        pass

    # -- routing ----------------------------------------------------------- #
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path in ("/", "/index.html"):
                self._send_file(UI_DIR / "index.html")
            elif path == "/api/config":
                self._send_json(config_payload())
            elif path == "/api/cards":
                self._send_json(cards_payload())
            elif path == "/api/optimize":
                self._optimize(parse_qs(parsed.query))
            else:
                self._send_static(path)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-response
        except Exception as exc:  # never let one bad request kill the server
            try:
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
        if not target.is_file():
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

    def _send_json(self, obj) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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

        emit("start", {"total": generations, "population": population,
                       "generations": generations, "seed": seed})

        pool = load_card_pool()
        ga = GeneticAlgorithm(
            pool, score,
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
                "best_fitness": fits[0],
                "avg_fitness": sum(fits) / len(fits),
                "worst_fitness": fits[-1],
                "diversity": len({d.key for d in ranked}),
                "pop_size": len(ranked),
                "best_avg_elixir": round(best_deck.avg_elixir, 2),
            }
            # Ship the full deck only when the best actually changes (saves bytes
            # and lets the UI animate real mutations instead of every tick).
            if best_deck.key != last_key["value"]:
                payload["deck"] = deck_payload(best_deck, fits[0])
                last_key["value"] = best_deck.key
            emit("progress", payload)

        try:
            best = ga.run(on_generation=on_generation)
        except Exception as exc:  # surface engine errors to the UI, don't crash
            emit("failed", {"message": str(exc)})
            return

        emit("done", deck_payload(best, ga.fitness(best)))


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
