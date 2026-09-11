"""Entry point. Run: python main.py  (--refresh rebuilds data/cards.csv, --seed N,
--fitness model to optimise against the learned matchup model)."""

from __future__ import annotations

import argparse
import sys

from optimizer import config, fitness
from optimizer.cr_api import load_card_pool
from optimizer.ga import GeneticAlgorithm


def fmt_fitness(value: float, kind: str) -> str:
    """Model fitness is an expected win rate; the heuristic is a 0-1 score."""
    return f"{value:.1%}" if kind == "model" else f"{value:.4f}"


def print_deck(deck, fitness: float, kind: str = "heuristic") -> None:
    label = "expected win rate" if kind == "model" else "fitness"
    print(f"\n=== Best deck  ({label} {fmt_fitness(fitness, kind)}) ===")
    for card in sorted(deck.cards, key=lambda c: (c.elixir, c.name)):
        if card.is_champion:
            tag = "CHAMPION"
        else:
            tag = {"evo": "EVO", "hero": "HERO"}.get(deck.form_of(card), "")
        suffix = f"   [{tag}]" if tag else ""
        print(f"   {card.elixir}  {card.name}{suffix}")
    print(f"   ---")
    print(f"   avg elixir : {deck.avg_elixir:.2f}")
    print(
        f"   evolutions : {len(deck.evolved_cards)}   heroes: {len(deck.hero_cards)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clash Royale deck optimizer")
    parser.add_argument(
        "--refresh", action="store_true",
        help="rebuild data/cards.csv from scratch (CR API + wiki scrape; see optimizer/build_dataset.py)"
    )
    parser.add_argument("--generations", type=int, default=config.GENERATIONS)
    parser.add_argument("--population", type=int, default=config.POPULATION_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--fitness", choices=fitness.CHOICES, default=config.FITNESS,
        help="'heuristic' (heuristic.py) or 'model' (learned matchup model: "
             "expected win rate vs the meta; needs data/matchup_model.pt)",
    )
    args = parser.parse_args()

    pool = load_card_pool(refresh=args.refresh)
    print(
        f"Loaded {len(pool)} cards "
        f"({len(pool.champion_ids)} champions, {len(pool.evolvable_ids)} with evolutions)."
    )

    try:
        score = fitness.make_fitness(args.fitness)
    except RuntimeError as exc:
        sys.exit(f"--fitness model: {exc}")
    if args.fitness == "model":
        info = fitness.describe("model")
        print(
            f"Fitness: learned matchup model on {info.get('device')} "
            f"({info.get('meta_decks')} meta decks, val log-loss "
            f"{info.get('val_logloss', float('nan')):.3f}"
            + (", SYNTHETIC training data" if info.get("synthetic") else "") + ")."
        )
    else:
        print("Fitness: heuristic.py")

    ga = GeneticAlgorithm(
        pool,
        score,
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
    )

    last = args.generations - 1

    def report(gen: int, population) -> None:
        if gen % 10 == 0 or gen == last:
            best_now = fmt_fitness(ga.fitness(population[0]), args.fitness)
            print(f"   gen {gen:4d} | best fitness {best_now}")

    print(f"\nEvolving {args.population} decks over {args.generations} generations...")
    best = ga.run(on_generation=report)

    ok, reason = best.is_valid()
    if not ok:
        print(f"\nWARNING: produced an invalid deck ({reason}).")
    print_deck(best, ga.fitness(best), args.fitness)


if __name__ == "__main__":
    main()
