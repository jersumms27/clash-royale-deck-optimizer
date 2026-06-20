"""Entry point. Run: python main.py  (use --refresh to re-fetch cards, --seed N)."""

from __future__ import annotations

import argparse

import config
from cr_api import load_card_pool
from ga import GeneticAlgorithm
from heuristic import score


def print_deck(deck, fitness: float) -> None:
    print(f"\n=== Best deck  (fitness {fitness:.4f}) ===")
    for card in sorted(deck.cards, key=lambda c: (c.elixir, c.name)):
        tags = []
        if card.id in deck.evolved:
            tags.append("EVO")
        if card.is_champion:
            tags.append("CHAMPION")
        suffix = f"   [{', '.join(tags)}]" if tags else ""
        print(f"   {card.elixir}  {card.name}{suffix}")
    print(f"   ---")
    print(f"   avg elixir : {deck.avg_elixir:.2f}")
    print(
        f"   evolutions : {len(deck.evolved_cards)}   champions: {len(deck.champions)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Clash Royale deck optimizer")
    parser.add_argument(
        "--refresh", action="store_true", help="re-fetch card data from the CR API"
    )
    parser.add_argument("--generations", type=int, default=config.GENERATIONS)
    parser.add_argument("--population", type=int, default=config.POPULATION_SIZE)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    pool = load_card_pool(refresh=args.refresh)
    print(
        f"Loaded {len(pool)} cards "
        f"({len(pool.champion_ids)} champions, {len(pool.evolvable_ids)} with evolutions)."
    )

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
            print(f"   gen {gen:4d} | best fitness {ga.fitness(population[0]):.4f}")

    print(f"\nEvolving {args.population} decks over {args.generations} generations...")
    best = ga.run(on_generation=report)

    ok, reason = best.is_valid()
    if not ok:
        print(f"\nWARNING: produced an invalid deck ({reason}).")
    print_deck(best, ga.fitness(best))


if __name__ == "__main__":
    main()
