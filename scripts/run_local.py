"""
Local simulation harness for Kaggriculture.

Run the agent against built-in opponents locally and measure performance.

Usage:
    python scripts/run_local.py                         # 5 games vs random
    python scripts/run_local.py --games 20              # 20 games vs random
    python scripts/run_local.py --opponent starter      # vs starter agent
    python scripts/run_local.py --seat 1                # play as player 1
    python scripts/run_local.py --detailed              # show day-by-day stats
"""

import sys
import os
import argparse
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def count_farm_structures(farm):
    coops, pastures, animals, plants, empty = 0, 0, 0, 0, 0
    for row in farm.get("tiles", []):
        for t in row:
            if t is None:
                empty += 1
            elif isinstance(t, dict):
                k = t.get("kind", "")
                if k == "PLANT" or "crop" in t:
                    plants += 1
                elif k == "COOP" or t.get("structure") == "COOP":
                    coops += 1
                elif k == "PASTURE" or t.get("structure") == "PASTURE":
                    pastures += 1
                if "animal" in t or k in ("COW", "SHEEP", "GOOSE"):
                    animals += 1
    return coops, pastures, animals, plants, empty


def run_single_episode(opponent: str, agent_seat: int = 0, detailed: bool = False):
    """Run a single 720-turn episode and print day-by-day stats."""
    from kaggle_environments import make
    from agent.main import agent

    print(f"\n{'='*70}")
    print(f"MATCH: Agent (Seat {agent_seat}) vs '{opponent}' (Seat {1 - agent_seat}) | 720 turns (30 days)")
    print(f"{'='*70}")

    agents = [agent, opponent] if agent_seat == 0 else [opponent, agent]
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=True)

    start_time = time.time()
    env.run(agents)
    elapsed = time.time() - start_time

    # Day-by-day logs
    print(f"\n--- Day-by-Day Tracking (Seat {agent_seat}) ---")
    print(f"{'Day':4s} | {'Money':>8s} | {'Hands':>5s} | {'Plants':>6s} | {'Empty':>5s} | {'Shed Items':>10s} | {'Pastures':>8s} | {'Coops':>5s} | {'Animals':>7s}")
    print("-" * 79)

    seen_days = set()
    for step_idx, step in enumerate(env.steps):
        obs = step[0].observation
        day = obs.get("day", step_idx // 24)
        hour = obs.get("hour", step_idx % 24)
        
        if hour == 23 and day not in seen_days:
            seen_days.add(day)
            my_farm = obs["farms"][agent_seat]
            my_priv = step[agent_seat].observation.get("private", {}) or {}
            shed = my_priv.get("shed", {})
            total_shed = sum(shed.values()) if isinstance(shed, dict) else 0
            coops, pastures, animals, plants, empty = count_farm_structures(my_farm)
            money = my_farm.get("money", 0)
            hands_count = len(my_farm.get("hands", []))

            print(f"D{day:02d}  | {money:8.0f} | {hands_count:5d} | {plants:6d} | {empty:5d} | {total_shed:10d} | {pastures:8d} | {coops:5d} | {animals:7d}")

    # Final summary
    final_step = env.steps[-1]
    my_res = final_step[agent_seat]
    opp_res = final_step[1 - agent_seat]
    my_farm = final_step[0].observation["farms"][agent_seat]
    opp_farm = final_step[0].observation["farms"][1 - agent_seat]

    c_my, p_my, a_my, pl_my, e_my = count_farm_structures(my_farm)
    c_opp, p_opp, a_opp, pl_opp, e_opp = count_farm_structures(opp_farm)

    my_reward = my_res.reward if my_res.reward is not None else 0
    opp_reward = opp_res.reward if opp_res.reward is not None else 0

    print(f"\n{'='*70}")
    print(f"EPISODE RESULT:")
    print(f"  Agent (Seat {agent_seat}):  Bank=${my_reward:,.0f} | Status={my_res.status} | Plants={pl_my} | Pastures={p_my} | Coops={c_my} | Animals={a_my}")
    print(f"  Opponent ('{opponent}'): Bank=${opp_reward:,.0f} | Status={opp_res.status} | Plants={pl_opp} | Pastures={p_opp} | Coops={c_opp} | Animals={a_opp}")
    print(f"  Margin:               {my_reward - opp_reward:+,.0f}")
    print(f"  Elapsed Time:         {elapsed:.2f}s")
    print(f"{'='*70}\n")

    return {
        "agent_seat": agent_seat,
        "opponent": opponent,
        "agent_bank": my_reward,
        "opp_bank": opp_reward,
        "agent_status": my_res.status,
        "opp_status": opp_res.status,
        "plants": pl_my,
        "pastures": p_my,
        "coops": c_my,
        "animals": a_my,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Kaggriculture agent locally")
    parser.add_argument("--games", "-n", type=int, default=1, help="Number of games to run")
    parser.add_argument("--opponent", "-o", type=str, default="random",
                        choices=["random", "starter", "pass", "self"],
                        help="Opponent to play against")
    parser.add_argument("--seat", "-s", type=int, default=0, choices=[0, 1],
                        help="Seat for our agent (0 or 1)")
    parser.add_argument("--detailed", "-d", action="store_true", default=True,
                        help="Show day-by-day details")
    args = parser.parse_args()

    run_single_episode(args.opponent, args.seat, args.detailed)


if __name__ == "__main__":
    main()

