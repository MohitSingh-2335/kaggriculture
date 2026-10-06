"""
Reusable benchmark harness for Kaggriculture.

Runs N episodes headless against specified opponents with seed control,
alternating seats, writes raw CSV records to data/benchmarks/, and prints
aggregate statistics to stdout.

Supports multi-opponent benchmarking across distinct strategy lineages.

Usage:
    python scripts/benchmark.py --opponents suite -n 20 --label multiopponent
    python scripts/benchmark.py --opponent kaggriculture-structured-economic-policy -n 30 --label baseline
    python scripts/benchmark.py --opponent random -n 30 --label random_check
"""

import sys
import os
import argparse
import csv
import datetime
import statistics
import tempfile
import json
import ast
import re
import random
from typing import List, Union, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

from kaggle_environments import make
from kaggle_environments.agent import get_last_callable
from agent.main import agent, get_logged_exceptions, clear_logged_exceptions
from scripts.extract_strong_opponents import clean_source
from src.constants import ALL_ANIMALS, MARKET_PARAMS


# Canonical multi-opponent suite covering distinct strategy lineages + floor check
DEFAULT_OPPONENTS = [
    "random",
    "kaggriculture-structured-economic-policy",
    "kaggriculture-breaking-the-tie-2883-score",
    "kaggriculture-3000-socre",
    "kaggriculture-ultimate-mega-ensemble-3000",
]


def load_opponent_callable(opponent_arg: str):
    """Resolve opponent string to a callable or built-in opponent name."""
    if opponent_arg in ("random", "starter"):
        return opponent_arg

    # Check if opponent_arg is a direct file path
    nb_path = opponent_arg
    if not os.path.exists(nb_path):
        # Check notebooks/ directory
        candidates = [
            os.path.join(PROJECT_ROOT, "notebooks", opponent_arg),
            os.path.join(PROJECT_ROOT, "notebooks", f"{opponent_arg}.ipynb"),
        ]
        for c in candidates:
            if os.path.exists(c):
                nb_path = c
                break

    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Could not find opponent file or notebook: {opponent_arg}")

    if nb_path.endswith(".py"):
        with open(nb_path, encoding="utf-8", errors="ignore") as f:
            code = f.read()
        return get_last_callable(code, path=nb_path)

    elif nb_path.endswith(".ipynb"):
        cleaned_code, status = clean_source(nb_path)
        if cleaned_code is None:
            raise ValueError(f"Failed to clean notebook source from {nb_path}: {status}")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(cleaned_code)
            tmp_py = tf.name
        try:
            # Snapshot directory state before executing notebook code to cleanup any artifacts
            before_files = set(os.listdir(PROJECT_ROOT))
            callable_agent = get_last_callable(cleaned_code, path=tmp_py)
            after_files = set(os.listdir(PROJECT_ROOT))
            for new_f in (after_files - before_files):
                new_path = os.path.join(PROJECT_ROOT, new_f)
                try:
                    if os.path.isfile(new_path):
                        os.remove(new_path)
                except Exception:
                    pass
            return callable_agent
        finally:
            if os.path.exists(tmp_py):
                try:
                    os.remove(tmp_py)
                except Exception:
                    pass
    else:
        raise ValueError(f"Unsupported opponent file type: {nb_path}")


def run_benchmark(opponents: Union[str, List[str]],
                  episodes: int = 20,
                  seed_start: int = 1000,
                  seeds: list = None,
                  label: str = "benchmark",
                  out_path: str = None) -> Dict[str, Any]:
    """
    Run benchmark episodes against one or multiple opponents.
    Outputs a combined CSV with an 'opponent' column alongside standard columns.
    """
    # Normalize opponents list
    if isinstance(opponents, str):
        opponents = [opponents]
    resolved_opponents = []
    for opp in opponents:
        if opp.lower() in ("default", "suite", "all"):
            resolved_opponents.extend(DEFAULT_OPPONENTS)
        elif "," in opp:
            resolved_opponents.extend([o.strip() for o in opp.split(",") if o.strip()])
        else:
            resolved_opponents.append(opp)
    opponents = resolved_opponents

    if seeds is None or len(seeds) == 0:
        base_seeds = [seed_start + i for i in range(episodes)]
    else:
        base_seeds = list(seeds)
        episodes = len(base_seeds)

    # Determine CSV output path
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    out_dir = os.path.join(PROJECT_ROOT, "data", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)

    if out_path is None:
        csv_filename = f"{date_str}_{label}.csv"
        out_path = os.path.join(out_dir, csv_filename)

    initial_root_files = set(os.listdir(PROJECT_ROOT))
    all_records = []
    per_opponent_stats = {}

    all_agent_banks = []
    all_opp_banks = []
    overall_agent_wins = 0
    overall_opp_wins = 0
    overall_ties = 0

    # Diagnostic exception tracking
    clear_logged_exceptions()
    exc_log_path = out_path.replace(".csv", "_exceptions.log")
    if os.path.exists(exc_log_path):
        try:
            os.remove(exc_log_path)
        except Exception:
            pass

    global_ep_idx = 0

    print("=" * 95)
    print(f"MULTI-OPPONENT BENCHMARK SUITE")
    print(f"Opponents ({len(opponents)}): {', '.join(opponents)}")
    print(f"Episodes per opponent: {episodes} (Seeds {base_seeds[0]}..{base_seeds[-1]})")
    print(f"Output CSV: {out_path}")
    print(f"Exception Log: {exc_log_path}")
    print("=" * 95)

    for opp_idx, opponent_name in enumerate(opponents, 1):
        print(f"\n[{opp_idx}/{len(opponents)}] Testing vs '{opponent_name}' ({episodes} episodes)...")
        opp_callable = load_opponent_callable(opponent_name)

        opp_agent_banks = []
        opp_other_banks = []
        opp_agent_wins = 0
        opp_other_wins = 0
        opp_ties = 0
        opp_exceptions_total = 0

        print(f"{'Ep':>3} | {'Seed':>6} | {'Seat':>4} | {'Our Bank':>10} | {'Opp Bank':>10} | {'Margin':>10} | {'Winner':>6} | {'Collapsed':>9} | {'Exceptions':>10}")
        print("-" * 95)

        for ep_idx, seed in enumerate(base_seeds, 1):
            global_ep_idx += 1
            agent_seat = (ep_idx - 1) % 2
            opp_seat = 1 - agent_seat

            random.seed(seed)
            try:
                import numpy as np
                np.random.seed(seed)
            except ImportError:
                pass

            prev_exc_count = len(get_logged_exceptions())
            agents = [agent, opp_callable] if agent_seat == 0 else [opp_callable, agent]
            env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed, "randomSeed": seed}, debug=True)
            env.run(agents)

            # Check exceptions logged during this episode
            ep_exceptions = get_logged_exceptions()[prev_exc_count:]
            ep_exc_count = len(ep_exceptions)
            opp_exceptions_total += ep_exc_count

            if ep_exceptions:
                try:
                    with open(exc_log_path, "a", encoding="utf-8") as ef:
                        for ex in ep_exceptions:
                            ef.write(
                                f"[Ep {global_ep_idx} | Opponent: {opponent_name} | Seed: {seed} | Seat: {agent_seat} | "
                                f"Step: {ex.get('step')} | Day: {ex.get('day')} | Hour: {ex.get('hour')}] "
                                f"Error: {ex.get('error')}\n{ex.get('traceback')}\n"
                            )
                except Exception:
                    pass

            final_step = env.steps[-1]
            p0_reward = float(final_step[0]["reward"] or 0.0)
            p1_reward = float(final_step[1]["reward"] or 0.0)

            agent_bank = p0_reward if agent_seat == 0 else p1_reward
            opp_bank = p1_reward if agent_seat == 0 else p0_reward

            # Determine winner
            if p0_reward > p1_reward:
                winner = "p0"
            elif p1_reward > p0_reward:
                winner = "p1"
            else:
                winner = "tie"

            if agent_bank > opp_bank:
                opp_agent_wins += 1
                overall_agent_wins += 1
                win_label = "AGENT"
            elif opp_bank > agent_bank:
                opp_other_wins += 1
                overall_opp_wins += 1
                win_label = "OPP"
            else:
                opp_ties += 1
                overall_ties += 1
                win_label = "TIE"

            collapsed_p0 = (p0_reward < 5000.0)
            collapsed_p1 = (p1_reward < 5000.0)
            agent_collapsed = (agent_bank < 5000.0)

            day_terminated = min(30, (len(env.steps) + 23) // 24)

            # Track terminal produce values (Step 717 & Final)
            shed_val_717 = 0.0
            carried_val_717 = 0.0
            unsold_val_final = 0.0
            if len(env.steps) > 717:
                obs_717 = env.steps[717][agent_seat]["observation"]
                if "private" in obs_717 and "shed" in obs_717["private"]:
                    shed_val_717 = sum(
                        v * MARKET_PARAMS.get(k, {}).get("base", 0)
                        for k, v in obs_717["private"]["shed"].items()
                        if k not in ALL_ANIMALS and v > 0
                    )
                if "private" in obs_717 and "inventories" in obs_717["private"]:
                    for inv in obs_717["private"]["inventories"]:
                        carried_val_717 += sum(
                            v * MARKET_PARAMS.get(k, {}).get("base", 0)
                            for k, v in inv.items()
                            if k not in ALL_ANIMALS and v > 0
                        )

            obs_final = final_step[agent_seat]["observation"]
            if "private" in obs_final:
                if "shed" in obs_final["private"]:
                    unsold_val_final += sum(
                        v * MARKET_PARAMS.get(k, {}).get("base", 0)
                        for k, v in obs_final["private"]["shed"].items()
                        if k not in ALL_ANIMALS and v > 0
                    )
                if "inventories" in obs_final["private"]:
                    for inv in obs_final["private"]["inventories"]:
                        unsold_val_final += sum(
                            v * MARKET_PARAMS.get(k, {}).get("base", 0)
                            for k, v in inv.items()
                            if k not in ALL_ANIMALS and v > 0
                        )


            # Track animal deaths (bought vs ended)

            animals_bought = 0
            for st in env.steps:
                acts = st[agent_seat].get("action", {})
                market_acts = acts.get("market", []) if isinstance(acts, dict) else []
                for m_act in market_acts:
                    if len(m_act) >= 2 and m_act[0] == "BUY_ANIMAL":
                        qty = int(m_act[2]) if len(m_act) >= 3 else 1
                        animals_bought += qty

            animals_ended = 0
            tiles = obs_final.get("farms", [{}, {}])[agent_seat].get("tiles", [])
            for row in tiles:
                for cell in row:
                    if isinstance(cell, dict) and cell.get("kind") in ("PASTURE", "COOP") and cell.get("animal"):
                        animals_ended += 1
            animal_deaths = max(0, animals_bought - animals_ended)

            quads_unlocked = len(obs_final.get("farms", [{}, {}])[agent_seat].get("unlocked_quadrants", []))

            max_hands_hired = 0
            for st in env.steps:
                farms = st[0].get("observation", {}).get("farms", [])
                if len(farms) > agent_seat:
                    h_cnt = len(farms[agent_seat].get("hands", []))
                    if h_cnt > max_hands_hired:
                        max_hands_hired = h_cnt

            record = {
                "episode_id": global_ep_idx,
                "seed": seed,
                "opponent": opponent_name,
                "final_bank_p0": int(p0_reward),
                "final_bank_p1": int(p1_reward),
                "winner": winner,
                "collapsed_p0": collapsed_p0,
                "collapsed_p1": collapsed_p1,
                "day_terminated": day_terminated,
                "agent_exceptions": ep_exc_count,
                "exception_steps": ",".join(str(ex.get("step")) for ex in ep_exceptions) if ep_exceptions else "",
                "shed_val_717": int(shed_val_717),
                "carried_val_717": int(carried_val_717),
                "unsold_val_final": int(unsold_val_final),
                "animals_bought": animals_bought,
                "animals_ended": animals_ended,
                "animal_deaths": animal_deaths,
                "quads_unlocked": quads_unlocked,
                "max_hands": max_hands_hired,
            }

            all_records.append(record)
            opp_agent_banks.append(agent_bank)
            opp_other_banks.append(opp_bank)
            all_agent_banks.append(agent_bank)
            all_opp_banks.append(opp_bank)

            margin = agent_bank - opp_bank
            coll_str = "YES" if agent_collapsed else "no"
            exc_str = f"{ep_exc_count} (s:{ep_exceptions[0].get('step')})" if ep_exceptions else "0"
            death_str = f"d:{animal_deaths}" if animal_deaths > 0 else "0"
            q_str = f"q:{quads_unlocked}"
            h_str = f"h:{max_hands_hired}"
            print(f"{ep_idx:3d} | {seed:6d} | {agent_seat:4d} | ${int(agent_bank):9,d} | ${int(opp_bank):9,d} | ${int(margin):+9,d} | {win_label:>6} | {coll_str:>9} | {exc_str:>10} | {death_str:>5} | {q_str:>4} | {h_str:>4}", flush=True)


        # Compute per-opponent statistics
        mean_agent = statistics.mean(opp_agent_banks)
        median_agent = statistics.median(opp_agent_banks)
        min_agent = min(opp_agent_banks)
        max_agent = max(opp_agent_banks)
        stdev_agent = statistics.stdev(opp_agent_banks) if len(opp_agent_banks) > 1 else 0.0

        mean_opp = statistics.mean(opp_other_banks)
        median_opp = statistics.median(opp_other_banks)
        min_opp = min(opp_other_banks)
        max_opp = max(opp_other_banks)

        collapses = sum(1 for b in opp_agent_banks if b < 5000.0)
        win_rate = (opp_agent_wins / episodes) * 100.0
        collapse_rate = (collapses / episodes) * 100.0

        per_opponent_stats[opponent_name] = {
            "episodes": episodes,
            "mean_agent": mean_agent,
            "median_agent": median_agent,
            "min_agent": min_agent,
            "max_agent": max_agent,
            "stdev_agent": stdev_agent,
            "mean_opp": mean_opp,
            "median_opp": median_opp,
            "min_opp": min_opp,
            "max_opp": max_opp,
            "agent_wins": opp_agent_wins,
            "opp_wins": opp_other_wins,
            "ties": opp_ties,
            "win_rate": win_rate,
            "collapses": collapses,
            "collapse_rate": collapse_rate,
            "agent_exceptions": opp_exceptions_total,
        }

        print("-" * 95)
        print(f"Summary vs {opponent_name}: Win Rate={win_rate:.1f}% ({opp_agent_wins}/{episodes}) | Mean Bank=${mean_agent:,.1f} | Collapses={collapses}/{episodes} ({collapse_rate:.1f}%) | Exceptions={opp_exceptions_total}")

    # Clean up any transient files written to project root by external notebook agents
    final_root_files = set(os.listdir(PROJECT_ROOT))
    for junk in (final_root_files - initial_root_files):
        if junk.endswith((".py", ".gif", ".png", ".log", ".txt", ".json")):
            junk_path = os.path.join(PROJECT_ROOT, junk)
            if os.path.isfile(junk_path):
                try:
                    os.remove(junk_path)
                except Exception:
                    pass

    # Write combined CSV
    fieldnames = [
        "episode_id", "seed", "opponent", "final_bank_p0", "final_bank_p1",
        "winner", "collapsed_p0", "collapsed_p1", "day_terminated",
        "agent_exceptions", "exception_steps",
        "shed_val_717", "carried_val_717", "unsold_val_final",
        "animals_bought", "animals_ended", "animal_deaths",
        "quads_unlocked",
        "max_hands",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)


    # Compute overall statistics
    total_episodes = len(all_records)
    overall_mean_agent = statistics.mean(all_agent_banks)
    overall_median_agent = statistics.median(all_agent_banks)
    overall_min_agent = min(all_agent_banks)
    overall_max_agent = max(all_agent_banks)
    overall_stdev_agent = statistics.stdev(all_agent_banks) if len(all_agent_banks) > 1 else 0.0

    overall_mean_opp = statistics.mean(all_opp_banks)
    overall_collapses = sum(1 for b in all_agent_banks if b < 5000.0)
    overall_win_rate = (overall_agent_wins / total_episodes) * 100.0
    overall_collapse_rate = (overall_collapses / total_episodes) * 100.0
    overall_exceptions = sum(r["agent_exceptions"] for r in all_records)

    overall_mean_shed_717 = statistics.mean(r["shed_val_717"] for r in all_records)
    overall_mean_carried_717 = statistics.mean(r["carried_val_717"] for r in all_records)
    overall_mean_unsold_final = statistics.mean(r["unsold_val_final"] for r in all_records)
    q4_unlocked_count = sum(1 for r in all_records if r["quads_unlocked"] >= 4)
    overall_mean_hands = statistics.mean(r["max_hands"] for r in all_records)
    overall_max_hands = max(r["max_hands"] for r in all_records)

    print("\n" + "=" * 95)
    print(f"MULTI-OPPONENT BENCHMARK OVERALL SUMMARY (Total Episodes: {total_episodes})")
    print(f"Raw Combined CSV: {out_path}")
    if overall_exceptions > 0:
        print(f"WARNING: {overall_exceptions} exceptions recorded! Log: {exc_log_path}")
    else:
        print(f"Exceptions Fired: 0 (clean run)")
    print("=" * 95)
    print(f"{'Opponent':<45} | {'Episodes':>8} | {'Win Rate':>9} | {'Mean Bank':>10} | {'Collapses':>9} | {'Exceptions':>10}")
    print("-" * 95)
    for opp_name, st in per_opponent_stats.items():
        print(f"{opp_name:<45} | {st['episodes']:8d} | {st['win_rate']:8.1f}% | ${st['mean_agent']:9,.0f} | {st['collapses']:3d}/{st['episodes']:2d} ({st['collapse_rate']:4.1f}%) | {st['agent_exceptions']:10d}")
    print("-" * 95)
    print(f"{'OVERALL TOTAL':<45} | {total_episodes:8d} | {overall_win_rate:8.1f}% | ${overall_mean_agent:9,.0f} | {overall_collapses:3d}/{total_episodes:2d} ({overall_collapse_rate:4.1f}%) | {overall_exceptions:10d}")
    print("=" * 95)
    print(f"Our Overall Bank:     Mean=${overall_mean_agent:,.1f} | Median=${overall_median_agent:,.1f} | Min=${overall_min_agent:,.0f} | Max=${overall_max_agent:,.0f} | StDev=${overall_stdev_agent:,.1f}")
    print(f"Opp Overall Bank:     Mean=${overall_mean_opp:,.1f}")
    print(f"Head-to-Head:         Agent Wins={overall_agent_wins} | Opponent Wins={overall_opp_wins} | Ties={overall_ties}")
    total_animals_bought = sum(r["animals_bought"] for r in all_records)
    total_animals_ended = sum(r["animals_ended"] for r in all_records)
    total_animal_deaths = sum(r["animal_deaths"] for r in all_records)

    print(f"Total Collapses:      {overall_collapses} / {total_episodes} ({overall_collapse_rate:.1f}%)")
    print(f"Total Exceptions:     {overall_exceptions} / {total_episodes} episodes")
    print(f"Q4 Unlocked:          {q4_unlocked_count} / {total_episodes} ({q4_unlocked_count / total_episodes * 100.0:.1f}%)")
    print(f"Peak Hands Hired:     Mean Max={overall_mean_hands:.1f} | Absolute Peak={overall_max_hands}")
    print(f"Animals (Agent):      Bought={total_animals_bought} | Ended={total_animals_ended} | Deaths={total_animal_deaths} (Mortality={total_animal_deaths / max(1, total_animals_bought) * 100.0:.1f}%)")
    print(f"Terminal Produce:     Mean Shed@717=${overall_mean_shed_717:,.1f} | Mean Carried@717=${overall_mean_carried_717:,.1f} | Mean Unsold Lost@720=${overall_mean_unsold_final:,.1f}")
    print("=" * 95 + "\n")


    return {
        "out_path": out_path,
        "total_episodes": total_episodes,
        "overall_win_rate": overall_win_rate,
        "overall_mean_agent": overall_mean_agent,
        "overall_median_agent": overall_median_agent,
        "overall_collapses": overall_collapses,
        "overall_collapse_rate": overall_collapse_rate,
        "per_opponent_stats": per_opponent_stats,
        "records": all_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Run headless kaggriculture benchmarks.")
    parser.add_argument("--opponents", "--opponent", dest="opponents", nargs="+", default=None,
                        help="One or more opponent names, notebook paths, or 'suite' for the standard multi-opponent benchmark.")
    parser.add_argument("-n", "--episodes", type=int, default=20, help="Number of episodes to run per opponent (default: 20).")
    parser.add_argument("--seed-start", type=int, default=1000, help="Starting random seed (default: 1000).")
    parser.add_argument("--label", type=str, default="multiopponent", help="Label for CSV filename.")
    parser.add_argument("--out", type=str, default=None, help="Custom output CSV path.")
    args = parser.parse_args()

    opponents = args.opponents
    if opponents is None or len(opponents) == 0:
        opponents = DEFAULT_OPPONENTS

    run_benchmark(
        opponents=opponents,
        episodes=args.episodes,
        seed_start=args.seed_start,
        label=args.label,
        out_path=args.out
    )


if __name__ == "__main__":
    main()
