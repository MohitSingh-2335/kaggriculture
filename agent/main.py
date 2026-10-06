"""
Kaggriculture submission entrypoint.

This is the file that ships to Kaggle (bundled with src/ as a tar.gz).
Imports the strategy engine and scheduler from src/ and wires them together.

The agent function receives a raw observation dict each turn and must return
a valid action dict with farmer, hands, and market keys.

Intelligence-driven design:
- Terminal liquidation at steps 717-718
- Market front-running when opponent mirrors our farm
- Weed recovery for blocked productive actions
- Front-loaded wheat purchase prevents feed denial
"""

import sys
import os
import traceback

# Diagnostic in-memory exception tracking for benchmarks and harness inspection
_LOGGED_EXCEPTIONS: list = []


def get_logged_exceptions() -> list:
    """Return a shallow copy of logged diagnostic exceptions."""
    return list(_LOGGED_EXCEPTIONS)


def clear_logged_exceptions():
    """Clear the in-memory diagnostic exception store."""
    _LOGGED_EXCEPTIONS.clear()


# Ensure src/ is importable when running inside Kaggle's sandbox
# In Kaggle's evaluation runtime, exec() is used without __file__ defined
if "__file__" in globals() and __file__:
    _agent_dir = os.path.dirname(os.path.abspath(__file__))
    _parent_dir = os.path.dirname(_agent_dir)
    for p in [_agent_dir, _parent_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)
else:
    for p in [os.getcwd(), "."]:
        if p not in sys.path:
            sys.path.insert(0, p)

from src.game_state import GameState
from src.scheduler import generate_tasks, assign_tasks, reset_scheduler_state
from src.strategy import (
    generate_market_orders, get_planting_targets, get_build_targets,
    reset_strategy_state,
)
from src.opponent_tracker import get_opponent_tracker, reset_opponent_tracker
from src.actions import build_response
def _detect_mirror_farm(state: GameState) -> bool:
    """
    Detect if the opponent's farm is a near-mirror of ours.

    Maintained for unit-test compatibility; production execution uses
    unconditional demand-aware front-running via _front_run_premium_sells.
    """
    my = state.my_farm
    opp = state.opponent_farm

    my_animals = {}
    for a in my.all_animals():
        my_animals[a.animal_type] = my_animals.get(a.animal_type, 0) + 1
    opp_animals = {}
    for a in opp.all_animals():
        opp_animals[a.animal_type] = opp_animals.get(a.animal_type, 0) + 1

    animal_diff = 0
    for atype in set(list(my_animals.keys()) + list(opp_animals.keys())):
        animal_diff += abs(my_animals.get(atype, 0) - opp_animals.get(atype, 0))

    my_plants = len(my.all_plants())
    opp_plants = len(opp.all_plants())

    return animal_diff <= 3 and abs(my_plants - opp_plants) <= 5


def _front_run_premium_sells(state: GameState, orders: list) -> list:
    """
    Unconditionally front-run premium product sales across all opponents.

    Intelligence finding: "the timing rule was doing real work rather than
    merely decorating a stronger base" — c15 went 14-2 against c14 without
    the wrapper; market timing accounts for ~80% of the competitive edge
    (MASTER_INTELLIGENCE.md:3.1).

    Promote SELL orders for zero-town-demand, high-elasticity goods
    (MELON, STRAWBERRY, MILK, WOOL) to the earliest market order slots (0–2),
    ahead of goods with off-market absorption or low elasticity
    (WHEAT, FERTILIZER, CARROT, EGG, TOMATO).
    """
    # Premium products that benefit from front-running (zero town demand, high elasticity)
    premium = {"MELON", "STRAWBERRY", "MILK", "WOOL"}

    # Split into premium sells and other orders
    premium_sells = []
    other_orders = []
    for order in orders:
        if len(order) >= 3 and order[0] == "SELL" and order[1] in premium:
            premium_sells.append(order)
        else:
            other_orders.append(order)

    # Premium sells go first (slots 0–2 capture pre-glut un-depressed prices)
    return premium_sells + other_orders


_last_step: int = -1
_last_market_orders: list = []


def agent(obs):
    """
    Main agent function called by kaggle_environments each turn.

    Args:
        obs: raw observation dict from the environment

    Returns:
        Action dict: {"farmer": [...], "hands": [[...], ...], "market": [[...], ...]}
    """
    global _last_step, _last_market_orders
    try:
        # Parse the raw observation into typed game state
        state = GameState.from_obs(obs)

        # Reset persistent module state on step 0 or when step rewinds (fresh episode)
        current_step = getattr(state, "step", 0)
        if current_step == 0 or current_step < _last_step:
            reset_scheduler_state()
            reset_strategy_state()
            reset_opponent_tracker()
            _last_market_orders = []
        _last_step = current_step

        # Step 0.5: Update opponent observation tracker before generating turn decisions
        try:
            get_opponent_tracker().update(obs, _last_market_orders, state=state)
        except Exception as oe:
            sys.stderr.write(f"[OPPONENT TRACKER EXCEPTION step={current_step}] {oe}\n")
            sys.stderr.flush()

        # Step 1: Generate market orders (isolated try/except for resilience)
        try:
            market_orders = generate_market_orders(state)
            market_orders = _front_run_premium_sells(state, market_orders)
            _last_market_orders = list(market_orders)
        except Exception as me:
            tb_str = traceback.format_exc()
            _LOGGED_EXCEPTIONS.append({
                "step": current_step, "subsystem": "market", "error": str(me), "traceback": tb_str
            })
            sys.stderr.write(f"[AGENT MARKET EXCEPTION step={current_step}] {me}\n{tb_str}\n")
            sys.stderr.flush()
            market_orders = []
            _last_market_orders = []

        # Step 2: Generate chore tasks and assign to units
        try:
            plant_targets = get_planting_targets(state)
            build_targets = get_build_targets(state)
            tasks = generate_tasks(state, plant_targets=plant_targets, build_targets=build_targets)
            farmer_action, hands_actions = assign_tasks(state, tasks)
        except Exception as se:
            tb_str = traceback.format_exc()
            _LOGGED_EXCEPTIONS.append({
                "step": current_step, "subsystem": "scheduler", "error": str(se), "traceback": tb_str
            })
            sys.stderr.write(f"[AGENT SCHEDULER EXCEPTION step={current_step}] {se}\n{tb_str}\n")
            sys.stderr.flush()
            farmer_action = ["PASS"]
            num_hands = len(state.private.inventories) - 1 if (state.private and state.private.inventories) else len(getattr(state.my_farm, "hands", []))
            hands_actions = [["PASS"] for _ in range(num_hands)]

        return build_response(farmer_action, hands_actions, market_orders)

    except Exception as e:
        # Outer catastrophic safety net: never crash
        tb_str = traceback.format_exc()
        step_val = obs.get("step") if isinstance(obs, dict) else (getattr(state, "step", None) if "state" in locals() else None)
        day_val = obs.get("day") if isinstance(obs, dict) else (getattr(state, "day", None) if "state" in locals() else None)
        hour_val = obs.get("hour") if isinstance(obs, dict) else (getattr(state, "hour", None) if "state" in locals() else None)
        player_val = obs.get("player") if isinstance(obs, dict) else (getattr(state, "player", None) if "state" in locals() else None)
        exc_record = {
            "step": step_val,
            "day": day_val,
            "hour": hour_val,
            "player": player_val,
            "error": str(e),
            "traceback": tb_str,
        }
        _LOGGED_EXCEPTIONS.append(exc_record)
        sys.stderr.write(
            f"[AGENT CATASTROPHIC EXCEPTION step={step_val} day={day_val} hour={hour_val} player={player_val}] {e}\n{tb_str}\n"
        )
        sys.stderr.flush()

        return {"farmer": ["PASS"], "hands": [], "market": []}


# Kaggle expects the agent function to be importable as the module-level callable
act = agent
