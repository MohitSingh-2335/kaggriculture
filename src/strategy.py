"""
Phase-based heuristic strategy for Kaggriculture.

Three phases:
- Early game (days 0-4):  Plant wheat, build cash flow
- Mid game (days 5-20):   Diversify, animals, expand land
- Late game (days 21-29): Liquidate everything, maximize final bank

Generates market orders (buy seeds, sell products, hire hands, buy land)
and planting targets for the scheduler.

Intelligence-driven design (from competitive analysis of 45 community notebooks):
- Terminal liquidation at steps 717-718
- Front-load wheat purchase to market slot 0 (prevents feed denial)
- Never buy SE quadrant (proven negative: lost 10-0 in tests)
- Aggressive hiring: 10 hands/day costs only $143 total
- Sell fertilizer as income stream (keep only 2)
- Target: 8 cows, 4-6 sheep, 3 quadrants (NE+SW)
"""

import os
from typing import List, Tuple, Dict, Optional, Any
from .game_state import GameState, Plant, Animal
from .opponent_tracker import get_opponent_tracker, reset_opponent_tracker
from .market_model import (
    calculate_price, predict_sell_revenue, best_products_to_sell,
    predict_buy_cost, marginal_price, optimal_sell_quantity, MARKET_I0,
)
from .actions import (
    farm_hand_cost, is_shed_adjacent, nearest_shed_tile,
)
from .constants import (
    CROP_DATA, ANIMAL_DATA, MARKET_PARAMS, SHOP_DEMAND,
    SINGLE_PRODUCT_SHOPS, ALL_CROPS, ALL_ANIMALS, ALL_PRODUCTS,
    QUADRANT_COSTS, TURNS_PER_DAY, SHED_CAPACITY,
    MAX_MARKET_ORDERS_PER_TURN, TOTAL_TURNS,
    TOWN_SHOP_SELL_INTERVAL, TOWN_CENTER_SELL_INTERVAL,
)

# --- Intelligence-derived constants ---
# Maximum number of hands to hire per day (hard ceiling for scaling)
MAX_DAILY_HIRES = 13
# Cost cutoff: stop hiring once the next hand costs more than this
# Must match the cutoff used in _generate_hire_orders so reserve calc stays consistent
HIRE_COST_CUTOFF = 55
# Minimum hires per day (cheap Fibonacci region, always affordable)
MIN_DAILY_HIRES = 10
# Target animal composition from top-20 convergence
TARGET_COWS = 8
TARGET_SHEEP = 6
TARGET_GOOSE = 0  # Confirmed negative under BFS baseline (-4.59% regression)
# Maximum quadrants to unlock (skip SE = 3 max)
MAX_QUADRANTS = 3
# Fertilizer reserve: always keep 2 in shed
FERTILIZER_RESERVE = 2
# Feed security fallback constants (Backlog Item #6)
# (Price ceiling is dynamic EV-based in _generate_feed_orders)
FEED_BUY_MAX_PER_TURN = 2

# Feed reserve protection constants (hysteresis band preserved from empirical benchmarks)
FEED_SELL_PROTECTION_DAYS = 3
FEED_SEED_PROTECTION_DAYS = 3
FEED_MARKET_BUY_TARGET_DAYS = 3
FEED_MARKET_BUY_MAX_UNITS = 10
FEED_EMERGENCY_PLANT_DAYS = 2


def _capped_daily_wage(max_hires: int) -> int:
    """Sum of hiring costs for up to max_hires, stopping when cost > HIRE_COST_CUTOFF.

    This ensures the reserve calculation matches the actual hiring cutoff
    used in _generate_hire_orders, not just MAX_DAILY_HIRES.
    """
    total = 0
    for i in range(max_hires):
        c = farm_hand_cost(i)
        if c > HIRE_COST_CUTOFF:
            break
        total += c
    return total


def _capped_remaining_hire_cost(hires_already: int, max_hires: int) -> int:
    """Sum of remaining hiring costs for today, respecting the cost cutoff."""
    total = 0
    for i in range(hires_already, max_hires):
        c = farm_hand_cost(i)
        if c > HIRE_COST_CUTOFF:
            break
        total += c
    return total


# Tracking daily starting money for rolling net-cash-flow analysis
_daily_start_money: Dict[int, int] = {}


def reset_strategy_state():
    """Reset strategy state tracking (e.g. between matches or in test suites)."""
    global _daily_start_money
    _daily_start_money.clear()
    reset_opponent_tracker()


def _track_daily_cash_flow(state: GameState):
    """Record money at the start of each day to track net cash flow over time."""
    global _daily_start_money
    if getattr(state, "step", 0) == 0:
        _daily_start_money.clear()
    day = state.day
    if day not in _daily_start_money or state.hour == 0:
        _daily_start_money[day] = state.my_money


def _is_cash_flow_throttled(state: GameState, current_money: Optional[int] = None) -> bool:
    """
    Rolling net-cash-flow tracker.
    Returns True if cash flow has been negative over the last 2-4 days
    AND money is below safe operating threshold (e.g. < $600 or < 3x daily wage bill).
    """
    day = state.day
    money = state.my_money if current_money is None else current_money
    daily_wage_10 = _capped_daily_wage(MIN_DAILY_HIRES)  # $143

    # If we have ample cash (>= $1000), no need to throttle
    if money >= 1000:
        return False

    # Need at least 2 days of history to detect a downward trend
    if day < 2:
        return False

    # Calculate daily money deltas over the last 3-4 days
    recent_deltas = []
    for d in range(max(1, day - 3), day + 1):
        prev_m = _daily_start_money.get(d - 1)
        cur_m = _daily_start_money.get(d, money)
        if prev_m is not None:
            recent_deltas.append(cur_m - prev_m)

    if not recent_deltas:
        return False

    consecutive_negative = sum(1 for delta in recent_deltas if delta < 0)
    net_recent_change = money - _daily_start_money.get(max(0, day - 3), money)

    # Condition: 2+ days of negative cash flow OR net 3-day drop of >$50,
    # combined with low cash reserve (< max(600, 3 * daily_wage_10))
    if (consecutive_negative >= 2 or net_recent_change < -50) and money < max(600, 3 * daily_wage_10):
        return True

    return False


TRAJECTORY_THROTTLED_THRESHOLD = 500


def _is_trajectory_throttled(state: GameState, threshold: Optional[int] = None, current_money: Optional[int] = None) -> bool:
    """
    Trajectory-based financial health tracker.
    Projects final money over a capped horizon (min(days_remaining, 10))
    based on the median of 5 trailing daily deltas.
    Triggers if median_daily_delta < 0 AND projected_final_money < threshold.
    """
    WINDOW = 5
    day = state.day
    if day < WINDOW:
        return False

    money = state.my_money if current_money is None else current_money
    deltas = []
    for d in range(day - WINDOW + 1, day + 1):
        prev_m = _daily_start_money.get(d - 1)
        cur_m = _daily_start_money.get(d, money)
        if prev_m is not None:
            deltas.append(cur_m - prev_m)

    if len(deltas) < WINDOW:
        return False

    sorted_deltas = sorted(deltas)
    median_daily_delta = sorted_deltas[len(sorted_deltas) // 2]

    config = getattr(state, "config", {}) or {}
    turns_per_day = config.get("turnsPerDay", getattr(state, "turns_per_day", TURNS_PER_DAY))
    episode_steps = config.get("episodeSteps", getattr(state, "episode_steps", TOTAL_TURNS))
    total_days = episode_steps // turns_per_day if turns_per_day > 0 else (TOTAL_TURNS // TURNS_PER_DAY)
    days_remaining = max(0, total_days - day)
    capped_horizon = min(days_remaining, 10)

    projected_final_money = money + median_daily_delta * capped_horizon

    limit = TRAJECTORY_THROTTLED_THRESHOLD if threshold is None else threshold
    if median_daily_delta < 0 and projected_final_money < limit:
        return True

    return False


def get_throttle_tier(state: GameState, current_money: Optional[int] = None) -> int:
    """
    Graduated financial throttle tier evaluation (0 to 3).
    Evaluated fresh from CURRENT cash every single turn — strictly non-latching.

    - Tier 0: Normal operation (unrestricted). Cash >= $600 or unthrottled.
    - Tier 1: Capital freeze (no new animals/land, no 10-day crops, maintain 8-10 hands).
              Cash $300-$600 with negative cash-flow/trajectory signals.
    - Tier 2: Maintenance mode (6 hands, prioritize watering existing crops over new planting).
              Cash $150-$300 with negative cash-flow/trajectory signals.
    - Tier 3: Acute emergency mode (3-4 hands, cheap emergency wheat only).
              Cash < $150 (runway < 1 day).
    """
    money = state.my_money if current_money is None else current_money
    if money >= 600:
        return 0
    if money < 150:
        return 3

    # Check underlying financial stress signals
    is_stressed = _is_cash_flow_throttled(state, current_money) or _is_trajectory_throttled(state, current_money=current_money)
    if not is_stressed:
        return 0

    if money >= 300:
        return 1
    return 2


_get_throttle_tier = get_throttle_tier


def _effective_max_hires(state) -> int:
    """Land-scaled and revenue-gated hand scaling.

    Early labor right-sizing:
    - 1 unlocked quadrant:  max 5 hands ($19/day payroll, covers ~27 daily chores)
    - 2 unlocked quadrants: max 8 hands ($54/day payroll, covers ~55 daily chores)
    - 3+ unlocked quadrants: max 10 hands ($143/day payroll)

    Days 11+ (on 3+ quadrants): unlock hand 11 when bank > 3x the cost of hand 11,
    each additional hand (12, 13) requires bank > 3x that hand's cost.
    Never exceed MAX_DAILY_HIRES (13).
    """
    farm = getattr(state, "my_farm", None)
    num_quads = len(getattr(farm, "unlocked_quadrants", [])) if farm else 1
    if num_quads <= 1:
        base_cap = 5
    elif num_quads == 2:
        base_cap = 8
    else:
        base_cap = 10

    day = getattr(state, "day", 0)
    money = getattr(state, "my_money", 0)

    if day <= 10 or base_cap < 10:
        return base_cap

    effective = base_cap
    for next_hire in range(base_cap, MAX_DAILY_HIRES):
        next_cost = farm_hand_cost(next_hire)
        # Must be able to sustain this hire: bank > 3x the next hand's cost
        if money > 3 * next_cost:
            effective = next_hire + 1
        else:
            break

    return effective


def _count_total_animals(state: GameState, animals_bought_today: int = 0) -> int:
    """
    Count all living animals associated with our farm:
    placed on farm tiles + stored in shed + carried by farm hands in inventories
    + optionally animals bought this turn.
    """
    farm_animals = len(state.my_farm.all_animals())
    shed_animals = sum(state.private.shed.get(a, 0) for a in ALL_ANIMALS)
    carried_animals = sum(inv.get(a, 0) for inv in state.private.inventories for a in ALL_ANIMALS)
    return farm_animals + shed_animals + carried_animals + animals_bought_today


# =============================================================================
# Phase Detection
# =============================================================================

def get_phase(day: int) -> str:
    """Determine current game phase from day number (0-29)."""
    if day < 5:
        return "early"
    elif day < 21:
        return "mid"
    else:
        return "late"


# =============================================================================
# Market Order Generation
# =============================================================================

def generate_market_orders(state: GameState) -> List[List]:
    """
    Generate all market orders for this turn.
    Called once per turn from the agent entrypoint.
    Respects MAX_MARKET_ORDERS_PER_TURN (10).
    """
    _track_daily_cash_flow(state)
    day = state.day
    phase = get_phase(day)

    # Intelligence: Steps 717-718 (Day 29, hours 21-22) — terminal liquidation
    # Sell EVERYTHING in shed (except live animals) to maximize final bank
    if (getattr(state, "step", 0) >= 717) or (state.day > 29 or (state.day == 29 and state.hour >= 21)):
        return _terminal_liquidation(state)

    orders = []
    current_money = state.my_money

    # -------------------------------------------------------------------------
    # CASH ACCOUNTING SEQUENCE (fund labor before capital purchases claim cash)
    # -------------------------------------------------------------------------
    # 1. Sell orders (generate revenue first)
    sell_orders = _generate_sell_orders(state, phase)

    # 2. Hire farm hands (cheap Fibonacci labor scaling — funded before capital purchases)
    hire_orders = _generate_hire_orders(state, phase, current_money)
    hires_so_far = state.my_farm.hires_today
    for _ in hire_orders:
        current_money -= farm_hand_cost(hires_so_far)
        hires_so_far += 1

    # 3. Land expansion (buy quadrants when profitable and affordable after labor)
    land_orders = _generate_land_orders(state, phase, current_money)
    for _ in land_orders:
        quad_idx = len(state.my_farm.unlocked_quadrants) - 1
        cost = QUADRANT_COSTS[quad_idx] if 0 <= quad_idx < len(QUADRANT_COSTS) else 2000
        current_money -= cost

    # 4. Animal purchases (cows and sheep in early and mid game after labor and land)
    animal_orders = []
    if phase in ("early", "mid"):
        animal_orders = _generate_animal_orders(state, current_money, len(land_orders))
        for ao in animal_orders:
            atype = ao[1]
            current_money -= ANIMAL_DATA.get(atype, {}).get("buy_cost", 400)

    # 5. Feed security fallback (BUY_PRODUCT WHEAT when shed wheat below safety buffer)
    feed_orders = _generate_feed_orders(state, current_money, len(animal_orders))
    for fo in feed_orders:
        fqty = fo[2] if len(fo) > 2 else 1
        fcost, _ = predict_buy_cost("WHEAT", fqty, _get_market_inventory(state, "WHEAT"))
        current_money -= fcost

    # 6. Seed purchases (buy seeds based on phase strategy)
    seed_orders = _generate_seed_orders(state, phase, current_money)
    for so in seed_orders:
        crop = so[1]
        qty = so[2] if len(so) > 2 else 1
        current_money -= CROP_DATA.get(crop, {}).get("seed_cost", 10) * qty

    # -------------------------------------------------------------------------
    # ORDER ASSEMBLY SEQUENCE (slot priority decoupled from cash accounting)
    # -------------------------------------------------------------------------
    orders.extend(sell_orders)
    orders.extend(land_orders)
    orders.extend(animal_orders)
    orders.extend(feed_orders)
    orders.extend(seed_orders)
    orders.extend(hire_orders)

    # Intelligence: front-load emergency feed and wheat seed purchases to market order slot 0
    # when wheat is needed (prevents starvation from feed denial)
    feed_and_wheat_orders = [
        o for o in orders
        if (len(o) >= 2 and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT")
        or (len(o) >= 2 and o[0] == "BUY_SEED" and o[1] == "WHEAT")
    ]
    other_orders = [
        o for o in orders
        if not (
            (len(o) >= 2 and o[0] == "BUY_PRODUCT" and o[1] == "WHEAT")
            or (len(o) >= 2 and o[0] == "BUY_SEED" and o[1] == "WHEAT")
        )
    ]
    orders = feed_and_wheat_orders + other_orders

    # Respect the max market orders per turn limit
    return orders[:MAX_MARKET_ORDERS_PER_TURN]


def _terminal_liquidation(state: GameState) -> List[List]:
    """
    End-game liquidation: sell EVERYTHING in the shed (excluding live animals).
    Orders products highest base-value first (MELON, WOOL, MILK, STRAWBERRY,
    FERTILIZER, TOMATO, EGG, CARROT, WHEAT sorted by base price descending).
    Step 717+ (Day 29, hours 21-23) are the final actionable turns.
    Unsold inventory does NOT count toward the bank.
    """
    orders = []
    shed = state.private.shed
    # Sort sellable products by base price descending, tie-breaking by name
    valid_products = [p for p in shed.keys() if p not in ALL_ANIMALS and shed[p] > 0]
    valid_products.sort(key=lambda p: (MARKET_PARAMS.get(p, {}).get("base", 0), p), reverse=True)

    for product in valid_products:
        orders.append(["SELL", product, shed[product]])

    return orders[:MAX_MARKET_ORDERS_PER_TURN]


# Maximum planting day per crop to guarantee full growth & harvest before Day 30
MAX_PLANTING_DAY = {
    "MELON": 15,       # 10 days growth + time to harvest & haul before Day 30
    "STRAWBERRY": 16,  # 10 days growth + multiple ongoing harvests
    "TOMATO": 19,      # 8 days growth + multiple ongoing harvests
    "CARROT": 26,      # 2 days fast turnaround
    "WHEAT": 26,       # 2 days fast turnaround (crucial for animal feed)
}

# Maximum concurrent plants per crop to match farm hand harvest & transport throughput
MAX_CONCURRENT_PLANTS = {
    "MELON": 16,       # Prevents massive 90+ melon harvest glut overwhelming hands
    "STRAWBERRY": 28,  # Concentrated strawberry cash-crop focus matching top ladder bots
    "TOMATO": 0,       # Deprioritized to free tiles and labor for Strawberry
    "CARROT": 0,       # Deprioritized to free tiles and labor for Strawberry
    "WHEAT": 28,       # Sized to sustain feed self-sufficiency and fill late-game tiles
}


def _get_max_concurrent_plants(crop: str, num_quadrants: int = 1) -> int:
    """Return maximum concurrent plants for a crop."""
    return MAX_CONCURRENT_PLANTS.get(crop, 12)

# Per-transaction sell caps to prevent lump-sum price crashing in market
SELL_BATCH_CAPS = {
    "MELON": 12,
    "STRAWBERRY": 8,
    "TOMATO": 10,
    "CARROT": 12,
    "WHEAT": 25,
    "MILK": 10,
    "WOOL": 10,
    "EGG": 15,
    "FERTILIZER": 15,
}


def demand_per_day(state: GameState, item: str) -> float:
    """
    Calculate total market consumption demand per day for a product.
    Sum: for each unlocked town shop that buys `item`, add (turns_per_day / shop_sell_interval)
    * (2 if shop is single-product else 1). Add turns_per_day / town_center_sell_interval
    for all non-FERTILIZER items. Uses actual config values from state when available.
    """
    config = getattr(state, "config", {}) or {}
    turns_per_day = config.get("turnsPerDay", getattr(state, "turns_per_day", TURNS_PER_DAY))
    shop_sell_interval = config.get("townShopSellInterval", TOWN_SHOP_SELL_INTERVAL)
    town_center_sell_interval = config.get("townCenterSellInterval", TOWN_CENTER_SELL_INTERVAL)

    total_demand = 0.0

    # Town shops demand
    if hasattr(state, "town") and hasattr(state.town, "unlocked_shops"):
        for shop in state.town.unlocked_shops:
            products = SHOP_DEMAND.get(shop, [])
            if item in products:
                multiplier = 2 if shop in SINGLE_PRODUCT_SHOPS else 1
                total_demand += (turns_per_day / shop_sell_interval) * multiplier

    # Town center consumes 1 of every non-FERTILIZER product every town_center_sell_interval turns
    if item != "FERTILIZER":
        total_demand += (turns_per_day / town_center_sell_interval)

    return total_demand


def _safe_sell_quantity(product: str, available_qty: int, current_inv: int) -> int:
    """
    Find safe sell quantity to avoid severe self-inflicted price decay.
    Caps batch size so price drop per transaction does not exceed ~12-15%.
    """
    if available_qty <= 1:
        return available_qty
    p_now = calculate_price(product, current_inv)
    min_acceptable_price = max(1, int(p_now * 0.88))

    best_q = 1
    for q in range(min(available_qty, 25), 0, -1):
        if calculate_price(product, current_inv + q) >= min_acceptable_price:
            best_q = q
            break
    return max(1, best_q)


def rank_sell_orders(orders: List[List], state: GameState) -> List[List]:
    """
    Rank SELL orders by impact-aware and demand-aware score.

    For each SELL order:
    - impact = quantity * (calculate_price(item, current_inv) - calculate_price(item, current_inv + quantity))
    - urgency = min(1.0, (excess_over_capacity / demand_per_day(state, item)) / 10.0)
    - final_score = impact * (1 + 0.25 * urgency)

    Orders are sorted by final_score descending before returning.
    Non-SELL orders (if any) are appended after the ranked SELL orders.
    """
    sell_scored = []
    other_orders = []

    market_inv = state.market.inventory if hasattr(state, "market") and hasattr(state.market, "inventory") else {}

    for order in orders:
        if len(order) >= 3 and order[0] == "SELL":
            item = order[1]
            qty = order[2]
            current_inv = market_inv.get(item, MARKET_I0)
            p_now = calculate_price(item, current_inv)
            p_after = calculate_price(item, current_inv + qty)
            impact = qty * (p_now - p_after)

            d_day = demand_per_day(state, item)
            excess_over_capacity = max(0.0, float(current_inv + qty - MARKET_I0))
            if d_day > 0:
                urgency = min(1.0, (excess_over_capacity / d_day) / 10.0)
            else:
                urgency = 1.0 if excess_over_capacity > 0 else 0.0

            final_score = impact * (1.0 + 0.25 * urgency)
            sell_scored.append((order, final_score))
        else:
            other_orders.append(order)

    sell_scored.sort(key=lambda x: x[1], reverse=True)
    sorted_sells = [order for order, _ in sell_scored]

    return sorted_sells + other_orders


def _generate_sell_orders(state: GameState, phase: str) -> List[List]:
    """
    Decide what to sell from the shed.
    Uses impact-aware sell ranking and demand-aware urgency to prioritize and size sales.
    Layers Clone-Horizon Opponent Preemption and Ledger Debt tracking to front-run dumps
    without double-selling inventory.
    """
    orders = []
    shed = state.private.shed
    market_inv = state.market.inventory
    day = state.day
    hour = state.hour
    current_step = getattr(state, "step", 0)

    # Intelligence: live animals in shed are for placement, never sell them
    produce_in_shed = {k: v for k, v in shed.items() if k not in ALL_ANIMALS and v > 0}

    # Final liquidation at step 717+ (Day 29 Hour 21+): handled by _terminal_liquidation
    if (current_step >= 717) or (state.day > 29 or (state.day == 29 and state.hour >= 21)):
        return _terminal_liquidation(state)

    # Normal phases: sell products when price is reasonable
    ranked = best_products_to_sell(produce_in_shed, market_inv)
    tracker = get_opponent_tracker()

    for product, qty, revenue, avg_price in ranked:
        cur_inv = market_inv.get(product, MARKET_I0)
        active_debt = tracker.get_ledger_debt(product)

        # Baseline available quantity after subtracting active ledger debt
        effective_qty = max(0, qty - active_debt)
        if effective_qty <= 0:
            continue

        if product == "FERTILIZER":
            sell_qty = effective_qty - FERTILIZER_RESERVE
            if sell_qty > 0:
                safe_qty = _safe_sell_quantity(product, sell_qty, cur_inv)
                orders.append(["SELL", product, min(sell_qty, safe_qty)])
            continue

        if product == "WHEAT":
            total_animals = _count_total_animals(state)
            wheat_reserve = max(0, total_animals * FEED_SELL_PROTECTION_DAYS)
            sell_qty = effective_qty - wheat_reserve
            if sell_qty <= 0:
                continue

            safe_qty = _safe_sell_quantity(product, sell_qty, cur_inv)

            # Check if opponent has an imminent dump rhythm or pipeline projection
            will_dump, _, reason = tracker.evaluate_preemption(product, day, hour, current_step, state)
            if will_dump and safe_qty > 0:
                orig_step = (day + 1) * 24 + 1 if reason == "pipeline" and hour in (22, 23) else current_step + 1
                tracker.record_preemptive_sale(product, safe_qty, current_step, orig_step)
                orders.append(["SELL", product, safe_qty])
                continue

            params = MARKET_PARAMS.get(product, {})
            base = params.get("base", 1)
            price_threshold = base * 0.5
            shed_fullness = state.private.total_shed_items() / SHED_CAPACITY
            if avg_price >= price_threshold or shed_fullness > 0.6:
                orders.append(["SELL", product, min(sell_qty, safe_qty)])
            continue

        # Other crops & livestock commodities: MILK, WOOL, MELON, STRAWBERRY, CARROT, TOMATO, EGG
        safe_qty = _safe_sell_quantity(product, effective_qty, cur_inv)

        # Clone-Horizon Opponent Preemption check (pipeline projection + reactive rhythm)
        will_dump, _, reason = tracker.evaluate_preemption(product, day, hour, current_step, state)
        if will_dump and safe_qty > 0:
            orig_step = (day + 1) * 24 + 1 if reason == "pipeline" and hour in (22, 23) else current_step + 1
            tracker.record_preemptive_sale(product, safe_qty, current_step, orig_step)
            orders.append(["SELL", product, safe_qty])
            continue

        params = MARKET_PARAMS.get(product, {})
        base = params.get("base", 1)

        # Sell if price is at least 50% of base, or if shed is getting full
        price_threshold = base * 0.5
        shed_fullness = state.private.total_shed_items() / SHED_CAPACITY

        if avg_price >= price_threshold or shed_fullness > 0.6:
            orders.append(["SELL", product, safe_qty])
        elif effective_qty > 8:
            orders.append(["SELL", product, max(1, min(effective_qty // 2, safe_qty))])

    return rank_sell_orders(orders, state)


def _generate_seed_orders(state: GameState, phase: str, current_money: Optional[int] = None) -> List[List]:
    """Decide which seeds to buy."""
    orders = []
    seeds = state.private.seeds
    money = state.my_money if current_money is None else current_money
    farm = state.my_farm
    
    # Account for empty tiles reserved for unplaced animals in shed
    unplaced_animals = sum(state.private.shed.get(a, 0) for a in ALL_ANIMALS)
    empty_count = max(0, len(farm.empty_tiles()) - unplaced_animals)

    if empty_count == 0:
        return orders  # No room to plant

    # If farm already has many crops in the ground, protect liquidity for wages until harvest
    # (Reduced by $143 today's-wage component now deducted upstream)
    min_liquidity = 557 if current_money is not None else 700
    if money < min_liquidity and len(farm.all_plants()) >= 20:
        return orders

    tier = get_throttle_tier(state, current_money)

    if tier == 3:
        # Tier 3: Acute emergency mode (< $150, runway < 1 day)
        # Only buy cheap wheat seeds if needed for feed or empty tiles, and only if affordable
        wheat_seeds = seeds.get("WHEAT", 0)
        needed = empty_count - wheat_seeds
        if needed > 0 and money >= 100:
            to_buy = min(needed, int((money - 50) // 10))
            if to_buy > 0:
                orders.append(["BUY_SEED", "WHEAT", to_buy])
        return orders

    if tier == 2:
        # Tier 2: Maintenance mode ($150-$300, runway 1-2 days)
        # Prioritize watering existing crops over new planting.
        # Only buy wheat seeds if needed for animal feed, or if farm has almost no crops in ground
        total_animals = _count_total_animals(state)
        wheat_in_shed = state.private.shed.get("WHEAT", 0)
        wheat_seeds = seeds.get("WHEAT", 0)
        needed = 0
        if total_animals > 0 and wheat_in_shed < total_animals * 2:
            needed = max(0, min(8, total_animals) - wheat_seeds)
        elif len(farm.all_plants()) < 5 and empty_count > 0:
            needed = min(empty_count - wheat_seeds, 5)

        if needed > 0 and money >= 100:
            to_buy = min(needed, int((money - 50) // 10))
            if to_buy > 0:
                orders.append(["BUY_SEED", "WHEAT", to_buy])
        return orders

    if phase == "early":
        # Early game: focus on wheat (cheap, 2-day turnaround; reserve 1-2 days wages)
        # When current_money is provided, today's $143 is already deducted upstream;
        # reserve only the remaining future-day portion ($200 - $143 = $57) to match baseline spendable cash
        if current_money is not None:
            wage_reserve = 57 if money > 207 else 0
        else:
            wage_reserve = 200 if money > 350 else 100
        spendable_money = max(0, money - wage_reserve)
        wheat_seeds = seeds.get("WHEAT", 0)
        needed = empty_count - wheat_seeds
        if needed > 0 and spendable_money >= 10:
            to_buy = min(needed, int(spendable_money // 10))
            if to_buy > 0:
                orders.append(["BUY_SEED", "WHEAT", to_buy])

    elif phase == "mid":
        # Dynamic rolling near-term obligations reserve for labor in mid game
        hires_today = farm.hires_today
        eff_hires = _effective_max_hires(state)
        target_daily_hires = eff_hires if state.day < 28 else 0
        if current_money is not None:
            # Upstream hire deduction already accounted for today's wages
            remaining_today_hire_cost = 0
        else:
            remaining_today_hire_cost = _capped_remaining_hire_cost(hires_today, target_daily_hires)
        future_days = min(5, max(0, 29 - state.day))
        daily_wage = _capped_daily_wage(eff_hires)
        wage_reserve = remaining_today_hire_cost + (future_days * daily_wage)
        spendable_money = max(0, money - wage_reserve)

        # Ensure wheat feed is grown if animals exist
        total_animals = _count_total_animals(state)
        if total_animals > 0 and state.private.shed.get("WHEAT", 0) < total_animals * FEED_SEED_PROTECTION_DAYS:
            wheat_seeds = seeds.get("WHEAT", 0)
            needed_feed = min(8, total_animals)
            if wheat_seeds < needed_feed and spendable_money >= 10:
                buy_wheat = min(needed_feed - wheat_seeds, int(spendable_money // 10))
                if buy_wheat > 0:
                    orders.append(["BUY_SEED", "WHEAT", buy_wheat])
                    spendable_money -= 10 * buy_wheat

        # Mid game: diversify based on what's profitable
        target_crops = _pick_crops_for_market(state, spendable_money)
        if tier == 1:
            # Tier 1: Capital freeze / risk mitigation: no new 10-day crops (Melon/Strawberry)
            target_crops = [(c, q) for c, q in target_crops if c not in ("MELON", "STRAWBERRY")]

        for crop, qty in target_crops:
            current = seeds.get(crop, 0)
            need = qty - current
            cost = CROP_DATA[crop]["seed_cost"]
            if need > 0 and spendable_money >= cost:
                buy_qty = min(need, int(spendable_money // cost))
                if buy_qty > 0:
                    orders.append(["BUY_SEED", crop, buy_qty])
                    spendable_money -= cost * buy_qty

        if tier == 1:
            # Fill remaining empty tiles with fast-turnaround wheat if affordable
            total_bought = sum(o[2] for o in orders if o[0] == "BUY_SEED")
            rem_empty = empty_count - total_bought - sum(seeds.values())
            if rem_empty > 0 and spendable_money >= 10:
                wheat_buy = min(rem_empty, int(spendable_money // 10))
                if wheat_buy > 0:
                    orders.append(["BUY_SEED", "WHEAT", wheat_buy])
                    spendable_money -= 10 * wheat_buy

    return orders


def _pick_crops_for_market(state: GameState, spendable_money_override: Optional[int] = None) -> List[Tuple[str, int]]:
    """
    Pick which crops to plant based on current market conditions and town demand.
    Returns list of (crop, quantity_to_buy).
    """
    farm = state.my_farm
    unplaced_animals = sum(state.private.shed.get(a, 0) for a in ALL_ANIMALS)
    empty_count = max(0, len(farm.empty_tiles()) - unplaced_animals)
    money = state.my_money
    day = state.day
    days_left = 29 - day

    # Count existing plants by type
    num_quadrants = len(getattr(farm, "unlocked_quadrants", [])) or 1
    existing = {}
    for plant in farm.all_plants():
        existing[plant.crop] = existing.get(plant.crop, 0) + 1

    # Score each crop based on profitability, harvest throughput, and time remaining
    crop_scores = []
    for crop in ALL_CROPS:
        data = CROP_DATA[crop]
        cost = data["seed_cost"]
        base_price = data["base_price"]

        # Check maturity cutoff: guarantee full growth + harvest window before Day 30
        if day > MAX_PLANTING_DAY.get(crop, 24):
            continue

        time_to_yield = data["time_to_first_yield"]
        if days_left < time_to_yield + 2:
            continue  # Not enough time to mature and harvest

        if cost > money:
            continue  # Can't afford

        # Check concurrent capacity: prevent overwhelming farm hands with 100+ units of 1 crop
        currently_committed = existing.get(crop, 0) + state.private.seeds.get(crop, 0)
        max_allowed = _get_max_concurrent_plants(crop, num_quadrants)
        if currently_committed >= max_allowed:
            continue

        # Simple profitability score: expected_revenue / cost
        if data["yield_type"] == "one_time":
            expected_yield = data.get("max_yield_unfertilized", data["max_yield"])
            # Check current market price
            market_inv = state.market.inventory.get(crop, MARKET_I0)
            current_price = calculate_price(crop, market_inv)
            expected_revenue = expected_yield * current_price
        else:
            # Ongoing: multiple yields
            max_yields = data["max_yield"]
            remaining_yields = min(max_yields, (days_left - time_to_yield) // data.get("yield_interval", 1) + 1)
            remaining_yields = max(0, min(remaining_yields, max_yields))
            product_name = crop
            market_inv = state.market.inventory.get(product_name, MARKET_I0)
            current_price = calculate_price(product_name, market_inv)
            expected_revenue = remaining_yields * current_price

        score = (expected_revenue - cost) / max(cost, 1)
        crop_scores.append((crop, score, cost))

    # Sort by score descending
    crop_scores.sort(key=lambda x: x[1], reverse=True)

    # Allocate seeds across available capacity
    if spendable_money_override is not None:
        spendable_money = spendable_money_override
    else:
        daily_wage = _capped_daily_wage(MAX_DAILY_HIRES)
        future_days = min(5, max(0, 29 - day))
        spendable_money = max(0, money - (future_days * daily_wage))

    result = []
    remaining_tiles = empty_count
    remaining_money = spendable_money

    profitable = [c for c in crop_scores if c[1] >= 0.5]
    if not profitable:
        if remaining_tiles > 0 and remaining_money >= 10 and day <= MAX_PLANTING_DAY.get("WHEAT", 26):
            currently_committed = existing.get("WHEAT", 0) + state.private.seeds.get("WHEAT", 0)
            room = max(0, _get_max_concurrent_plants("WHEAT", num_quadrants) - currently_committed)
            qty = min(room, remaining_tiles, int(remaining_money // 10))
            if qty > 0:
                result.append(("WHEAT", qty))
        return result

    # Distribute empty tiles across profitable crops so short-cycle crops sustain cash flow
    # while scaling to available land instead of a flat 3
    cap_per_crop = max(6, (remaining_tiles + len(profitable) - 1) // len(profitable))

    for crop, score, cost in profitable:
        if remaining_tiles <= 0 or remaining_money < cost:
            break

        currently_committed = existing.get(crop, 0) + state.private.seeds.get(crop, 0)
        max_allowed = _get_max_concurrent_plants(crop, num_quadrants)
        room_for_crop = max(0, max_allowed - currently_committed)
        if room_for_crop <= 0:
            continue

        # Scale quantity to actual available capacity, per-crop cap, and budget
        # For concentrated Strawberry focus, allow it to fill available room and budget
        alloc_cap = room_for_crop if crop == "STRAWBERRY" else cap_per_crop
        qty = min(room_for_crop, alloc_cap, remaining_tiles, int(remaining_money // cost))
        if qty > 0:
            result.append((crop, qty))
            remaining_tiles -= qty
            remaining_money -= cost * qty

    # Ensure at least 4-6 fast cash-flow crops (Wheat/Carrot, 2-day turnaround)
    # sustain daily wages while long-cycle crops (Melon/Strawberry, 10 days) grow
    fast_crops_count = existing.get("WHEAT", 0) + existing.get("CARROT", 0) + sum(q for c, q in result if c in ("WHEAT", "CARROT"))
    if fast_crops_count < 6 and remaining_tiles > 0 and remaining_money >= 10 and day <= MAX_PLANTING_DAY.get("WHEAT", 26):
        fast_crop = "WHEAT" if _get_max_concurrent_plants("CARROT", num_quadrants) == 0 or remaining_money < 20 else "CARROT"
        cost = 10 if fast_crop == "WHEAT" else 20
        needed_fast = min(6 - fast_crops_count, remaining_tiles, int(remaining_money // cost))
        if needed_fast > 0:
            result.append((fast_crop, needed_fast))
            remaining_tiles -= needed_fast
            remaining_money -= cost * needed_fast

    # Default to wheat if nothing was selected
    if not result and remaining_tiles > 0 and remaining_money >= 10 and day <= MAX_PLANTING_DAY.get("WHEAT", 26):
        currently_committed = existing.get("WHEAT", 0) + state.private.seeds.get("WHEAT", 0)
        room = max(0, _get_max_concurrent_plants("WHEAT", num_quadrants) - currently_committed)
        qty = min(room, remaining_tiles, int(remaining_money // 10))
        if qty > 0:
            result.append(("WHEAT", qty))

    return result


def _generate_animal_orders(
    state: GameState,
    current_money: Optional[int] = None,
    land_bought_today: int = 0,
) -> List[List]:
    """
    Decide whether to buy animals.

    Intelligence finding: top-20 all converge on 8 cows + 4-6 sheep.
    Cows produce milk ($160 base), sheep produce wool ($200 base).
    Goose is generally skipped (egg $50 is low-value).
    """
    orders = []
    if get_throttle_tier(state, current_money) >= 1:
        return orders

    farm = state.my_farm
    money = state.my_money if current_money is None else current_money
    day = state.day
    days_left = 29 - day

    # Rolling near-term obligations reserve:
    # 1. Remaining hire costs for today's target hands
    hires_today = farm.hires_today
    eff_hires = _effective_max_hires(state)
    target_daily_hires = eff_hires if day < 28 else 0
    if current_money is not None:
        # Upstream hire deduction already accounted for today's wages
        remaining_today_hire_cost = 0
    else:
        remaining_today_hire_cost = _capped_remaining_hire_cost(hires_today, target_daily_hires)

    # 2. Wages for next 4-5 days of operations (uses same cutoff as actual hiring)
    future_days = min(5, max(0, days_left))
    daily_wage = _capped_daily_wage(eff_hires)
    future_wages_reserve = future_days * daily_wage
    wage_reserve = remaining_today_hire_cost + future_wages_reserve

    # 3. Seed reserve: capital required to plant empty tiles (including tiles on newly purchased land)
    seed_cost_per_tile = CROP_DATA.get("WHEAT", {}).get("seed_cost", 10)
    seeds_in_stock = sum(state.private.seeds.values())
    unplaced_animals = sum(state.private.shed.get(a, 0) for a in ALL_ANIMALS)
    empty_structures_count = len(list(farm.empty_structures()))
    base_empty_tiles = len(farm.empty_tiles())
    new_quad_tiles = 25 * land_bought_today
    total_empty_tiles = base_empty_tiles + new_quad_tiles

    # Deduct structure tiles for planned animal purchases (up to 2) so we don't double-count them as crop tiles
    planned_animal_tiles = min(2, max(0, TARGET_COWS + TARGET_SHEEP - _count_total_animals(state)))
    tiles_needing_seed = max(0, total_empty_tiles - unplaced_animals - empty_structures_count - planned_animal_tiles - seeds_in_stock)
    seed_reserve = seed_cost_per_tile * tiles_needing_seed

    # 4. Feed buffer: ensure initial feed security for living and upcoming animals
    total_animals = _count_total_animals(state)
    total_wheat = state.private.shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for inv in state.private.inventories)
    target_feed_units = min(10, max(6, 3 * (total_animals + 2)))
    feed_deficit = max(0, target_feed_units - total_wheat)
    feed_unit_price = calculate_price("WHEAT", _get_market_inventory(state, "WHEAT"))
    feed_buffer = feed_deficit * feed_unit_price

    # Total rolling obligations reserve: wages + seed planting + feed buffer
    required_reserve = wage_reserve + seed_reserve + feed_buffer

    # Count existing animals on farm
    existing_animals = {}
    for animal in farm.all_animals():
        existing_animals[animal.animal_type] = existing_animals.get(animal.animal_type, 0) + 1

    # Count animals in shed and carried in inventories
    shed = state.private.shed
    carried_animals = {}
    for inv in state.private.inventories:
        for atype in ALL_ANIMALS:
            carried_animals[atype] = carried_animals.get(atype, 0) + inv.get(atype, 0)

    empty_structures = list(farm.empty_structures())
    empty_tiles = farm.empty_tiles()
    available_empty_tiles = len(empty_tiles)

    # Dynamic self-correcting purchase ordering: prioritize species proportionally further behind its target
    # e.g. (cows_needed / TARGET_COWS) vs (sheep_needed / TARGET_SHEEP)
    cows_needed = max(0, TARGET_COWS - (
        existing_animals.get("COW", 0) + shed.get("COW", 0) + carried_animals.get("COW", 0)
    ))
    sheep_needed = max(0, TARGET_SHEEP - (
        existing_animals.get("SHEEP", 0) + shed.get("SHEEP", 0) + carried_animals.get("SHEEP", 0)
    ))
    goose_needed = max(0, TARGET_GOOSE - (
        existing_animals.get("GOOSE", 0) + shed.get("GOOSE", 0) + carried_animals.get("GOOSE", 0)
    ))

    animal_purchase_queue = []
    c_need = cows_needed
    s_need = sheep_needed

    while c_need > 0 or s_need > 0:
        c_ratio = c_need / TARGET_COWS if TARGET_COWS > 0 else 0
        s_ratio = s_need / TARGET_SHEEP if TARGET_SHEEP > 0 else 0
        if s_ratio > c_ratio:
            animal_purchase_queue.append("SHEEP")
            s_need -= 1
        else:
            animal_purchase_queue.append("COW")
            c_need -= 1

    while goose_needed > 0:
        animal_purchase_queue.append("GOOSE")
        goose_needed -= 1

    num_quads = len(getattr(farm, "unlocked_quadrants", [])) if farm else 1
    if num_quads <= 1:
        max_land_animals = 6
    elif num_quads == 2:
        max_land_animals = 10
    else:
        max_land_animals = TARGET_COWS + TARGET_SHEEP

    current_animals = total_animals

    for animal_type in animal_purchase_queue:
        if current_animals >= max_land_animals:
            break

        data = ANIMAL_DATA[animal_type]
        buy_cost = data["buy_cost"]
        time_to_yield = data["time_to_first_yield"]

        # Need enough days for first yield
        if days_left < time_to_yield + 2:
            continue

        # Check rolling near-term obligations BEFORE queuing EACH animal purchase
        if money - buy_cost < required_reserve:
            break

        # Check if we have an empty matching structure
        matching = [s for s in empty_structures if _animal_matches_structure(animal_type, s.structure)]

        if matching:
            orders.append(["BUY_ANIMAL", animal_type, 1])
            money -= buy_cost
            empty_structures.remove(matching[0])
            current_animals += 1
        elif available_empty_tiles >= 2:
            # We have room to build a structure on an empty tile
            orders.append(["BUY_ANIMAL", animal_type, 1])
            money -= buy_cost
            available_empty_tiles -= 1
            current_animals += 1
        else:
            break

    return orders


def _animal_matches_structure(animal_type: str, structure: str) -> bool:
    """Check if an animal type matches a structure type."""
    if animal_type == "GOOSE" and structure == "COOP":
        return True
    if animal_type in ("COW", "SHEEP") and structure == "PASTURE":
        return True
    return False


def _get_market_inventory(state: GameState, product: str) -> int:
    """Safely get product inventory from Market object or dict."""
    market = getattr(state, "market", None)
    if market is None:
        return MARKET_I0
    if hasattr(market, "inventory_of"):
        return market.inventory_of(product)
    if hasattr(market, "inventory") and isinstance(market.inventory, dict):
        return market.inventory.get(product, MARKET_I0)
    if isinstance(market, dict):
        inv = market.get("inventory", market)
        if isinstance(inv, dict):
            return inv.get(product, MARKET_I0)
    return MARKET_I0


def _generate_feed_orders(
    state: GameState,
    current_money: Optional[int] = None,
    animals_bought_today: int = 0,
) -> List[List]:
    """
    Emergency feed-security fallback: generate BUY_PRODUCT WHEAT order when
    shed wheat stock is below safety threshold for living/committed animals.

    Parameters & thresholds:
    - Safety threshold: min(3 * total_animals, 10).
      Covers 2-day field germination lag + 1 buffer day per animal (referencing
      SEP L1888-1895), capped at 10 units to prevent hoarding when crops mature.
    - Per-turn cap: FEED_BUY_MAX_PER_TURN = 2 units/turn. Prevents single-turn
      spot market price escalation across 24 daily turns.
    - Dynamic EV valuation: replaces fixed price ceiling with live marginal daily
      revenue per animal (e.g. 0.5 * spot MILK for cows, 0.333 * spot WOOL for sheep).
      Feed is bought if unit_price < marginal_daily_value for at least one animal type
      currently at risk of starvation.
    - Liquidity safety: preserves a $50 cash reserve unless animals are in
      immediate starvation danger (shed_wheat == 0 or consecutive_unfed >= 1).
    """
    orders = []
    farm = state.my_farm
    money = state.my_money if current_money is None else current_money

    # Living animals on farm, in shed, carried, or bought this turn
    total_animals = _count_total_animals(state, animals_bought_today)

    if total_animals == 0:
        return orders

    # Current wheat stock across shed and carried inventories
    shed_wheat = state.private.shed.get("WHEAT", 0)
    carried_wheat = sum(inv.get("WHEAT", 0) for inv in state.private.inventories)
    total_wheat = shed_wheat + carried_wheat

    # Check shed capacity limit (100 total items)
    shed_total_items = sum(state.private.shed.values())
    if shed_total_items >= 100:
        return orders

    # Target safety buffer: 3 days of feed per animal, capped at 10 units
    safety_threshold = min(FEED_MARKET_BUY_TARGET_DAYS * total_animals, FEED_MARKET_BUY_MAX_UNITS)
    if total_wheat >= safety_threshold:
        return orders

    wheat_deficit = safety_threshold - total_wheat
    room_in_shed = 100 - shed_total_items
    max_to_buy = min(wheat_deficit, FEED_BUY_MAX_PER_TURN, room_in_shed)

    if max_to_buy <= 0:
        return orders

    # Dynamic EV valuation: compute marginal daily revenue per animal type present (COW, SHEEP)
    present_animal_types = set()
    for a in farm.all_animals():
        if a.animal_type in ("COW", "SHEEP"):
            present_animal_types.add(a.animal_type)
    for anim_name in ("COW", "SHEEP"):
        if state.private.shed_count(anim_name) > 0:
            present_animal_types.add(anim_name)
        for inv in state.private.inventories:
            if inv.get(anim_name, 0) > 0:
                present_animal_types.add(anim_name)
    if animals_bought_today > 0:
        present_animal_types.add("COW")

    marginal_values = []
    for atype in present_animal_types:
        if atype in ANIMAL_DATA:
            info = ANIMAL_DATA[atype]
            interval = info.get("yield_interval", 1)
            prod = info.get("product")
            spot_p = state.market.price_of(prod)
            if spot_p <= 0:
                spot_p = info.get("base_price", 0)
            marginal_values.append((1.0 / interval) * spot_p)

    if not marginal_values:
        marginal_values = [0.5 * ANIMAL_DATA["COW"]["base_price"]]

    max_marginal_value = max(marginal_values)

    # Relax cash reserve to $0 if animals are in immediate starvation danger, else protect $50
    starving_now = (shed_wheat == 0 and len(farm.all_animals()) > 0) or any(a.consecutive_unfed >= 1 for a in farm.all_animals())
    min_reserve = 0 if starving_now else 50

    # Starvation override ceiling derived from remaining amortized asset value:
    # When an animal has gone unfed for >=1 consecutive day (genuine starvation risk, dies if unfed today),
    # override the dynamic EV ceiling with an asset-preservation threshold reflecting the animal's
    # remaining amortized value (buy_cost minus value already extracted, with a $60 safety floor).
    # In normal conditions (consecutive_unfed == 0), the dynamic daily yield EV ceiling continues to govern.
    effective_ceiling = max_marginal_value
    starvation_risk = any(a.consecutive_unfed >= 1 for a in farm.all_animals())
    if starvation_risk:
        starvation_ceilings = [60.0]  # Minimum safety threshold floor
        for a in farm.all_animals():
            if a.consecutive_unfed >= 1 and a.animal_type in ANIMAL_DATA:
                info = ANIMAL_DATA[a.animal_type]
                buy_cost = float(info.get("buy_cost", 400))
                interval = float(info.get("yield_interval", 2))
                ttf = float(info.get("time_to_first_yield", 8))
                prod = info.get("product", "MILK")
                spot_p = float(state.market.price_of(prod))
                if spot_p <= 0:
                    spot_p = float(info.get("base_price", 160))

                # Amortized value: buy_cost minus value already extracted
                days_active = max(0, state.day - a.placed_day)
                days_producing = max(0, days_active - int(ttf))
                yields_so_far = days_producing // int(interval)
                value_extracted = yields_so_far * spot_p
                remaining_amortized = max(60.0, buy_cost - value_extracted)

                starvation_ceilings.append(remaining_amortized)

        starvation_ceiling = max(starvation_ceilings)
        effective_ceiling = max(max_marginal_value, starvation_ceiling)

    market_inventory = _get_market_inventory(state, "WHEAT")
    qty_to_buy = 0
    est_cost = 0
    inv = market_inventory

    for _ in range(max_to_buy):
        inv -= 1
        unit_price = calculate_price("WHEAT", inv)
        # Dynamic EV check: buy feed if unit_price < effective_ceiling.
        # Under normal conditions: effective_ceiling == max_marginal_value (daily yield EV).
        # Under starvation risk: effective_ceiling is elevated to protect the unamortized animal asset.
        if unit_price >= effective_ceiling:
            break
        if money - (est_cost + unit_price) < min_reserve:
            break
        qty_to_buy += 1
        est_cost += unit_price

    if qty_to_buy > 0:
        orders.append(["BUY_PRODUCT", "WHEAT", qty_to_buy])

    return orders


def _generate_hire_orders(state: GameState, phase: str, current_money: Optional[int] = None) -> List[List]:
    """
    Decide whether to hire farm hands.

    Revenue-gated scaling: ramps from MIN_DAILY_HIRES (10) to MAX_DAILY_HIRES (13)
    based on economic health. Stops once cost > HIRE_COST_CUTOFF (55) or
    _effective_max_hires ceiling is reached.
    """
    orders = []
    farm = state.my_farm
    money = state.my_money if current_money is None else current_money
    hires_today = farm.hires_today

    if (getattr(state, "step", 0) >= 717) or (state.day > 29 or (state.day == 29 and state.hour >= 21)):
        return orders  # Stop hiring once terminal liquidation begins

    # If no crops and no animals on farm in late game, do not hire
    active_workload = len(farm.all_plants()) + len(farm.all_animals()) + sum(state.private.shed.get(a, 0) for a in ALL_ANIMALS)
    if state.day >= 24 and active_workload == 0:
        return orders

    eff_max = _effective_max_hires(state)

    # Graduated throttle tier response
    tier = get_throttle_tier(state, current_money)
    if tier == 1:
        # Tier 1: Maintain 8-10 hands ($54-$143) to guarantee crops are watered
        eff_max = min(eff_max, 10 if money >= 400 else 8)
    elif tier == 2:
        # Tier 2: Step down to 6 hands ($20) to prioritize watering existing crops
        eff_max = min(eff_max, 6)
    elif tier >= 3:
        # Tier 3: Acute emergency mode (3-4 cheap hands)
        if money < 50:
            return orders
        eff_max = min(eff_max, 4 if money >= 100 else 3)

    # Hire up to the effective ceiling, stopping at cost cutoff
    while hires_today < eff_max:
        next_cost = farm_hand_cost(hires_today)

        # Stop if cost exceeds cutoff (consistent with reserve calc)
        if next_cost > HIRE_COST_CUTOFF:
            break

        # Stop if we can't afford it
        if next_cost > money:
            break

        orders.append(["HIRE"])
        money -= next_cost
        hires_today += 1

    return orders


def _generate_land_orders(state: GameState, phase: str, current_money: Optional[int] = None) -> List[List]:
    """
    Decide whether to buy new land quadrants.

    Intelligence finding: SE quadrant ($4000) is a proven NEGATIVE result.
    Every four-quadrant variant lost 10-0 in competition tests.
    Cap at 3 quadrants (NE + SW only).
    """
    orders = []
    if get_throttle_tier(state, current_money) >= 1:
        return orders
    farm = state.my_farm
    money = state.my_money if current_money is None else current_money
    day = state.day

    num_quadrants = len(farm.unlocked_quadrants)
    if num_quadrants >= MAX_QUADRANTS:  # Cap at 3 — never buy SE
        return orders

    cost_index = num_quadrants - 1  # 0-indexed into QUADRANT_COSTS
    if cost_index >= len(QUADRANT_COSTS):
        return orders

    cost = QUADRANT_COSTS[cost_index]

    # Land expansion capital buffers: calibrated to cover wage bill (5-7 days) + seed capital
    # When current_money is provided, today's $143 wage bill is already deducted upstream
    BUFFER_Q2_OVERRIDE = 1050
    BUFFER_Q2_STANDALONE = 1200
    BUFFER_Q3_OVERRIDE = 1350
    BUFFER_Q3_STANDALONE = 1500
    buffer_q2 = BUFFER_Q2_OVERRIDE if current_money is not None else BUFFER_Q2_STANDALONE
    buffer_q3 = BUFFER_Q3_OVERRIDE if current_money is not None else BUFFER_Q3_STANDALONE
    if num_quadrants == 1 and day >= 5 and money >= cost + buffer_q2:
        orders.append(["BUY_LAND"])
    elif num_quadrants == 2 and day >= 9 and money >= cost + buffer_q3:
        orders.append(["BUY_LAND"])
    # num_quadrants == 3 → NEVER buy SE (see intelligence report)

    return orders


# =============================================================================
# Planting Target Selection
# =============================================================================

def get_planting_targets(state: GameState) -> List[Tuple[int, int, str]]:
    """
    Determine what and where to plant.
    Returns list of (x, y, crop) targets.
    """
    phase = get_phase(state.day)
    if phase == "late":
        return []  # Don't plant in late game

    farm = state.my_farm
    seeds = state.private.seeds
    empty = list(farm.empty_tiles())

    # Stop planting on Day 25+ or at step >= 711 (Day 29 Hour 15+) so hands focus on harvesting/selling
    if state.day >= 25 or (
        (os.environ.get("DISABLE_TERMINAL_RECALL") != "1") and (
            (getattr(state, "step", 0) >= 711) or (state.day == 29 and state.hour >= 15)
        )
    ):
        return []

    # Reserve empty tiles for unhoused animals in shed + carried
    empty_structures = farm.empty_structures()
    empty_pastures = len([s for s in empty_structures if s.structure == "PASTURE"])
    empty_coops = len([s for s in empty_structures if s.structure == "COOP"])
    shed = state.private.shed
    unhoused_pasture = max(0, (shed.get("COW", 0) + shed.get("SHEEP", 0) +
                              sum(inv.get("COW", 0) + inv.get("SHEEP", 0) for inv in state.private.inventories)) - empty_pastures)
    unhoused_coop = max(0, (shed.get("GOOSE", 0) + sum(inv.get("GOOSE", 0) for inv in state.private.inventories)) - empty_coops)
    total_unhoused = unhoused_pasture + unhoused_coop

    # Only use remaining empty tiles for crops after reserving space for unhoused animals
    available_for_crops = empty[total_unhoused:] if total_unhoused < len(empty) else []
    targets = []

    # Plant seeds we have: if we have animals and low wheat in shed, prioritize wheat first to sustain feed!
    total_animals = _count_total_animals(state)
    wheat_in_shed = shed.get("WHEAT", 0)
    if total_animals > 0 and wheat_in_shed < total_animals * FEED_EMERGENCY_PLANT_DAYS:
        crop_priority = ["WHEAT", "MELON", "STRAWBERRY", "TOMATO", "CARROT"]
    else:
        crop_priority = ["MELON", "STRAWBERRY", "TOMATO", "CARROT", "WHEAT"]
    days_left = 29 - state.day

    for crop in crop_priority:
        count = seeds.get(crop, 0)
        if count <= 0:
            continue

        # Check maturity cutoff and time viability
        if state.day > MAX_PLANTING_DAY.get(crop, 24):
            continue

        data = CROP_DATA.get(crop, {})
        if days_left < data.get("time_to_first_yield", 999) + 2:
            continue

        for _ in range(count):
            if not available_for_crops:
                break
            pos = available_for_crops.pop(0)
            targets.append((pos[0], pos[1], crop))

    return targets


# =============================================================================
# Build Structure Targets
# =============================================================================

def get_build_targets(state: GameState) -> List[Tuple[int, int, str]]:
    """
    Determine where to build coops/pastures.
    Builds structures for all unhoused animals in shed + carried in inventories.
    Returns list of (x, y, build_action) targets.
    """
    # Stop building structures at step >= 711 (Day 29 Hour 15+) as animals cannot produce before game end
    if (os.environ.get("DISABLE_TERMINAL_RECALL") != "1") and (
        (getattr(state, "step", 0) >= 711) or (state.day > 29 or (state.day == 29 and state.hour >= 15))
    ):
        return []

    farm = state.my_farm
    empty = farm.empty_tiles()

    if not empty:
        return []

    targets = []
    empty_structures = farm.empty_structures()

    # Track how many empty structures are available
    empty_coops = len([s for s in empty_structures if s.structure == "COOP"])
    empty_pastures = len([s for s in empty_structures if s.structure == "PASTURE"])

    shed = state.private.shed

    # Count animals in shed + carried by units
    geese_count = shed.get("GOOSE", 0) + sum(inv.get("GOOSE", 0) for inv in state.private.inventories)
    cows_count = shed.get("COW", 0) + sum(inv.get("COW", 0) for inv in state.private.inventories)
    sheep_count = shed.get("SHEEP", 0) + sum(inv.get("SHEEP", 0) for inv in state.private.inventories)
    pasture_animals = cows_count + sheep_count

    # Build structures for all unhoused animals in shed + carried
    needed_coops = max(0, geese_count - empty_coops)
    needed_pastures = max(0, pasture_animals - empty_pastures)

    # Place structures on empty tiles (using reserved front of empty tiles list)
    available_tiles = list(empty)
    for _ in range(needed_coops):
        if available_tiles:
            pos = available_tiles.pop(0)
            targets.append((pos[0], pos[1], "BUILD_COOP"))

    for _ in range(needed_pastures):
        if available_tiles:
            pos = available_tiles.pop(0)
            targets.append((pos[0], pos[1], "BUILD_PASTURE"))

    return targets
