"""
Market price model for Kaggriculture.

Implements the exact price function from the competition spec:
    price(inv) = base + sign · amp · f(|inv − I0|)

Supports all 6 shape functions: linear, sq, sqrt, log, log10, hinge.
Provides revenue/cost prediction for selling/buying N units.
"""

import math
from typing import Dict, List, Tuple, Optional
from .constants import MARKET_PARAMS, MARKET_I0


# =============================================================================
# Shape Functions
# =============================================================================

def _shape_linear(x: float) -> float:
    return x

def _shape_sq(x: float) -> float:
    return x * x

def _shape_sqrt(x: float) -> float:
    return math.sqrt(x)

def _shape_log(x: float) -> float:
    """ln(1 + x), so f(0) = 0."""
    return math.log1p(x)

def _shape_log10(x: float) -> float:
    return math.log10(1 + x)

def _shape_hinge(x: float, T: float) -> float:
    """
    Hinge function: u + 8 * max(0, u - 1)^2, where u = x / T.
    Below T it's linear in u; above T the quadratic term dominates.
    f(T) = 1 by construction.
    """
    if T <= 0:
        return 0.0
    u = x / T
    return u + 8.0 * max(0.0, u - 1.0) ** 2


SHAPE_FUNCTIONS = {
    "linear": _shape_linear,
    "sq": _shape_sq,
    "sqrt": _shape_sqrt,
    "log": _shape_log,
    "log10": _shape_log10,
    # hinge handled separately because it needs T
}


def _evaluate_shape(func_name: str, x: float, T: float) -> float:
    """Evaluate a shape function by name."""
    if func_name == "hinge":
        return _shape_hinge(x, T)
    func = SHAPE_FUNCTIONS.get(func_name)
    if func is None:
        raise ValueError(f"Unknown shape function: {func_name}")
    return func(x)


# =============================================================================
# Price Function
# =============================================================================

def calculate_price(product: str, inventory: int, params: Optional[dict] = None) -> int:
    """
    Calculate the market sell price for a product at a given inventory level.

    Uses the exact formula from the spec:
        price(inv) = base + sign · amp · f(|inv − I0|)

    Returns the price floored at $1 and rounded to nearest dollar.
    """
    if params is None:
        params = MARKET_PARAMS.get(product)
    if params is None:
        return 1

    base = params["base"]
    T = params["T"]
    I0 = MARKET_I0

    if inventory == I0:
        return base

    diff = abs(inventory - I0)

    if inventory < I0:
        # Scarcity → price goes up
        func_name = params["below_func"]
        target = params["below_target"]
        sign = 1
    else:
        # Glut → price goes down
        func_name = params["above_func"]
        target = params["above_target"]
        sign = -1

    # amp = target * base / f(T)
    f_T = _evaluate_shape(func_name, T, T)
    if f_T == 0:
        return base

    amp = target * base / f_T

    # price = base + sign * amp * f(diff)
    f_diff = _evaluate_shape(func_name, diff, T)
    raw_price = base + sign * amp * f_diff

    # Floor at $1, round to nearest dollar
    return max(1, round(raw_price))


# =============================================================================
# Revenue / Cost Prediction
# =============================================================================

def predict_sell_revenue(product: str, quantity: int, current_inventory: int) -> Tuple[int, List[int]]:
    """
    Predict total revenue from selling `quantity` units of `product`.

    Each unit sold increases market inventory by 1, which may decrease
    the price for the next unit. Exception: if price hits $1 floor,
    the unit is sold but NOT added to inventory.

    Returns:
        (total_revenue, list_of_per_unit_prices)
    """
    total = 0
    prices = []
    inv = current_inventory

    for _ in range(quantity):
        # Sell price is quoted at PRE-sell inventory
        price = calculate_price(product, inv)
        total += price
        prices.append(price)
        # Unit is added to inventory unless price was at floor
        if price > 1:
            inv += 1

    return total, prices


def predict_buy_cost(product: str, quantity: int, current_inventory: int) -> Tuple[int, List[int]]:
    """
    Predict total cost of buying `quantity` units of `product` via BUY_PRODUCT.

    Only WHEAT and FERTILIZER can be bought. Buy price is quoted at
    POST-buy inventory (after removing 1 unit).

    Returns:
        (total_cost, list_of_per_unit_prices)
    """
    total = 0
    prices = []
    inv = current_inventory

    for _ in range(quantity):
        # Buy removes from inventory, then quotes price
        inv -= 1
        price = calculate_price(product, inv)
        total += price
        prices.append(price)

    return total, prices


def best_products_to_sell(
    shed: Dict[str, int],
    market_inventory: Dict[str, int],
) -> List[Tuple[str, int, int, float]]:
    """
    Rank products by marginal sell revenue.

    Returns list of (product, quantity_available, total_revenue, avg_price_per_unit)
    sorted by avg_price_per_unit descending.
    """
    results = []
    for product, qty in shed.items():
        if qty <= 0:
            continue
        if product in ("GOOSE", "COW", "SHEEP"):
            # Can't sell animals (they're not products)
            continue
        inv = market_inventory.get(product, MARKET_I0)
        revenue, _ = predict_sell_revenue(product, qty, inv)
        avg = revenue / qty if qty > 0 else 0
        results.append((product, qty, revenue, avg))

    results.sort(key=lambda x: x[3], reverse=True)
    return results


def marginal_price(product: str, current_inventory: int) -> int:
    """Price you'd get for selling 1 unit right now."""
    return calculate_price(product, current_inventory)


# Marginal production cost per unit: seed/animal amortization + essential labor/feed.
MARGINAL_PRODUCTION_COST: Dict[str, int] = {
    "WHEAT": 3,        # Seed $10 / ~3-4 yield = $2.50-$3.33
    "CARROT": 7,       # Seed $20 / 3 yield = $6.67
    "TOMATO": 13,      # Seed $50 / 4 scheduled yields = $12.50
    "STRAWBERRY": 25,  # Seed $100 / 4 scheduled yields = $25.00
    "MELON": 20,       # Seed $80 / 4-6 yield = $13.33-$20.00
    "EGG": 15,         # Goose amortized
    "MILK": 20,        # Cow amortized + wheat feed cost
    "WOOL": 30,        # Sheep amortized + wheat feed cost
    "FERTILIZER": 5,   # Labor collection opportunity cost
}


def optimal_sell_quantity(
    product: str,
    available_qty: int,
    current_inv: int,
    max_slippage: float = 0.15,
    min_price_override: Optional[int] = None,
) -> int:
    """
    Compute maximum sellable units before price drops below marginal production cost
    or exceeds allowable per-transaction price slippage.
    Uses the exact market price impact formula from calculate_price().
    """
    if available_qty <= 0:
        return 0
    if available_qty == 1:
        return 1

    marginal_cost = MARGINAL_PRODUCTION_COST.get(product, 1)
    if min_price_override is not None:
        floor_price = max(marginal_cost, min_price_override)
    else:
        floor_price = marginal_cost

    # Current spot price quoted at pre-sell inventory
    p_spot = calculate_price(product, current_inv)
    # Per-transaction slippage ceiling: do not let price decay by more than max_slippage in one order
    slippage_floor = max(1, int(p_spot * (1.0 - max_slippage)))
    target_min_price = max(floor_price, slippage_floor)

    best_q = 1
    inv = current_inv
    for q in range(1, available_qty + 1):
        p_unit = calculate_price(product, inv)
        if p_unit < target_min_price:
            break
        best_q = q
        if p_unit > 1:
            inv += 1

    return max(1, best_q)

