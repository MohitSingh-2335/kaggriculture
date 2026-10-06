"""
Game constants for Kaggriculture.

All values extracted from the competition specification (docs/Overview.md).
Grouped by category for easy reference. Every number is traceable to the spec.
"""

# =============================================================================
# Board & Timing
# =============================================================================

BOARD_SIZE = 10
TURNS_PER_DAY = 24
TOTAL_DAYS = 30
TOTAL_TURNS = TURNS_PER_DAY * TOTAL_DAYS  # 720

STARTING_MONEY = 3000
SHED_CAPACITY = 100
MAX_MARKET_ORDERS_PER_TURN = 10
WEED_SPAWN_CHANCE = 0.005

# Shed-adjacent tiles (the 4 center tiles)
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]

# Quadrant costs (sequential unlock)
QUADRANT_COSTS = [1000, 2000, 4000]

# Farm hand hiring cost: farmHandCostMult * fib(n) where n = hires already today
# fib: 1, 1, 2, 3, 5, 8, 13, 21, ...
FARM_HAND_COST_MULT = 1

# Town shop timing
TOWN_SHOP_UNLOCK_INTERVAL = 3   # days between shop unlocks
TOWN_SHOP_SELL_INTERVAL = 4     # turns between shop consumption ticks
TOWN_CENTER_SELL_INTERVAL = 24  # turns between town center consumption (once/day)
MAX_TOWN_SHOPS = 8

# =============================================================================
# Crop Data
# =============================================================================

# Yield type: "one_time" or "ongoing"
# time_to_first_yield: days until first harvest is possible
# time_to_max_yield: day at which yield stops increasing (one-time) or N/A
# max_yield: maximum units harvestable (one-time) or max_held (ongoing)
# max_yield_unfertilized: max yield without fertilizer (one-time crops only)
# subsequent_yields: for ongoing crops, the interval and count
# action_cost: always 1 for planting
# bonus_window_start: day at which bonus watering starts (ceil(max_yield_day/2))

CROP_DATA = {
    "WHEAT": {
        "seed_cost": 10,
        "base_price": 25,
        "yield_type": "one_time",
        "time_to_first_yield": 2,
        "time_to_max_yield": 4,
        "max_yield": 6,
        "max_yield_unfertilized": 4,
        "bonus_window_start": 2,  # ceil(4/2) = 2
    },
    "CARROT": {
        "seed_cost": 20,
        "base_price": 35,
        "yield_type": "one_time",
        "time_to_first_yield": 2,
        "time_to_max_yield": 3,
        "max_yield": 4,
        "max_yield_unfertilized": 3,
        "bonus_window_start": 2,  # ceil(3/2) = 2
    },
    "TOMATO": {
        "seed_cost": 50,
        "base_price": 60,
        "yield_type": "ongoing",
        "time_to_first_yield": 8,
        "time_to_max_yield": 11,
        "max_yield": 4,  # 4 scheduled yields
        "yield_interval": 1,  # every day
        "yield_ages": [8, 9, 10, 11],
    },
    "STRAWBERRY": {
        "seed_cost": 100,
        "base_price": 120,
        "yield_type": "ongoing",
        "time_to_first_yield": 10,
        "time_to_max_yield": 16,
        "max_yield": 4,  # 4 scheduled yields
        "yield_interval": 2,  # every other day
        "yield_ages": [10, 12, 14, 16],
    },
    "MELON": {
        "seed_cost": 80,
        "base_price": 250,
        "yield_type": "one_time",
        "time_to_first_yield": 10,
        "time_to_max_yield": 10,
        "max_yield": 6,
        "max_yield_unfertilized": 6,
        "bonus_window_start": 6,  # bonus window 6-12, cap reached at 10
    },
}

# =============================================================================
# Animal Data
# =============================================================================

ANIMAL_DATA = {
    "GOOSE": {
        "buy_cost": 300,
        "product": "EGG",
        "base_price": 50,
        "structure": "COOP",
        "time_to_first_yield": 4,
        "yield_interval": 1,   # every day
        "max_held": 4,
        "build_cost": 1,  # action to build coop
    },
    "COW": {
        "buy_cost": 400,
        "product": "MILK",
        "base_price": 160,
        "structure": "PASTURE",
        "time_to_first_yield": 8,
        "yield_interval": 2,   # every two days
        "max_held": 6,
        "build_cost": 1,
    },
    "SHEEP": {
        "buy_cost": 500,
        "product": "WOOL",
        "base_price": 200,
        "structure": "PASTURE",
        "time_to_first_yield": 6,
        "yield_interval": 3,   # every three days
        "max_held": 6,
        "build_cost": 1,
    },
}

# =============================================================================
# Market Parameters
# =============================================================================

# Shape functions: "linear", "sq", "sqrt", "log", "log10", "hinge"
# I0 = starting inventory (10000 for all)
# T = anchor throughput
# below_func / below_target = scarcity side (inv < I0 → price up)
# above_func / above_target = glut side (inv > I0 → price down)

MARKET_I0 = 10000  # Starting inventory for all products

MARKET_PARAMS = {
    "WHEAT": {
        "base": 25, "T": 400,
        "below_func": "sqrt", "below_target": 0.80,
        "above_func": "log",  "above_target": 0.20,
    },
    "CARROT": {
        "base": 35, "T": 450,
        "below_func": "hinge", "below_target": 1.00,
        "above_func": "sqrt",  "above_target": 0.70,
    },
    "TOMATO": {
        "base": 60, "T": 200,
        "below_func": "hinge", "below_target": 0.40,
        "above_func": "sqrt",  "above_target": 0.60,
    },
    "STRAWBERRY": {
        "base": 120, "T": 100,
        "below_func": "sqrt",   "below_target": 0.70,
        "above_func": "linear", "above_target": 1.60,
    },
    "MELON": {
        "base": 250, "T": 300,
        "below_func": "log", "below_target": 0.20,
        "above_func": "sq",  "above_target": 3.60,
    },
    "EGG": {
        "base": 50, "T": 332,
        "below_func": "hinge", "below_target": 0.40,
        "above_func": "log",   "above_target": 0.20,
    },
    "MILK": {
        "base": 160, "T": 122,
        "below_func": "sqrt",   "below_target": 0.60,
        "above_func": "linear", "above_target": 1.60,
    },
    "WOOL": {
        "base": 200, "T": 105,
        "below_func": "log", "below_target": 0.20,
        "above_func": "sq",  "above_target": 3.20,
    },
    "FERTILIZER": {
        "base": 100, "T": 200,
        "below_func": "linear", "below_target": 0.40,
        "above_func": "linear", "above_target": 0.40,
    },
}

# =============================================================================
# Town Shop Demand
# =============================================================================

# Each shop instance consumes 1 of each demanded product every
# TOWN_SHOP_SELL_INTERVAL turns. Single-product shops consume 2x.

SHOP_DEMAND = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],           # single-product → 2x
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],         # single-product → 2x
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

# Single-product shops get 2x consumption
SINGLE_PRODUCT_SHOPS = {"YARN_STORE", "PET_CAFE"}

# All sellable products
ALL_PRODUCTS = [
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
]

# Products that can be bought from the market via BUY_PRODUCT
BUYABLE_PRODUCTS = {"WHEAT", "FERTILIZER"}

# All crop types
ALL_CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]

# All animal types
ALL_ANIMALS = ["GOOSE", "COW", "SHEEP"]

# Direction vectors
DIRECTIONS = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST":  (1, 0),
    "WEST":  (-1, 0),
}
