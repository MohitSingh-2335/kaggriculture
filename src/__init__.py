"""
Kaggriculture — src package.

Core agent logic: game state parsing, market model, strategy, scheduling.
"""

from .game_state import GameState, Plant, Animal, Farm, Market, Town, PrivateState
from .constants import *
from .market_model import calculate_price, predict_sell_revenue, best_products_to_sell
from .actions import build_response, move_toward, manhattan_distance
from .scheduler import generate_tasks, assign_tasks
from .strategy import generate_market_orders, get_planting_targets, get_phase
