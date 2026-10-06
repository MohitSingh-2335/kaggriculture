"""
Action helpers and pathfinding for Kaggriculture.

Provides movement planning, shed adjacency checks, and response dict construction.
"""

from typing import Tuple, List, Optional, Dict, Any
from collections import deque
from .constants import BOARD_SIZE, SHED_TILES, DIRECTIONS


# =============================================================================
# Distance & Pathfinding
# =============================================================================

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance between two positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def move_toward(start: Tuple[int, int], target: Tuple[int, int]) -> str:
    """
    Return the direction string (NORTH/SOUTH/EAST/WEST) to move one step
    from `start` toward `target`. Returns "PASS" if already at target.

    Uses simple priority: reduce the larger axis difference first.
    """
    dx = target[0] - start[0]
    dy = target[1] - start[1]

    if dx == 0 and dy == 0:
        return "PASS"

    # Prefer reducing the larger difference first
    if abs(dx) >= abs(dy):
        return "EAST" if dx > 0 else "WEST"
    else:
        return "SOUTH" if dy > 0 else "NORTH"


def path_to(
    start: Tuple[int, int],
    target: Tuple[int, int],
    board_size: int = BOARD_SIZE,
) -> List[str]:
    """
    BFS shortest path from start to target on a board_size × board_size grid.
    Returns a list of direction strings. All tiles are passable (including LOCKED).
    """
    if start == target:
        return []

    visited = {start}
    queue = deque([(start, [])])

    while queue:
        pos, path = queue.popleft()

        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = pos[0] + dx, pos[1] + dy

            if 0 <= nx < board_size and 0 <= ny < board_size and (nx, ny) not in visited:
                new_path = path + [direction]
                if (nx, ny) == target:
                    return new_path
                visited.add((nx, ny))
                queue.append(((nx, ny), new_path))

    return []  # Should never happen on an open grid


def nearest_tile(
    pos: Tuple[int, int],
    targets: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Find the nearest target position by Manhattan distance."""
    if not targets:
        return None
    return min(targets, key=lambda t: manhattan_distance(pos, t))


# =============================================================================
# Shed Helpers
# =============================================================================

def is_shed_adjacent(pos: Tuple[int, int], board_size: int = BOARD_SIZE) -> bool:
    """Check if position is one of the 4 shed-adjacent center tiles."""
    if board_size == BOARD_SIZE:
        return pos in SHED_TILES
    half = board_size // 2
    return pos in {
        (half - 1, half - 1), (half, half - 1),
        (half - 1, half),     (half, half),
    }


def shed_adjacent_tiles(board_size: int = BOARD_SIZE) -> List[Tuple[int, int]]:
    """Return the 4 shed-adjacent tile positions."""
    if board_size == BOARD_SIZE:
        return list(SHED_TILES)
    half = board_size // 2
    return [
        (half - 1, half - 1), (half, half - 1),
        (half - 1, half),     (half, half),
    ]


def nearest_shed_tile(pos: Tuple[int, int], board_size: int = BOARD_SIZE) -> Tuple[int, int]:
    """Find the nearest shed-adjacent tile."""
    tiles = shed_adjacent_tiles(board_size)
    return min(tiles, key=lambda t: manhattan_distance(pos, t))


def pickup_action(product: str, quantity: Any = 1) -> List[Any]:
    """Construct a PICKUP action list."""
    return ["PICKUP", product, str(quantity)]


# =============================================================================
# Response Construction
# =============================================================================

def build_response(
    farmer_action: Optional[List[str]] = None,
    hands_actions: Optional[List[List[str]]] = None,
    market_orders: Optional[List[List]] = None,
) -> Dict[str, Any]:
    """
    Build the action response dict for the Kaggle environment.

    Args:
        farmer_action: e.g. ["WATER"] or ["PLANT", "WHEAT"] or ["NORTH"]
        hands_actions: list of actions for each hired hand
        market_orders: e.g. [["BUY_SEED", "WHEAT", 1], ["SELL", "CARROT", 5]]
    """
    if farmer_action is None:
        farmer_action = ["PASS"]
    if hands_actions is None:
        hands_actions = []
    if market_orders is None:
        market_orders = []

    return {
        "farmer": farmer_action,
        "hands": hands_actions,
        "market": market_orders,
    }


# =============================================================================
# Fibonacci (for farm hand cost calculation)
# =============================================================================

def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (1-indexed: fib(1)=1, fib(2)=1, fib(3)=2, ...)."""
    if n <= 0:
        return 0
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


def farm_hand_cost(hires_already_today: int, cost_mult: int = 1) -> int:
    """Cost to hire the next farm hand, given how many have already been hired today."""
    return cost_mult * fibonacci(hires_already_today + 1)
