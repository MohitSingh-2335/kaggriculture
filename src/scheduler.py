"""
Multi-unit task scheduler for Kaggriculture.

Assigns the farmer and hired farm hands to prioritized tasks each turn.
Prevents duplicate work (two units doing the same task).

Priority order (intelligence-driven):
1. Feed animals (they die after 2 unfed days)
2. Water plants (CRITICAL: melons in ages 6-12 MUST be watered for bonus yield)
3. Harvest ready animals (collect products before max_held cap)
4. Harvest ready plants (collect yield before decay)
5. Collect fertilizer from animals
6. Plant new crops (if seeds available and empty tiles)
7. Build structures (BUILD_COOP/BUILD_PASTURE for animals in shed)
8. Place animals (PICKUP from shed -> move to structure -> PLACE)
9. Drop inventory at shed (MUST happen before SELL can see items)
10. Dig weeds (PRIORITY_DIG_PRODUCTIVE=8 if blocking plant/build, else PRIORITY_DIG=9)
11. Care for animals (bonus yield)
"""

import os
from typing import List, Tuple, Optional, Dict, Any
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from .game_state import GameState, Plant, Animal, Weed, Farm, EmptyStructure
from .actions import (
    manhattan_distance, move_toward, is_shed_adjacent, nearest_shed_tile,
    path_to, pickup_action,
)
from .constants import ALL_ANIMALS, ANIMAL_DATA, SHED_TILES, BOARD_SIZE, MARKET_PARAMS, CROP_DATA
from .strategy import (
    get_build_targets, get_planting_targets, _animal_matches_structure,
    TARGET_COWS, TARGET_SHEEP,
)


# Task types and their priorities (lower = higher priority)
PRIORITY_FEED = 1            # Never loses to anything: prevents animal starvation & escape
PRIORITY_WATER = 2           # Critical: crops die after 2 unwatered days
PRIORITY_BUILD = 3           # Build structures immediately when unhoused animals exist
PRIORITY_HARVEST_ANIMAL = 4  # High-value daily animal yields (milk, wool)
PRIORITY_HARVEST_PLANT = 5   # Harvest mature crops for continuous cash flow
PRIORITY_COLLECT_FERTILIZER = 6
PRIORITY_PLANT = 7           # Plant seeds
PRIORITY_PLACE = 7           # Place animals into structures
PRIORITY_DROP = 8            # Drop harvested items to shed
PRIORITY_CARE = 9            # Demoted: WATER always beats CARE (melon bonus window)
PRIORITY_DIG = 9             # Standard weed digging (reactive state machine repairs productive tiles)


class Task:
    """A pending task that needs a unit to execute it."""

    def __init__(self, priority: int, action: List[str], target_pos: Tuple[int, int],
                 description: str = ""):
        self.priority = priority
        self.action = action          # The action to execute when at target
        self.target_pos = target_pos  # Where the unit needs to be
        self.description = description
        self.assigned_to: Optional[int] = None  # Unit index (0=farmer, 1+=hands)

    def __repr__(self):
        return f"Task({self.description}, pos={self.target_pos}, pri={self.priority})"


def generate_tasks(
    state: GameState,
    plant_targets: Optional[List[Tuple[int, int, str]]] = None,
    build_targets: Optional[List[Tuple[int, int, str]]] = None,
) -> List[Task]:
    """
    Generate all pending tasks from the current game state.
    Returns tasks sorted by priority (most urgent first).
    """
    tasks: List[Task] = []
    farm = state.my_farm

    # --- Feed animals (CRITICAL: they escape after 2 unfed days) ---
    # Feeding requires 1 WHEAT in unit inventory.
    animals_needing_feed = farm.animals_needing_feed()
    if animals_needing_feed:
        # Prioritize endangered animals (consecutive_unfed >= 1) first, then older investments
        animals_needing_feed.sort(
            key=lambda a: (getattr(a, "consecutive_unfed", 0), -getattr(a, "placed_day", 0)),
            reverse=True,
        )
        wheat_in_shed = state.private.shed.get("WHEAT", 0)
        units_carrying_wheat = sum(
            1 for i in range(1 + len(farm.hands_positions))
            if (state.private.farmer_inventory() if i == 0 else state.private.hand_inventory(i - 1)).get("WHEAT", 0) > 0
        )
        for animal in animals_needing_feed:
            tasks.append(Task(
                priority=PRIORITY_FEED,
                action=["FEED"],
                target_pos=(animal.x, animal.y),
                description=f"Feed {animal.animal_type} at ({animal.x},{animal.y})"
            ))
        if units_carrying_wheat < len(animals_needing_feed) and wheat_in_shed > 0:
            needed_pickups = len(animals_needing_feed) - units_carrying_wheat
            for i in range(min(wheat_in_shed, needed_pickups)):
                shed_tile = SHED_TILES[i % len(SHED_TILES)]
                tasks.append(Task(
                    priority=PRIORITY_FEED,
                    action=pickup_action("WHEAT", 1),
                    target_pos=shed_tile,
                    description="Pickup wheat from shed for feeding"
                ))

    is_terminal_phase = (os.environ.get("DISABLE_TERMINAL_RECALL") != "1") and (
        (getattr(state, "step", 0) >= 711) or (state.day > 29 or (state.day == 29 and state.hour >= 15))
    )

    # --- Water plants (CRITICAL: they die after 2 unwatered days) ---
    # Intelligence: melons in ages 6-12 MUST be watered for bonus yield
    # At step >= 711 (Day 29 Hour 15+), cancel watering as crops cannot mature before game end
    if not is_terminal_phase:
        for plant in farm.plants_needing_water():
            tasks.append(Task(
                priority=PRIORITY_WATER,
                action=["WATER"],
                target_pos=(plant.x, plant.y),
                description=f"Water {plant.crop} at ({plant.x},{plant.y})"
            ))

    # --- Harvest animals ---
    for animal in farm.harvestable_animals():
        tasks.append(Task(
            priority=PRIORITY_HARVEST_ANIMAL,
            action=["HARVEST"],
            target_pos=(animal.x, animal.y),
            description=f"Harvest {animal.animal_type} at ({animal.x},{animal.y})"
        ))

    # --- Harvest plants ---
    for plant in farm.harvestable_plants(state.day):
        tasks.append(Task(
            priority=PRIORITY_HARVEST_PLANT,
            action=["HARVEST"],
            target_pos=(plant.x, plant.y),
            description=f"Harvest {plant.crop} at ({plant.x},{plant.y})"
        ))

    # --- Collect fertilizer from animals ---
    for animal in farm.animals_with_fertilizer():
        tasks.append(Task(
            priority=PRIORITY_COLLECT_FERTILIZER,
            action=["COLLECT_FERTILIZER"],
            target_pos=(animal.x, animal.y),
            description=f"Collect fertilizer from {animal.animal_type} at ({animal.x},{animal.y})"
        ))

    # --- BUILD tasks (priority 7: between PLANT and DROP) ---
    # Cancelled at step >= 711 as structures cannot be populated before game end
    if not is_terminal_phase:
        if build_targets is None:
            build_targets = get_build_targets(state)
        for bx, by, baction in build_targets:
            tasks.append(Task(
                priority=PRIORITY_BUILD,
                action=[baction],
                target_pos=(bx, by),
                description=f"{baction} at ({bx},{by})"
            ))

    # --- PLACE tasks: 3-step animal chain (priority 6: between HARVEST and PLANT) ---
    empty_structures = list(farm.empty_structures())
    claimed_structures = set()

    # Step 2 -> 3: Units currently carrying an animal -> move to structure -> PLACE
    for unit_idx in range(1 + len(farm.hands_positions)):
        if unit_idx == 0:
            inv = state.private.farmer_inventory()
            pos = farm.farmer_pos
        else:
            inv = state.private.hand_inventory(unit_idx - 1)
            pos = farm.hands_positions[unit_idx - 1] if unit_idx - 1 < len(farm.hands_positions) else None

        if not inv or pos is None:
            continue

        for animal_type in ALL_ANIMALS:
            if inv.get(animal_type, 0) > 0:
                matched_struct = False
                for struct in empty_structures:
                    if (struct.x, struct.y) not in claimed_structures and _animal_matches_structure(animal_type, struct.structure):
                        claimed_structures.add((struct.x, struct.y))
                        tasks.append(Task(
                            priority=PRIORITY_PLACE,
                            action=["PLACE", animal_type],
                            target_pos=(struct.x, struct.y),
                            description=f"Place {animal_type} at ({struct.x},{struct.y}) for unit {unit_idx}"
                        ))
                        matched_struct = True
                        break
                if not matched_struct:
                    shed_tile = pos if is_shed_adjacent(pos) else nearest_shed_tile(pos)
                    tasks.append(Task(
                        priority=PRIORITY_DROP,
                        action=["DROP"],
                        target_pos=shed_tile,
                        description=f"Emergency drop unplaceable {animal_type} at shed for unit {unit_idx}"
                    ))

    # Step 1: Animals in shed with matching empty structure on farm -> PICKUP from shed
    # Fair round-robin across species: prioritize whichever species is proportionally further behind on farm
    shed = state.private.shed
    available_structures = [s for s in empty_structures if (s.x, s.y) not in claimed_structures]

    cows_on_farm = len([a for a in farm.all_animals() if a.animal_type == "COW"]) + sum(inv.get("COW", 0) for inv in state.private.inventories)
    sheep_on_farm = len([a for a in farm.all_animals() if a.animal_type == "SHEEP"]) + sum(inv.get("SHEEP", 0) for inv in state.private.inventories)

    target_sheep = float(TARGET_SHEEP) if TARGET_SHEEP > 0 else 6.0
    target_cows = float(TARGET_COWS) if TARGET_COWS > 0 else 8.0
    if (sheep_on_farm / target_sheep) < (cows_on_farm / target_cows):
        species_order = ["SHEEP", "COW", "GOOSE"]
    else:
        species_order = ["COW", "SHEEP", "GOOSE"]

    shed_counts = {a: shed.get(a, 0) for a in ALL_ANIMALS}
    animal_pickup_queue = []
    while any(c > 0 for c in shed_counts.values()):
        for a in species_order:
            if shed_counts[a] > 0:
                animal_pickup_queue.append(a)
                shed_counts[a] -= 1

    debug_on = os.environ.get("DEBUG_ANIMAL_LOGISTICS") == "1"
    if debug_on and shed.get("SHEEP", 0) > 0:
        matching_sheep = len([s for s in available_structures if _animal_matches_structure("SHEEP", s.structure)])
        matching_cow = len([s for s in available_structures if _animal_matches_structure("COW", s.structure)])
        print(f"[D{state.day:02d}:H{state.hour:02d}] SHEEP in shed={shed.get('SHEEP', 0)}, COW in shed={shed.get('COW', 0)}")
        print(f"  Queue: {animal_pickup_queue}")
        print(f"  Available structures: total={len(available_structures)}, match_sheep={matching_sheep}, match_cow={matching_cow}")

    animal_pickup_idx = 0
    for animal_type in animal_pickup_queue:
        matching_struct = None
        for struct in available_structures:
            if (struct.x, struct.y) not in claimed_structures and _animal_matches_structure(animal_type, struct.structure):
                matching_struct = struct
                break
        if matching_struct is not None:
            claimed_structures.add((matching_struct.x, matching_struct.y))
            shed_tile = SHED_TILES[animal_pickup_idx % len(SHED_TILES)]
            animal_pickup_idx += 1
            tasks.append(Task(
                priority=PRIORITY_PLACE,
                action=pickup_action(animal_type, 1),
                target_pos=shed_tile,
                description=f"Pickup {animal_type} from shed"
            ))
            if debug_on and shed.get("SHEEP", 0) > 0:
                print(f"  -> {animal_type}: matched struct at ({matching_struct.x},{matching_struct.y}), CLAIMED, appended PICKUP task to ({shed_tile[0]},{shed_tile[1]})")
        else:
            if debug_on and shed.get("SHEEP", 0) > 0:
                print(f"  -> {animal_type}: NO matching struct available, SKIPPED")

    # --- DROP inventory at shed ---
    # Intelligence: HARVEST -> unit inventory (NOT shed). Must DROP at shed
    # for SELL to see the items. This is a critical logistics step.
    for unit_idx in range(1 + len(farm.hands_positions)):
        if unit_idx == 0:
            inv = state.private.farmer_inventory()
            pos = farm.farmer_pos
        else:
            inv = state.private.hand_inventory(unit_idx - 1)
            pos = farm.hands_positions[unit_idx - 1] if unit_idx - 1 < len(farm.hands_positions) else None

        if not inv or pos is None:
            continue

        # If carrying sellable crops, products, or excess items, drop at shed
        has_items = any(
            v > 0 for k, v in inv.items()
            if k not in ("COW", "SHEEP", "GOOSE")  # Don't drop animals meant for placing
        )
        if has_items and not is_shed_adjacent(pos):
            shed_tile = nearest_shed_tile(pos)
            tasks.append(Task(
                priority=PRIORITY_DROP,
                action=["DROP"],
                target_pos=shed_tile,
                description=f"Drop inventory at shed for unit {unit_idx}"
            ))

    # --- Care for animals (bonus yield) ---
    for animal in farm.all_animals():
        tasks.append(Task(
            priority=PRIORITY_CARE,
            action=["CARE"],
            target_pos=(animal.x, animal.y),
            description=f"Care for {animal.animal_type} at ({animal.x},{animal.y})"
        ))

    # --- PLANT tasks (priority 7: between HARVEST and DROP) ---
    if not is_terminal_phase:
        if plant_targets is None:
            plant_targets = get_planting_targets(state)
        for px, py, crop in plant_targets:
            tasks.append(Task(
                priority=PRIORITY_PLANT,
                action=["PLANT", crop],
                target_pos=(px, py),
                description=f"Plant {crop} at ({px},{py})"
            ))

    # --- Weeds (demoted to background priority 9) ---
    for weed in farm.all_weeds():
        tasks.append(Task(
            priority=PRIORITY_DIG,
            action=["DIG"],
            target_pos=(weed.x, weed.y),
            description=f"Dig weed at ({weed.x},{weed.y})"
        ))

    tasks.sort(key=lambda t: t.priority)
    return tasks


# --- Weed Repair State Machine ---
_WEED_REPLAY_STEPS = 8
_pending_weed_repairs: Dict[int, Dict[str, Any]] = {}


# --- Unit Path Commitments & State ---
_unit_paths: Dict[int, Dict[str, Any]] = {}
_last_day_seen: int = -1


def reset_scheduler_state():
    """Reset scheduler state machines (e.g. between matches or in test suites)."""
    global _pending_weed_repairs, _unit_paths, _last_day_seen
    _pending_weed_repairs.clear()
    _unit_paths.clear()
    _last_day_seen = -1


def _clean_weed_repairs(farm: Farm):
    """Clean up expired repairs."""
    expired_units = []
    for u_idx, repair in _pending_weed_repairs.items():
        repair["turns_left"] -= 1
        if repair["turns_left"] <= 0:
            expired_units.append(u_idx)
    for u_idx in expired_units:
        del _pending_weed_repairs[u_idx]


def get_unit_positions(farm: Farm) -> List[Tuple[int, Tuple[int, int]]]:
    """Return list of (unit_idx, pos)."""
    units = [(0, farm.farmer_pos)]
    for i, pos in enumerate(farm.hands_positions):
        units.append((i + 1, pos))
    return units


def _step_toward(
    unit_idx: int,
    unit_pos: Tuple[int, int],
    target_pos: Tuple[int, int],
    action: Optional[List[str]] = None,
) -> str:
    """
    Get the next directional step toward target_pos using BFS shortest-path routing
    with per-unit path caching. Falls back to move_toward if path_to finds no path.
    """
    if unit_idx in _unit_paths and _unit_paths[unit_idx].get("target_pos") == target_pos and _unit_paths[unit_idx].get("path"):
        return _unit_paths[unit_idx]["path"].pop(0)

    full_path = path_to(unit_pos, target_pos, BOARD_SIZE)
    if full_path:
        step = full_path.pop(0)
        _unit_paths[unit_idx] = {
            "target_pos": target_pos,
            "path": full_path,
            "action": list(action) if action else [],
        }
        return step
    return move_toward(unit_pos, target_pos)


def _animal_asset_preservation_value(animal: Animal, state: GameState) -> float:
    """
    Calculate the remaining amortized value of an animal asset:
    buy_cost minus value already extracted, with a $60 safety floor.
    Matches the asset preservation logic in src/strategy.py:_calculate_wheat_to_buy.
    """
    atype = animal.animal_type if isinstance(animal.animal_type, str) else "COW"
    info = ANIMAL_DATA.get(atype, {})
    buy_cost = float(info.get("buy_cost", 400))
    interval = float(info.get("yield_interval", 2))
    ttf = float(info.get("time_to_first_yield", 8))
    prod = info.get("product", "MILK")
    spot_p = float(state.market.price_of(prod))
    if spot_p <= 0:
        spot_p = float(info.get("base_price", 160))

    days_active = max(0, state.day - getattr(animal, "placed_day", 0))
    days_producing = max(0, days_active - int(ttf))
    yields_so_far = days_producing // int(interval)
    value_extracted = yields_so_far * spot_p
    return max(60.0, buy_cost - value_extracted)


def _score_task_value(task: Task, state: GameState) -> float:
    """
    Score the estimated economic value in coins of completing a pending task.
    Uses existing constants from ANIMAL_DATA, CROP_DATA, and market prices.
    """
    farm = state.my_farm
    tx, ty = task.target_pos
    action = task.action

    if not action:
        return 10.0

    act_type = action[0]

    # 1. Animal Harvest
    if task.priority == PRIORITY_HARVEST_ANIMAL or act_type == "HARVEST":
        cell = farm.tiles[ty][tx] if (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE) else None
        if isinstance(cell, Animal):
            atype = cell.animal_type if isinstance(cell.animal_type, str) else "COW"
            info = ANIMAL_DATA.get(atype, {})
            prod = info.get("product", "MILK")
            spot = float(state.market.price_of(prod))
            base = float(info.get("base_price", 160))
            return max(spot, base if spot <= 0 else spot)

    # 2. Feed Animals
    if task.priority == PRIORITY_FEED or act_type == "FEED":
        cell = farm.tiles[ty][tx] if (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE) else None
        if isinstance(cell, Animal):
            amortized = _animal_asset_preservation_value(cell, state)
            unfed = getattr(cell, "consecutive_unfed", 0)

            if unfed >= 1:
                # Acute danger: 1 day from death! Must decisively beat any competing harvest ($150-250)
                # Remaining amortized asset value plus urgency margin scaling with hour of day.
                urgency = state.hour * 5.0
                return max(400.0, amortized + 100.0) + urgency

            # Routine feeding (consecutive_unfed == 0):
            # Scale up with time-of-day so delivery completes with margin before Hour 23 cutoff.
            # Morning (Hour 0-6): routine daily yield EV (routine chores can interleave).
            # Midday/Afternoon (Hour 7-23): progressively ramps toward full asset preservation.
            atype = cell.animal_type if isinstance(cell.animal_type, str) else "COW"
            info = ANIMAL_DATA.get(atype, {})
            interval = float(info.get("yield_interval", 2))
            prod = info.get("product", "MILK")
            spot = float(state.market.price_of(prod))
            base = float(info.get("base_price", 160))
            p = spot if spot > 0 else base
            base_val = (1.0 / interval) * p

            margin_frac = min(1.0, max(0.0, (state.hour - 6) / 14.0))
            target_asset_val = max(220.0, amortized * 0.8)
            return (1.0 - margin_frac) * base_val + margin_frac * target_asset_val

        # Pickup wheat from shed for feeding: inherits urgency from animals needing feed
        animals_needing_feed = [a for a in farm.all_animals() if a.needs_feed]
        if not animals_needing_feed:
            return 70.0

        endangered = [a for a in animals_needing_feed if getattr(a, "consecutive_unfed", 0) >= 1]
        if endangered:
            # Animal is 1 day away from death; picking up wheat is prerequisite to saving it!
            max_amortized = max(_animal_asset_preservation_value(a, state) for a in endangered)
            return max(400.0, max_amortized + 100.0) + (state.hour * 5.0)

        # Routine feeding: scale pickup value to mirror routine feed value ramp
        margin_frac = min(1.0, max(0.0, (state.hour - 6) / 14.0))
        return (1.0 - margin_frac) * 70.0 + margin_frac * 240.0


    # 3. Plant Harvest
    if task.priority == PRIORITY_HARVEST_PLANT:
        cell = farm.tiles[ty][tx] if (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE) else None
        if isinstance(cell, Plant):
            crop = cell.crop if isinstance(cell.crop, str) else "WHEAT"
            crop_info = CROP_DATA.get(crop, {})
            spot = float(state.market.price_of(crop))
            base = float(crop_info.get("base_price", 25))
            p = spot if spot > 0 else base
            return p * max(1, getattr(cell, "yield_units", 1))

    # 4. Water Plants
    if task.priority == PRIORITY_WATER or act_type == "WATER":
        cell = farm.tiles[ty][tx] if (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE) else None
        if isinstance(cell, Plant):
            crop = cell.crop if isinstance(cell.crop, str) else "WHEAT"
            crop_info = CROP_DATA.get(crop, {})
            spot = float(state.market.price_of(crop))
            base = float(crop_info.get("base_price", 25))
            p = spot if spot > 0 else base
            if getattr(cell, "consecutive_unwatered", 0) >= 1:
                return p + float(crop_info.get("seed_cost", 10))
            planted_day = getattr(cell, "planted_day", 0)
            if crop == "MELON" and 6 <= (state.day - planted_day) <= 12:
                return 60.0
            return 35.0

    # 5. Collect Fertilizer
    if task.priority == PRIORITY_COLLECT_FERTILIZER or act_type == "COLLECT_FERTILIZER":
        spot = float(state.market.price_of("FERTILIZER"))
        return spot if spot > 0 else 20.0

    # 6. Build Structure
    if task.priority == PRIORITY_BUILD or act_type.startswith("BUILD"):
        return 60.0

    # 7. Animal Place / Logistics
    if task.priority == PRIORITY_PLACE or act_type == "PLACE":
        return 50.0

    # 8. Plant Seeds
    if task.priority == PRIORITY_PLANT or act_type == "PLANT":
        crop = action[1] if len(action) > 1 else "WHEAT"
        if crop == "MELON":
            return 45.0
        return 20.0

    # 9. Drop
    if act_type == "DROP":
        return 15.0

    return 10.0


def assign_tasks(
    state: GameState,
    tasks: List[Task],
    plant_targets: Optional[List[Tuple[int, int, str]]] = None,
    build_targets: Optional[List[Tuple[int, int, str]]] = None
) -> Tuple[List[str], List[List[str]]]:
    """
    Assign tasks to units greedily based on priority and distance.
    Unit movement uses BFS shortest-path routing with persistent path caching.
    Returns (farmer_action, hands_actions).
    """
    global _last_day_seen, _unit_paths
    farm = state.my_farm
    units = get_unit_positions(farm)  # List of (unit_idx, (x, y))

    # Reset unit path commitments on day rollover (all hands are dismissed and re-spawn at shed)
    if state.day != _last_day_seen:
        _unit_paths.clear()
        _last_day_seen = state.day

    is_terminal_phase = (os.environ.get("DISABLE_TERMINAL_RECALL") != "1") and (
        (getattr(state, "step", 0) >= 711) or (state.day > 29 or (state.day == 29 and state.hour >= 15))
    )

    unit_actions = {}
    claimed_positions = set()

    # --- Reactive Weed-Repair State Machine ---
    _clean_weed_repairs(farm)
    if not is_terminal_phase:
        for unit_idx, unit_pos in units:
            if unit_idx in _pending_weed_repairs:
                repair = _pending_weed_repairs[unit_idx]
                tx, ty = repair["target_pos"]
                intended = repair["intended_action"]
                if unit_pos == (tx, ty) and not isinstance(farm.tiles[ty][tx], Weed):
                    unit_actions[unit_idx] = intended
                    claimed_positions.add((tx, ty))
                    del _pending_weed_repairs[unit_idx]
                    if unit_idx in _unit_paths:
                        del _unit_paths[unit_idx]
                elif unit_pos != (tx, ty):
                    direction = _step_toward(unit_idx, unit_pos, (tx, ty))
                    unit_actions[unit_idx] = [direction]
                    claimed_positions.add((tx, ty))
    else:
        _pending_weed_repairs.clear()

    # Inject dynamic targets
    if build_targets and not is_terminal_phase:
        existing_build_targets = {t.target_pos for t in tasks if t.priority == PRIORITY_BUILD}
        for x, y, build_action in build_targets:
            if (x, y) not in existing_build_targets:
                tasks.append(Task(
                    priority=PRIORITY_BUILD,
                    action=[build_action],
                    target_pos=(x, y),
                    description=f"Build at ({x},{y})"
                ))
        tasks.sort(key=lambda t: t.priority)

    if plant_targets and not is_terminal_phase:
        existing_plant_targets = {t.target_pos for t in tasks if t.priority == PRIORITY_PLANT}
        for x, y, crop in plant_targets:
            if (x, y) not in existing_plant_targets:
                tasks.append(Task(
                    priority=PRIORITY_PLANT,
                    action=["PLANT", crop],
                    target_pos=(x, y),
                    description=f"Plant {crop} at ({x},{y})"
                ))
        tasks.sort(key=lambda t: t.priority)

    # --- Terminal Pre-Liquidation Hand Recall (Step >= 711 / Day 29 Hour 15+) ---
    # Force every hand (and farmer) carrying inventory to path directly to the nearest shed tile and DROP,
    # overriding any other task so all produce reaches shed before terminal liquidation at step 717.
    if is_terminal_phase:
        for unit_idx, unit_pos in units:
            if unit_idx in unit_actions:
                continue
            inv = state.private.farmer_inventory() if unit_idx == 0 else state.private.hand_inventory(unit_idx - 1)
            has_items = any(
                v > 0 for k, v in inv.items()
                if k not in ("COW", "SHEEP", "GOOSE")
            )
            if has_items:
                if is_shed_adjacent(unit_pos):
                    unit_actions[unit_idx] = ["DROP"]
                    if unit_idx in _unit_paths:
                        del _unit_paths[unit_idx]
                else:
                    shed_tile = nearest_shed_tile(unit_pos)
                    direction = _step_toward(unit_idx, unit_pos, shed_tile)
                    unit_actions[unit_idx] = [direction]

    MAX_ANIMAL_LOGISTICS_PER_TURN = 2
    animal_logistics_assigned = 0

    # Units carrying animals: prioritize assigning them directly to their PLACE task
    # Never block placing an animal already in hand
    for unit_idx, unit_pos in units:
        if unit_idx in unit_actions:
            continue
        inv = state.private.farmer_inventory() if unit_idx == 0 else state.private.hand_inventory(unit_idx - 1)
        for task in tasks:
            if task.priority == PRIORITY_PLACE and len(task.action) >= 2 and task.action[0] == "PLACE":
                animal_type = task.action[1]
                if inv.get(animal_type, 0) > 0 and task.target_pos not in claimed_positions:
                    if unit_pos == task.target_pos:
                        unit_actions[unit_idx] = task.action
                        if unit_idx in _unit_paths:
                            del _unit_paths[unit_idx]
                    else:
                        direction = _step_toward(unit_idx, unit_pos, task.target_pos, task.action)
                        unit_actions[unit_idx] = [direction]
                    claimed_positions.add(task.target_pos)
                    task.assigned_to = unit_idx
                    animal_logistics_assigned += 1
                    break

    use_value_density = (os.environ.get("SCHEDULER_ASSIGNMENT", "value_density").lower() != "greedy")

    if use_value_density:
        # Value-density task assignment (coins-per-turn matching)
        unassigned_units = {u_idx: u_pos for u_idx, u_pos in units if u_idx not in unit_actions}
        eligible_tasks = [t for t in tasks if t.target_pos not in claimed_positions]

        while unassigned_units and eligible_tasks:
            best_pair = None
            best_score = -1.0

            for t in eligible_tasks:
                if t.target_pos in claimed_positions:
                    continue
                if t.priority == PRIORITY_PLACE and animal_logistics_assigned >= MAX_ANIMAL_LOGISTICS_PER_TURN:
                    continue

                t_val = _score_task_value(t, state)

                for u_idx, u_pos in unassigned_units.items():
                    if t.action == ["FEED"]:
                        u_inv = state.private.farmer_inventory() if u_idx == 0 else state.private.hand_inventory(u_idx - 1)
                        if u_inv.get("WHEAT", 0) <= 0:
                            continue

                    # Units already carrying wheat must deliver it, not hoard extra wheat from shed
                    if len(t.action) >= 2 and t.action[0] == "PICKUP" and t.action[1] == "WHEAT":
                        u_inv = state.private.farmer_inventory() if u_idx == 0 else state.private.hand_inventory(u_idx - 1)
                        if u_inv.get("WHEAT", 0) > 0:
                            continue

                    dist = manhattan_distance(u_pos, t.target_pos)
                    density = t_val / (1.0 + dist)
                    if density > best_score:
                        best_score = density
                        best_pair = (u_idx, u_pos, t)

            if best_pair is None:
                break

            u_idx, u_pos, t = best_pair
            del unassigned_units[u_idx]
            eligible_tasks.remove(t)

            if u_pos == t.target_pos:
                if t.action and (t.action[0] == "PLANT" or t.action[0].startswith("BUILD")):
                    tx, ty = t.target_pos
                    if isinstance(farm.tiles[ty][tx], Weed):
                        _pending_weed_repairs[u_idx] = {
                            "intended_action": list(t.action),
                            "target_pos": (tx, ty),
                            "turns_left": _WEED_REPLAY_STEPS,
                        }
                        unit_actions[u_idx] = ["DIG"]
                        if u_idx in _unit_paths:
                            del _unit_paths[u_idx]
                    else:
                        unit_actions[u_idx] = t.action
                        if u_idx in _unit_paths:
                            del _unit_paths[u_idx]
                else:
                    unit_actions[u_idx] = t.action
                    if u_idx in _unit_paths:
                        del _unit_paths[u_idx]
            else:
                direction = _step_toward(u_idx, u_pos, t.target_pos, t.action)
                unit_actions[u_idx] = [direction]

            claimed_positions.add(t.target_pos)
            t.assigned_to = u_idx
            if t.priority == PRIORITY_PLACE:
                animal_logistics_assigned += 1
    else:
        # Greedy assignment fallback:
        # Units carrying wheat: prioritize assigning them to FEED task (endangered animals first, then nearest)
        for unit_idx, unit_pos in units:
            if unit_idx in unit_actions:
                continue
            inv = state.private.farmer_inventory() if unit_idx == 0 else state.private.hand_inventory(unit_idx - 1)
            if inv.get("WHEAT", 0) > 0:
                available_feeds = [
                    t for t in tasks
                    if t.priority == PRIORITY_FEED and t.action == ["FEED"] and t.target_pos not in claimed_positions
                ]
                if available_feeds:
                    def _feed_key(t):
                        tx, ty = t.target_pos
                        cell = farm.tiles[ty][tx] if (0 <= tx < BOARD_SIZE and 0 <= ty < BOARD_SIZE) else None
                        unfed = getattr(cell, "consecutive_unfed", 0) if isinstance(cell, Animal) else 0
                        dist = manhattan_distance(unit_pos, t.target_pos)
                        return (-unfed, dist)

                    available_feeds.sort(key=_feed_key)
                    best_task = available_feeds[0]
                    if unit_pos == best_task.target_pos:
                        unit_actions[unit_idx] = best_task.action
                        if unit_idx in _unit_paths:
                            del _unit_paths[unit_idx]
                    else:
                        direction = _step_toward(unit_idx, unit_pos, best_task.target_pos, best_task.action)
                        unit_actions[unit_idx] = [direction]
                    claimed_positions.add(best_task.target_pos)
                    best_task.assigned_to = unit_idx

        # For remaining tasks, assign the nearest available unit
        for task in tasks:

            if task.target_pos in claimed_positions:
                continue

            # Cap animal placement/pickup logistics (PLACE / PICKUP) so crops are never starved
            # Structure construction (BUILD) is uncapped to eliminate pasture lag
            if task.priority == PRIORITY_PLACE:
                if animal_logistics_assigned >= MAX_ANIMAL_LOGISTICS_PER_TURN:
                    continue

            best_unit = None
            best_dist = float('inf')

            for unit_idx, unit_pos in units:
                if unit_idx in unit_actions:
                    continue  # Already assigned
                # If task is FEED, unit MUST have WHEAT in inventory
                if task.action == ["FEED"]:
                    u_inv = state.private.farmer_inventory() if unit_idx == 0 else state.private.hand_inventory(unit_idx - 1)
                    if u_inv.get("WHEAT", 0) <= 0:
                        continue
                dist = manhattan_distance(unit_pos, task.target_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_unit = (unit_idx, unit_pos)

            if best_unit is None:
                continue  # No eligible unit for THIS task; keep assigning others

            unit_idx, unit_pos = best_unit

            if unit_pos == task.target_pos:
                # At target — check if intended action is PLANT or BUILD on a weedy tile
                if task.action and (task.action[0] == "PLANT" or task.action[0].startswith("BUILD")):
                    tx, ty = task.target_pos
                    if isinstance(farm.tiles[ty][tx], Weed):
                        _pending_weed_repairs[unit_idx] = {
                            "intended_action": list(task.action),
                            "target_pos": (tx, ty),
                            "turns_left": _WEED_REPLAY_STEPS,
                        }
                        unit_actions[unit_idx] = ["DIG"]
                        if unit_idx in _unit_paths:
                            del _unit_paths[unit_idx]
                    else:
                        unit_actions[unit_idx] = task.action
                        if unit_idx in _unit_paths:
                            del _unit_paths[unit_idx]
                else:
                    unit_actions[unit_idx] = task.action
                    if unit_idx in _unit_paths:
                        del _unit_paths[unit_idx]
            else:
                # Not at target — move toward it using BFS shortest path
                direction = _step_toward(unit_idx, unit_pos, task.target_pos, task.action)
                unit_actions[unit_idx] = [direction]

            claimed_positions.add(task.target_pos)
            task.assigned_to = unit_idx
            if task.priority == PRIORITY_PLACE:
                animal_logistics_assigned += 1

    # --- Handle drop inventory at shed for unassigned units with items ---
    for unit_idx, unit_pos in units:
        if unit_idx in unit_actions:
            continue

        if unit_idx == 0:
            inv = state.private.farmer_inventory()
        else:
            inv = state.private.hand_inventory(unit_idx - 1)

        if inv and sum(inv.values()) > 0:
            if is_shed_adjacent(unit_pos):
                unit_actions[unit_idx] = ["DROP"]
                if unit_idx in _unit_paths:
                    del _unit_paths[unit_idx]
            else:
                shed_tile = nearest_shed_tile(unit_pos)
                direction = _step_toward(unit_idx, unit_pos, shed_tile)
                unit_actions[unit_idx] = [direction]
            continue

    # --- Default: unassigned units PASS ---
    for unit_idx, _ in units:
        if unit_idx not in unit_actions:
            unit_actions[unit_idx] = ["PASS"]
            if unit_idx in _unit_paths:
                del _unit_paths[unit_idx]

    # Extract farmer action and hands actions
    farmer_action = unit_actions.get(0, ["PASS"])
    hands_actions = []
    for i in range(len(farm.hands_positions)):
        hands_actions.append(unit_actions.get(i + 1, ["PASS"]))

    return farmer_action, hands_actions

