"""
Typed game state model for Kaggriculture.

Parses the raw observation dict into clean dataclasses with helper methods.
Zero external dependencies — uses only Python stdlib.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from .constants import (
    BOARD_SIZE, TURNS_PER_DAY, SHED_TILES, CROP_DATA, ANIMAL_DATA,
)


# =============================================================================
# Tile Types
# =============================================================================

@dataclass
class Plant:
    """A planted crop on a tile."""
    crop: str                    # "WHEAT", "CARROT", etc.
    planted_day: int
    watered_today: bool
    consecutive_unwatered: int
    yield_units: int
    max_lifespan_step: int       # -1 for ongoing crops before cap
    fertilized_until_day: int    # -1 if not fertilized
    x: int = 0
    y: int = 0

    @property
    def age_days(self) -> int:
        """Days since planting (requires external day context; use GameState.plant_age())."""
        raise NotImplementedError("Plant.age_days requires current day; use GameState.plant_age(plant)")

    @property
    def is_harvestable(self) -> bool:
        return self.yield_units > 0

    @property
    def needs_water(self) -> bool:
        return not self.watered_today

    @property
    def is_fertilized(self) -> bool:
        return self.fertilized_until_day >= 0


@dataclass
class Animal:
    """An animal occupying a structure tile."""
    animal_type: str             # "GOOSE", "COW", "SHEEP"
    structure: str               # "COOP" or "PASTURE"
    placed_day: int
    yield_units: int
    fed_today: bool
    consecutive_unfed: int
    cared_today: bool
    fertilizer_available: bool
    pending_care_bonus: int
    x: int = 0
    y: int = 0

    @property
    def is_harvestable(self) -> bool:
        return self.yield_units > 0

    @property
    def needs_feed(self) -> bool:
        return not self.fed_today

    @property
    def needs_care(self) -> bool:
        return not self.cared_today

    @property
    def can_collect_fertilizer(self) -> bool:
        return self.fertilizer_available


@dataclass
class EmptyStructure:
    """A coop or pasture without an animal."""
    structure: str  # "COOP" or "PASTURE"
    x: int = 0
    y: int = 0


@dataclass
class Weed:
    """A weed on a tile — must be dug before the tile can be used."""
    x: int = 0
    y: int = 0


# =============================================================================
# Farm
# =============================================================================

@dataclass
class Farm:
    """One player's farm state (public, visible to both players)."""
    money: float
    tiles: List[List[Any]]  # 2D grid: None, "LOCKED", Plant, Animal, EmptyStructure, Weed
    farmer_pos: Tuple[int, int]       # (x, y)
    hands_positions: List[Tuple[int, int]]  # [(x, y), ...]
    unlocked_quadrants: List[str]     # ["NW", "NE", ...]
    hires_today: int

    def tile_at(self, x: int, y: int) -> Any:
        """Get tile at (x, y). Returns None, 'LOCKED', or a tile object."""
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            return self.tiles[y][x]
        return "LOCKED"

    def empty_tiles(self) -> List[Tuple[int, int]]:
        """All unlocked empty tiles (None)."""
        result = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.tiles[y][x] is None:
                    result.append((x, y))
        return result

    def all_plants(self) -> List[Plant]:
        """All plants on the farm."""
        result = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                t = self.tiles[y][x]
                if isinstance(t, Plant):
                    result.append(t)
        return result

    def all_animals(self) -> List[Animal]:
        """All animals on the farm."""
        result = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                t = self.tiles[y][x]
                if isinstance(t, Animal):
                    result.append(t)
        return result

    def all_weeds(self) -> List[Weed]:
        """All weeds on the farm."""
        result = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                t = self.tiles[y][x]
                if isinstance(t, Weed):
                    result.append(t)
        return result

    def plants_needing_water(self) -> List[Plant]:
        """Plants that haven't been watered today."""
        return [p for p in self.all_plants() if p.needs_water]

    def animals_needing_feed(self) -> List[Animal]:
        """Animals that haven't been fed today."""
        return [a for a in self.all_animals() if a.needs_feed]

    def harvestable_plants(self, current_day: Optional[int] = None) -> List[Plant]:
        """Plants with yield_units > 0 and (if current_day provided) mature."""
        result = []
        for p in self.all_plants():
            if not p.is_harvestable:
                continue
            if current_day is not None:
                first_yield = CROP_DATA.get(p.crop, {}).get("time_to_first_yield", 0)
                if current_day - p.planted_day < first_yield:
                    continue
            result.append(p)
        return result

    def harvestable_animals(self) -> List[Animal]:
        """Animals with yield_units > 0."""
        return [a for a in self.all_animals() if a.is_harvestable]

    def animals_with_fertilizer(self) -> List[Animal]:
        """Animals that have fertilizer available to collect."""
        return [a for a in self.all_animals() if a.can_collect_fertilizer]

    def empty_structures(self) -> List[EmptyStructure]:
        """Coops/pastures without animals."""
        result = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                t = self.tiles[y][x]
                if isinstance(t, EmptyStructure):
                    result.append(t)
        return result

    def num_unlocked_tiles(self) -> int:
        """Count of unlocked (non-LOCKED) tiles."""
        count = 0
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.tiles[y][x] != "LOCKED":
                    count += 1
        return count


# =============================================================================
# Market & Town
# =============================================================================

@dataclass
class Market:
    """Current market state."""
    inventory: Dict[str, int]   # product -> inventory level
    prices: Dict[str, int]      # product -> current sell price

    def price_of(self, product: str) -> int:
        return self.prices.get(product, 0)

    def inventory_of(self, product: str) -> int:
        return self.inventory.get(product, 0)


@dataclass
class Town:
    """Town state — unlocked shops."""
    unlocked_shops: List[str]   # e.g. ["BAKERY", "BAKERY", "PET_CAFE"]


# =============================================================================
# Private State
# =============================================================================

@dataclass
class PrivateState:
    """Player's private state (not visible to opponent)."""
    shed: Dict[str, int]           # product -> count in shed
    seeds: Dict[str, int]          # crop -> seed count
    inventories: List[Dict[str, int]]  # [farmer_inv, hand1_inv, ...]

    def shed_count(self, product: str) -> int:
        return self.shed.get(product, 0)

    def seed_count(self, crop: str) -> int:
        return self.seeds.get(crop, 0)

    def total_seeds(self) -> int:
        return sum(self.seeds.values())

    def total_shed_items(self) -> int:
        return sum(self.shed.values())

    def farmer_inventory(self) -> Dict[str, int]:
        if self.inventories:
            return self.inventories[0]
        return {}

    def hand_inventory(self, index: int) -> Dict[str, int]:
        idx = index + 1  # 0 is farmer
        if idx < len(self.inventories):
            return self.inventories[idx]
        return {}


# =============================================================================
# Top-Level Game State
# =============================================================================

@dataclass
class GameState:
    """Complete parsed game state for a single turn."""
    player: int                # 0 or 1
    day: int                   # 0-indexed
    hour: int                  # 0-indexed turn within day
    step: int                  # Absolute step number (0-719)
    my_farm: Farm
    opponent_farm: Farm
    market: Market
    town: Town
    private: PrivateState

    @property
    def turn(self) -> int:
        """Absolute turn number (0-719)."""
        return self.day * TURNS_PER_DAY + self.hour

    @property
    def turns_remaining(self) -> int:
        """Turns left in the game."""
        return 720 - self.turn

    @property
    def days_remaining(self) -> int:
        """Full days remaining (not counting current partial day)."""
        return 29 - self.day

    @property
    def is_first_turn_of_day(self) -> bool:
        return self.hour == 0

    @property
    def is_last_turn_of_day(self) -> bool:
        return self.hour == TURNS_PER_DAY - 1

    @property
    def my_money(self) -> float:
        return self.my_farm.money

    @property
    def opponent_money(self) -> float:
        return self.opponent_farm.money

    def plant_age(self, plant: Plant) -> int:
        """Age in days of a plant."""
        return self.day - plant.planted_day

    def animal_age(self, animal: Animal) -> int:
        """Age in days of an animal."""
        return self.day - animal.placed_day

    def get_town_demand_products(self) -> set:
        """Return set of all products currently demanded by unlocked town shops."""
        from .constants import SHOP_DEMAND
        demanded = set()
        for shop in self.town.unlocked_shops:
            for prod in SHOP_DEMAND.get(shop, []):
                demanded.add(prod)
        return demanded

    def get_product_town_consumption_rate(self, product: str) -> int:
        """Units of product consumed per 24-hour day across all unlocked town shops."""
        from .constants import SHOP_DEMAND, SINGLE_PRODUCT_SHOPS, TOWN_SHOP_SELL_INTERVAL
        rate = 0
        for shop in self.town.unlocked_shops:
            prods = SHOP_DEMAND.get(shop, [])
            if product in prods:
                mult = 2 if shop in SINGLE_PRODUCT_SHOPS else 1
                rate += mult * (24 // TOWN_SHOP_SELL_INTERVAL)
        return rate

    @staticmethod
    def from_obs(obs: dict) -> 'GameState':
        """Parse raw observation dict into a typed GameState."""
        player_id = obs["player"]
        day = obs["day"]
        hour = obs["hour"]
        # Step may be provided directly, or compute from day/hour
        step = int(obs.get("step", day * TURNS_PER_DAY + hour) or 0)

        # Parse both farms
        farms_raw = obs["farms"]
        my_farm = _parse_farm(farms_raw[player_id])
        opp_id = 1 - player_id
        opponent_farm = _parse_farm(farms_raw[opp_id])

        # Parse market
        market_raw = obs["market"]
        market = Market(
            inventory=dict(market_raw.get("inventory", {})),
            prices=dict(market_raw.get("prices", {})),
        )

        # Parse town
        town_raw = obs.get("town", {})
        town = Town(
            unlocked_shops=list(town_raw.get("unlocked_shops", [])),
        )

        # Parse private state
        private_raw = obs.get("private", {})
        inventories = []
        for inv in private_raw.get("inventories", []):
            if inv is None:
                inventories.append({})
            elif isinstance(inv, dict):
                inventories.append(dict(inv))
            else:
                inventories.append({})

        private = PrivateState(
            shed=dict(private_raw.get("shed", {})),
            seeds=dict(private_raw.get("seeds", {})),
            inventories=inventories,
        )

        return GameState(
            player=player_id,
            day=day,
            hour=hour,
            step=step,
            my_farm=my_farm,
            opponent_farm=opponent_farm,
            market=market,
            town=town,
            private=private,
        )


# =============================================================================
# Parsing Helpers
# =============================================================================

def _parse_farm(farm_raw: dict) -> Farm:
    """Parse a raw farm dict into a Farm object."""
    tiles = []
    raw_tiles = farm_raw.get("tiles", [])
    for y, row in enumerate(raw_tiles):
        parsed_row = []
        for x, cell in enumerate(row):
            parsed_row.append(_parse_tile(cell, x, y))
        tiles.append(parsed_row)

    farmer_raw = farm_raw.get("farmer", [0, 0])
    hands_raw = farm_raw.get("hands", [])

    return Farm(
        money=farm_raw.get("money", 0),
        tiles=tiles,
        farmer_pos=(farmer_raw[0], farmer_raw[1]),
        hands_positions=[(h[0], h[1]) for h in hands_raw],
        unlocked_quadrants=list(farm_raw.get("unlocked_quadrants", ["NW"])),
        hires_today=farm_raw.get("hires_today", 0),
    )


def _parse_tile(cell: Any, x: int, y: int) -> Any:
    """Parse a single tile from the observation."""
    if cell is None:
        return None
    if cell == "LOCKED":
        return "LOCKED"
    if not isinstance(cell, dict):
        return None

    kind = cell.get("kind", "")

    if kind == "PLANT":
        return Plant(
            crop=cell.get("crop", "WHEAT"),
            planted_day=cell.get("planted_day", 0),
            watered_today=cell.get("watered_today", False),
            consecutive_unwatered=cell.get("consecutive_unwatered", 0),
            yield_units=cell.get("yield_units", 0),
            max_lifespan_step=cell.get("max_lifespan_step", -1),
            fertilized_until_day=cell.get("fertilized_until_day", -1),
            x=x,
            y=y,
        )

    if kind == "WEED":
        return Weed(x=x, y=y)

    if kind in ("COOP", "PASTURE"):
        animal_type = cell.get("animal")
        if animal_type is not None:
            return Animal(
                animal_type=animal_type,
                structure=kind,
                placed_day=cell.get("placed_day", 0),
                yield_units=cell.get("yield_units", 0),
                fed_today=cell.get("fed_today", False),
                consecutive_unfed=cell.get("consecutive_unfed", 0),
                cared_today=cell.get("cared_today", False),
                fertilizer_available=cell.get("fertilizer_available", False),
                pending_care_bonus=cell.get("pending_care_bonus", 0),
                x=x,
                y=y,
            )
        else:
            return EmptyStructure(structure=kind, x=x, y=y)

    return None
