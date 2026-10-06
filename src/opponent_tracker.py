"""
Opponent Observation Tracker & Clone-Horizon Preemption System with Ledger Debt Tracking.

Tracks turn-by-turn shared market inventory deltas attributable to the opponent,
detects recurring liquidation rhythms (hourly and multi-day dump cycles),
and enables 1-turn early preemption backed by a strict ledger debt mechanism
to prevent double-selling inventory.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

from .constants import (
    SHOP_DEMAND,
    SINGLE_PRODUCT_SHOPS,
    TOWN_SHOP_SELL_INTERVAL,
    TOWN_CENTER_SELL_INTERVAL,
    ALL_PRODUCTS,
    CROP_DATA,
    ANIMAL_DATA,
)

ANIMAL_PRODUCT = {
    "COW": "MILK",
    "SHEEP": "WOOL",
    "GOOSE": "EGG",
}


def _extract_tile_info(cell: Any) -> Tuple[Optional[str], Optional[str], int, int]:
    """
    Extract (kind, item_name, yield_units, start_day) from a tile cell.
    Works with typed dataclasses (Plant, Animal) or raw observation dicts.
    Returns:
        (kind, item_name, yield_units, start_day)
        e.g. ("PLANT", "MELON", 0, 0) or ("ANIMAL", "COW", 1, 0)
    """
    if cell is None or cell == "LOCKED":
        return None, None, 0, 0

    # Dataclass Plant
    if hasattr(cell, "crop"):
        return "PLANT", getattr(cell, "crop", ""), getattr(cell, "yield_units", 0), getattr(cell, "planted_day", 0)

    # Dataclass Animal
    if hasattr(cell, "animal_type"):
        return "ANIMAL", getattr(cell, "animal_type", ""), getattr(cell, "yield_units", 0), getattr(cell, "placed_day", 0)

    # Raw dict
    if isinstance(cell, dict):
        kind = cell.get("kind", "")
        if kind == "PLANT":
            return "PLANT", cell.get("crop", ""), cell.get("yield_units", 0), cell.get("planted_day", 0)
        animal = cell.get("animal")
        if animal:
            return "ANIMAL", animal, cell.get("yield_units", 0), cell.get("placed_day", 0)

    return None, None, 0, 0


class OpponentTracker:
    """
    Online tracker for opponent market interactions and liquidation rhythms.
    Persists across turns of an episode; resets cleanly at episode boundaries.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all state for a new episode."""
        self.last_step = -1
        self.last_market_inventory: Dict[str, int] = {}
        self.last_our_orders: List[List[Any]] = []
        self.last_town_shops: List[str] = []

        # Granular history: history[day][hour][product] = {'sold': int, 'bought': int}
        self.history = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"sold": 0, "bought": 0})))

        # Aggregate volumes
        self.daily_sales = defaultdict(lambda: defaultdict(int))     # day -> product -> total_sold
        self.hourly_sales = defaultdict(lambda: defaultdict(int))    # hour -> product -> total_sold
        self.day_hour_sales = defaultdict(lambda: defaultdict(int))  # (day, hour) -> product -> sold
        self.total_sales = defaultdict(int)                          # product -> total_sold
        self.total_buys = defaultdict(int)                           # product -> total_bought

        # Ledger debt tracking: product -> list of active debts: [{'qty': int, 'orig_day': int, 'orig_hour': int, 'orig_step': int}]
        self.active_debts: List[Dict[str, Any]] = []

        # Opponent farm tile inspection & pipeline tracking
        self.last_opp_tiles: Dict[Tuple[int, int], Tuple[Optional[str], Optional[str], int, int]] = {}
        self.current_opp_tiles: Optional[List[List[Any]]] = None
        self.harvested_by_opp: Dict[str, int] = defaultdict(int)
        self.harvest_events: List[Tuple[int, str, int]] = []

        # Diagnostics / statistics across turns
        self.preemptions_count = defaultdict(int)  # product -> count of preemptions fired
        self.preemptions_volume = defaultdict(int) # product -> total units preemptively sold

    def update(self, obs: Any, our_submitted_orders: Optional[List[List[Any]]] = None, state: Optional[Any] = None):
        """
        Update tracker with the latest turn observation.
        Reconstructs the opponent's net transactions from the previous turn's market delta.
        """
        # Extract step, market inventory, town shops
        if isinstance(obs, dict) and "step" in obs:
            step = int(obs["step"])
        elif state is not None and hasattr(state, "step"):
            step = int(state.step)
        else:
            step = int(getattr(obs, "step", 0) or 0)
        
        # Episode boundary check: step 0 or backward step transition
        if step == 0 or step < self.last_step:
            self.reset()

        if state is not None and hasattr(state, "market") and hasattr(state.market, "inventory"):
            cur_inv = dict(state.market.inventory)
        else:
            market = getattr(obs, "market", {}) if not isinstance(obs, dict) else obs.get("market", {})
            if hasattr(market, "inventory"):
                cur_inv = dict(market.inventory)
            elif isinstance(market, dict) and "inventory" in market:
                cur_inv = dict(market["inventory"])
            else:
                cur_inv = {}

        if state is not None and hasattr(state, "town") and hasattr(state.town, "unlocked_shops"):
            cur_unlocked = list(state.town.unlocked_shops)
        else:
            town = getattr(obs, "town", {}) if not isinstance(obs, dict) else obs.get("town", {})
            if hasattr(town, "unlocked_shops"):
                cur_unlocked = list(town.unlocked_shops)
            elif isinstance(town, dict) and "unlocked_shops" in town:
                cur_unlocked = list(town["unlocked_shops"])
            else:
                cur_unlocked = []

        # If we have a prior step's baseline, calculate the delta
        if self.last_step >= 0 and self.last_market_inventory and cur_inv:
            prev_step = self.last_step
            prev_day = prev_step // 24
            prev_hour = prev_step % 24

            # 1. Compute our own net market contribution from orders submitted last turn (at prev_step)
            our_net = defaultdict(int)
            if our_submitted_orders:
                for o in our_submitted_orders:
                    if isinstance(o, (list, tuple)) and len(o) >= 3:
                        op, prod, qty = o[0], o[1], o[2]
                        if op == "SELL":
                            our_net[prod] += int(qty)
                        elif op == "BUY_PRODUCT":
                            our_net[prod] -= int(qty)

            # 2. Compute town consumption during prev_step
            # Shops consume on step % 4 == 0
            # Town center consumes on step % 24 == 0 (non-fertilizer)
            c_town = defaultdict(int)
            if prev_step % TOWN_SHOP_SELL_INTERVAL == 0 and self.last_town_shops:
                for shop in self.last_town_shops:
                    prods = SHOP_DEMAND.get(shop, [])
                    mult = 2 if shop in SINGLE_PRODUCT_SHOPS else 1
                    for p in prods:
                        c_town[p] += mult
            if prev_step % TOWN_CENTER_SELL_INTERVAL == 0:
                for p in ALL_PRODUCTS:
                    if p != "FERTILIZER":
                        c_town[p] += 1

            # 3. Derive opponent net: Delta_opp = (cur_inv - prev_inv) - our_net + c_town
            for prod in ALL_PRODUCTS:
                prev_p = self.last_market_inventory.get(prod, 10000)
                cur_p = cur_inv.get(prod, prev_p)
                delta_inv = cur_p - prev_p
                opp_net = delta_inv - our_net[prod] + c_town[prod]

                if opp_net > 0:
                    sold_qty = opp_net
                    self.history[prev_day][prev_hour][prod]["sold"] += sold_qty
                    self.daily_sales[prev_day][prod] += sold_qty
                    self.hourly_sales[prev_hour][prod] += sold_qty
                    self.day_hour_sales[(prev_day, prev_hour)][prod] += sold_qty
                    self.total_sales[prod] += sold_qty
                elif opp_net < 0:
                    bought_qty = -opp_net
                    self.history[prev_day][prev_hour][prod]["bought"] += bought_qty
                    self.total_buys[prod] += bought_qty

        # Age out expired ledger debts
        self.clear_expired_debt(step)

        # Inspect opponent farm tiles for harvest events and pipeline tracking
        self._inspect_opponent_tiles(obs, state, step)

        # Cache current turn state for next turn's delta calculation
        self.last_step = step
        self.last_market_inventory = cur_inv
        self.last_town_shops = cur_unlocked

    def _inspect_opponent_tiles(self, obs: Any, state: Optional[Any], step: int):
        """
        Inspect opponent farm tiles, detect harvest events, and update pipeline state.
        """
        opp_tiles = None
        if state is not None and hasattr(state, "opponent_farm") and state.opponent_farm is not None:
            opp_tiles = getattr(state.opponent_farm, "tiles", None)
        elif hasattr(obs, "opponent_farm") and obs.opponent_farm is not None:
            opp_tiles = getattr(obs.opponent_farm, "tiles", None)
        elif isinstance(obs, dict) and "farms" in obs:
            player_id = obs.get("player", 0)
            opp_id = 1 - player_id
            farms = obs.get("farms", [])
            if len(farms) > opp_id and isinstance(farms[opp_id], dict):
                opp_tiles = farms[opp_id].get("tiles", None)

        if not opp_tiles:
            return

        self.current_opp_tiles = opp_tiles
        new_tile_map = {}

        for y, row in enumerate(opp_tiles):
            for x, cell in enumerate(row):
                kind, item, yield_units, start_day = _extract_tile_info(cell)
                new_tile_map[(x, y)] = (kind, item, yield_units, start_day)

                if self.last_opp_tiles and (x, y) in self.last_opp_tiles:
                    prev_kind, prev_item, prev_yield, prev_day = self.last_opp_tiles[(x, y)]

                    # Check for plant harvest (exclude weeds)
                    if prev_kind == "PLANT" and prev_yield > 0:
                        if kind != "WEED":
                            if kind != "PLANT" or item != prev_item or yield_units < prev_yield:
                                harvested = prev_yield if (kind != "PLANT" or item != prev_item) else (prev_yield - yield_units)
                                if harvested > 0:
                                    self.harvested_by_opp[prev_item] += harvested
                                    self.harvest_events.append((step, prev_item, harvested))

                    # Check for animal harvest
                    elif prev_kind == "ANIMAL" and prev_yield > 0:
                        if yield_units < prev_yield:
                            harvested = prev_yield - yield_units
                            prod = ANIMAL_PRODUCT.get(prev_item)
                            if prod and harvested > 0:
                                self.harvested_by_opp[prod] += harvested
                                self.harvest_events.append((step, prod, harvested))

        self.last_opp_tiles = new_tile_map

    def _get_current_opp_tiles(self, state: Optional[Any] = None) -> Optional[List[List[Any]]]:
        """Return the current opponent farm tiles."""
        if state is not None and hasattr(state, "opponent_farm") and state.opponent_farm is not None:
            return getattr(state.opponent_farm, "tiles", None)
        return self.current_opp_tiles

    def get_dump_ratio(self, product: str) -> Tuple[float, str]:
        """
        Calibrate opponent's dump-latency for `product`.
        Returns:
            (dump_ratio, status)
            where status is in ("UNESTABLISHED", "QUICK_DUMPER", "HOARDER")
        """
        harvested = self.harvested_by_opp.get(product, 0)
        MIN_CALIBRATION_HARVESTS = 4

        if harvested < MIN_CALIBRATION_HARVESTS:
            return 0.0, "UNESTABLISHED"

        sold = self.total_sales.get(product, 0)
        ratio = min(1.0, sold / max(1, harvested))

        if ratio >= 0.60:
            return ratio, "QUICK_DUMPER"
        else:
            return ratio, "HOARDER"

    def project_pipeline_dump(
        self,
        product: str,
        current_day: int,
        current_hour: int,
        current_step: int,
        state: Optional[Any] = None,
    ) -> Tuple[bool, int]:
        """
        Project expected opponent market-dump volume and timing for `product`,
        1-2 turns ahead of when their crops/animals will actually yield.

        Guarded by calibration: only triggers if opponent is a confirmed QUICK_DUMPER.
        """
        ratio, status = self.get_dump_ratio(product)
        if status != "QUICK_DUMPER":
            return False, 0

        opp_tiles = self._get_current_opp_tiles(state)
        if not opp_tiles:
            return False, 0

        projected_volume = 0
        next_day = current_day + 1
        is_imminent_day_refresh = (current_hour in (22, 23))

        for row in opp_tiles:
            for cell in row:
                kind, item, yield_units, start_day = _extract_tile_info(cell)
                if not kind or not item:
                    continue

                # Case A: Imminent day refresh (Hour 22, 23)
                if is_imminent_day_refresh:
                    if kind == "ANIMAL":
                        prod = ANIMAL_PRODUCT.get(item)
                        if prod == product:
                            adata = ANIMAL_DATA.get(item, {})
                            first_yield = adata.get("time_to_first_yield", 0)
                            interval = adata.get("yield_interval", 1)
                            days_since = next_day - start_day - first_yield
                            if days_since >= 0 and days_since % interval == 0:
                                projected_volume += 1

                    elif kind == "PLANT" and item == product:
                        cdata = CROP_DATA.get(item, {})
                        is_ongoing = (cdata.get("yield_type") == "ongoing")
                        first_yield = cdata.get("time_to_first_yield", 0)

                        if is_ongoing:
                            interval = cdata.get("yield_interval", 1)
                            days_since = next_day - start_day - first_yield
                            max_scheduled = cdata.get("max_yield", 4)
                            if days_since >= 0 and days_since % interval == 0:
                                production_count = (days_since // interval) + 1
                                if production_count <= max_scheduled:
                                    projected_volume += 1
                        else:
                            max_yield_day = cdata.get("time_to_max_yield", first_yield)
                            maturity_day = start_day + max_yield_day
                            if next_day == maturity_day:
                                expected_units = max(1, yield_units) if yield_units > 0 else cdata.get("max_yield_unfertilized", 4)
                                projected_volume += expected_units

                # Case B: Mature produce currently sitting unharvested on opponent tiles
                if kind == "ANIMAL":
                    prod = ANIMAL_PRODUCT.get(item)
                    if prod == product and yield_units > 0:
                        projected_volume += yield_units
                elif kind == "PLANT" and item == product:
                    cdata = CROP_DATA.get(item, {})
                    first_yield = cdata.get("time_to_first_yield", 0)
                    if (current_day - start_day >= first_yield) and yield_units > 0:
                        projected_volume += yield_units

        if projected_volume > 0:
            return True, projected_volume

        return False, 0

    def evaluate_preemption(
        self,
        product: str,
        current_day: int,
        current_hour: int,
        current_step: int,
        state: Optional[Any] = None,
    ) -> Tuple[bool, int, str]:
        """
        Unified preemption evaluator.
        1. Calibrated pipeline projection (earlier-firing enhancement).
        2. Reactive liquidation rhythm (fallback floor).

        Returns:
            (will_preempt, estimated_volume, reason)
            where reason in ("pipeline", "reactive", "none")
        """
        # 1. Calibrated pipeline projection
        pipe_dump, pipe_vol = self.project_pipeline_dump(
            product, current_day, current_hour, current_step, state
        )
        if pipe_dump and pipe_vol > 0:
            return True, pipe_vol, "pipeline"

        # 2. Fallback to existing reactive rhythm detection
        react_dump, react_vol = self.detect_liquidation_rhythm(
            product, current_day, current_hour
        )
        if react_dump and react_vol > 0:
            return True, react_vol, "reactive"

        return False, 0, "none"

    # =========================================================================
    # Pattern Detection
    # =========================================================================

    def detect_liquidation_rhythm(self, product: str, current_day: int, current_hour: int) -> Tuple[bool, int]:
        """
        Detect if the opponent exhibits a predictable liquidation rhythm for `product`
        at the upcoming hour (current_hour + 1) or upcoming day.

        Returns:
            (will_dump, estimated_volume)
        """
        target_hour = (current_hour + 1) % 24
        target_day = current_day if target_hour != 0 else current_day + 1

        # Rhythm 1: Concentrated Daily Liquidation Hour (e.g. Hour 1 Dump Rhythm)
        # Check if the opponent historically sells this product heavily at target_hour
        hour_volume = self.hourly_sales[target_hour][product]
        prod_total = self.total_sales[product]

        # Condition 1A: Established recurring dump hour
        # If at least 2 distinct days have seen sales of this product at target_hour,
        # and target_hour accounts for >= 25% of all opponent sales for this product
        distinct_days_at_target_hour = sum(
            1 for d in range(current_day)
            if self.history[d][target_hour][product]["sold"] > 0
        )

        if distinct_days_at_target_hour >= 2 and prod_total >= 10:
            if (hour_volume / prod_total) >= 0.25:
                est_vol = max(1, hour_volume // max(1, distinct_days_at_target_hour))
                return True, est_vol

        # Rhythm 2: Fixed Meta Dump Rhythm at Hour 1 (Common across SEP, BTT, 3000, Mega)
        # Even on earlier days (day >= 4), if opponent has shown any dump at Hour 1 (>= 5 units)
        # and upcoming hour is Hour 1, flag preemption for premium commodities (MILK, WOOL, MELON, STRAWBERRY)
        if target_hour == 1 and current_day >= 4:
            h1_vol = self.hourly_sales[1][product]
            if h1_vol >= 5 or (product in ("MILK", "WOOL") and self.total_sales[product] >= 8):
                est_vol = max(4, h1_vol // max(1, current_day))
                return True, est_vol

        # Rhythm 3: Common Multi-Day Large Dump Days (Days 10, 15, 20, 24, 25, 29)
        # If tomorrow is a major liquidation day and we are at Hour 23, or if opponent
        # dumped >= 20 units on a previous cycle
        COMMON_DUMP_DAYS = {10, 15, 20, 24, 25, 29}
        if target_day in COMMON_DUMP_DAYS and target_hour in (0, 1):
            if product in ("MELON", "STRAWBERRY", "WOOL", "MILK"):
                # If opponent has any active sales or has demonstrated dump capacity
                if prod_total >= 5 or self.daily_sales[target_day - 5][product] >= 10:
                    est_vol = max(8, prod_total // max(1, current_day))
                    return True, est_vol

        return False, 0

    # =========================================================================
    # Ledger Debt Tracking
    # =========================================================================

    def record_preemptive_sale(self, product: str, quantity: int, current_step: int, orig_step: int):
        """
        Record internal ledger debt for a sale that was pulled forward by 1 turn.
        The debt offsets what would otherwise be sold on orig_step.
        """
        if quantity <= 0:
            return
        debt_entry = {
            "product": product,
            "quantity": int(quantity),
            "created_step": current_step,
            "orig_step": orig_step,
        }
        self.active_debts.append(debt_entry)
        self.preemptions_count[product] += 1
        self.preemptions_volume[product] += int(quantity)

    def get_ledger_debt(self, product: str) -> int:
        """Return the sum of active ledger debt for `product`."""
        return sum(d["quantity"] for d in self.active_debts if d["product"] == product)

    def clear_expired_debt(self, current_step: int):
        """
        Retire ledger debt entries once the originally scheduled turn has passed.
        """
        self.active_debts = [d for d in self.active_debts if current_step <= d["orig_step"]]


# Global singleton instance
_tracker_instance = OpponentTracker()


def get_opponent_tracker() -> OpponentTracker:
    """Return the global OpponentTracker singleton."""
    global _tracker_instance
    return _tracker_instance


def reset_opponent_tracker():
    """Explicitly reset the global OpponentTracker singleton."""
    global _tracker_instance
    _tracker_instance.reset()
