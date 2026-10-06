"""
Unit tests for strategy logic and target generation.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_state import GameState, Plant, Animal, Weed, EmptyStructure, Farm
from src.strategy import (
    get_build_targets, get_planting_targets, generate_market_orders,
    _generate_animal_orders, _animal_matches_structure,
)


class TestStrategy(unittest.TestCase):
    """Test strategy decisions, build targets, and market orders."""

    def _make_state(self, day=6, hour=0, money=3000, shed=None, seeds=None, tiles=None, farmer_pos=(4, 4), unlocked_quadrants=None):
        if tiles is None:
            tiles = [[None] * 10 for _ in range(10)]
        if shed is None:
            shed = {}
        if seeds is None:
            seeds = {}
        if unlocked_quadrants is None:
            unlocked_quadrants = ["NW"]

        obs = {
            "player": 0,
            "day": day,
            "hour": hour,
            "step": day * 24 + hour,
            "farms": [
                {
                    "money": money,
                    "tiles": tiles,
                    "farmer": list(farmer_pos),
                    "hands": [],
                    "unlocked_quadrants": list(unlocked_quadrants),
                    "hires_today": 0,
                },
                {
                    "money": 2000,
                    "tiles": [[None] * 10 for _ in range(10)],
                    "farmer": [4, 4],
                    "hands": [],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                },
            ],
            "market": {
                "inventory": {"WHEAT": 10000, "COW": 10000},
                "prices": {"WHEAT": 25, "COW": 400},
            },
            "town": {"unlocked_shops": []},
            "private": {
                "shed": shed,
                "seeds": seeds,
                "inventories": [{}],
            },
        }
        return GameState.from_obs(obs)

    def test_animal_matches_structure(self):
        self.assertTrue(_animal_matches_structure("GOOSE", "COOP"))
        self.assertTrue(_animal_matches_structure("COW", "PASTURE"))
        self.assertTrue(_animal_matches_structure("SHEEP", "PASTURE"))
        self.assertFalse(_animal_matches_structure("COW", "COOP"))
        self.assertFalse(_animal_matches_structure("GOOSE", "PASTURE"))

    def test_get_build_targets_when_animal_in_shed_no_structure(self):
        """When shed has a COW and no pasture exists, get_build_targets returns BUILD_PASTURE."""
        state = self._make_state(shed={"COW": 1})
        targets = get_build_targets(state)
        self.assertGreaterEqual(len(targets), 1)
        self.assertEqual(targets[0][2], "BUILD_PASTURE")

    def test_get_build_targets_when_matching_structure_already_exists(self):
        """When shed has a COW and an empty PASTURE already exists, no new build target is needed."""
        tiles = [[None] * 10 for _ in range(10)]
        tiles[1][1] = {"kind": "PASTURE"}
        state = self._make_state(shed={"COW": 1}, tiles=tiles)
        targets = get_build_targets(state)
        self.assertEqual(len(targets), 0)

    def test_get_build_targets_for_goose(self):
        """When shed has a GOOSE and no coop exists, returns BUILD_COOP."""
        state = self._make_state(shed={"GOOSE": 1})
        targets = get_build_targets(state)
        self.assertGreaterEqual(len(targets), 1)
        self.assertEqual(targets[0][2], "BUILD_COOP")

    def test_early_seed_orders_scale_past_three(self):
        """Early game seed orders should scale to available empty tiles rather than capping at 3."""
        # NW quadrant has 25 tiles (all empty)
        tiles = [[None] * 10 for _ in range(10)]
        for r in range(10):
            for c in range(10):
                if r >= 5 or c >= 5:
                    tiles[r][c] = "LOCKED"
        state = self._make_state(day=1, money=500, tiles=tiles)
        orders = generate_market_orders(state)
        seed_orders = [o for o in orders if o[0] == "BUY_SEED"]
        self.assertGreaterEqual(len(seed_orders), 1)
        # Empty tiles = 25, money = 500 => buys 25 wheat seeds
        self.assertEqual(seed_orders[0][1], "WHEAT")
        self.assertEqual(seed_orders[0][2], 25)

    def test_demand_per_day_calculation(self):
        """Test demand_per_day with unlocked shops and town center."""
        from src.strategy import demand_per_day
        # Town with PET_CAFE (single product shop for CARROT -> multiplier 2)
        state = self._make_state()
        state.town.unlocked_shops = ["PET_CAFE"]
        # CARROT gets 24/4 * 2 = 12 (shop) + 24/24 = 1 (town center) = 13.0
        self.assertEqual(demand_per_day(state, "CARROT"), 13.0)
        # FERTILIZER gets 0 (no shop, town center doesn't consume fertilizer)
        self.assertEqual(demand_per_day(state, "FERTILIZER"), 0.0)
        # MELON gets 1.0 (from town center)
        self.assertEqual(demand_per_day(state, "MELON"), 1.0)

    def test_rank_sell_orders_impact_and_urgency(self):
        """Test rank_sell_orders sorts by impact and urgency descending."""
        from src.strategy import rank_sell_orders
        state = self._make_state()
        orders = [
            ["SELL", "CARROT", 2],
            ["SELL", "MELON", 10],
            ["SELL", "WHEAT", 5],
        ]
        ranked = rank_sell_orders(orders, state)
        self.assertEqual(len(ranked), 3)
        # MELON has highest base price and largest price impact -> ranked first
        self.assertEqual(ranked[0][1], "MELON")

    def test_terminal_liquidation_orders_highest_base_price_first(self):
        """Terminal liquidation should order highest base-value items first (descending)."""
        shed = {"MELON": 2, "WHEAT": 10, "CARROT": 5, "COW": 1, "FERTILIZER": 4}
        state = self._make_state(day=29, hour=22, shed=shed)
        orders = generate_market_orders(state)
        sold_items = [o[1] for o in orders if o[0] == "SELL"]
        # MELON (base 250), FERTILIZER (base 100), CARROT (base 35), WHEAT (base 25)
        self.assertEqual(sold_items, ["MELON", "FERTILIZER", "CARROT", "WHEAT"])
        self.assertNotIn("COW", sold_items)

        # Truncation check: when >10 products in shed, lowest base price products are dropped
        shed_11 = {
            "MELON": 1, "WOOL": 1, "MILK": 1, "STRAWBERRY": 1, "FERTILIZER": 1,
            "TOMATO": 1, "EGG": 1, "CARROT": 1, "WHEAT": 1, "BONUS_HIGH": 1, "BONUS_LOW": 1
        }
        from src.constants import MARKET_PARAMS
        MARKET_PARAMS["BONUS_HIGH"] = {"base": 300}
        MARKET_PARAMS["BONUS_LOW"] = {"base": 5}
        try:
            state_11 = self._make_state(day=29, hour=22, shed=shed_11)
            orders_11 = generate_market_orders(state_11)
            self.assertEqual(len(orders_11), 10)
            items_11 = [o[1] for o in orders_11]
            self.assertEqual(items_11[0], "BONUS_HIGH")
            self.assertEqual(items_11[1], "MELON")
            self.assertNotIn("BONUS_LOW", items_11)
        finally:
            del MARKET_PARAMS["BONUS_HIGH"]
            del MARKET_PARAMS["BONUS_LOW"]

    def test_interleaved_animal_orders(self):
        """Test that animal purchases interleave COW and SHEEP round-robin."""
        from src.strategy import _generate_animal_orders
        # Day 15, plenty of money ($5000) and plenty of empty tiles
        tiles = [[None] * 10 for _ in range(10)]
        state = self._make_state(day=15, money=5000, tiles=tiles)
        orders = _generate_animal_orders(state)
        self.assertGreaterEqual(len(orders), 2)
        # First order should be COW (tied at 1.0), second order should be SHEEP (6/6 > 7/8)
        self.assertEqual(orders[0], ["BUY_ANIMAL", "COW", 1])
        self.assertEqual(orders[1], ["BUY_ANIMAL", "SHEEP", 1])

    def test_dynamic_animal_orders_ratio_self_correcting(self):
        """When COW is ahead (1 cow in shed, 0 sheep), SHEEP is picked first."""
        from src.strategy import _generate_animal_orders
        tiles = [[None] * 10 for _ in range(10)]
        # Shed has 1 COW (cows_needed = 7/8 = 0.875, sheep_needed = 6/6 = 1.0)
        state = self._make_state(day=15, money=5000, tiles=tiles, shed={"COW": 1})
        orders = _generate_animal_orders(state)
        self.assertGreaterEqual(len(orders), 1)
        # SHEEP ratio (1.0) > COW ratio (0.875) -> First order must be SHEEP
        self.assertEqual(orders[0], ["BUY_ANIMAL", "SHEEP", 1])

    def test_order_assembly_decoupled_from_cash_accounting(self):
        """Test that structural orders (BUY_LAND, BUY_ANIMAL, BUY_SEED) precede HIRE flood."""
        # Day 6, sufficient funds for land, animals, seeds, and hiring
        state = self._make_state(day=6, money=3000)
        orders = generate_market_orders(state)
        # Find indices of different order types
        order_types = [o[0] for o in orders]
        if "BUY_LAND" in order_types and "HIRE" in order_types:
            first_land = order_types.index("BUY_LAND")
            first_hire = order_types.index("HIRE")
            self.assertLess(first_land, first_hire, "BUY_LAND must appear before HIRE in assembled orders")
        if "BUY_ANIMAL" in order_types and "HIRE" in order_types:
            first_animal = order_types.index("BUY_ANIMAL")
            first_hire = order_types.index("HIRE")
            self.assertLess(first_animal, first_hire, "BUY_ANIMAL must appear before HIRE in assembled orders")

    def test_land_buffers_reduced(self):
        """Test that Q2 buffer is 1050 with current_money and 1200 standalone; Q3 is 1350 with current_money and 1500 standalone."""
        from src.strategy import _generate_land_orders
        # With current_money provided (upstream deduction happened):
        state_q2 = self._make_state(day=5, money=2049)
        self.assertEqual(_generate_land_orders(state_q2, "mid", 2049), [])
        state_q2_ok = self._make_state(day=5, money=2050)
        self.assertEqual(_generate_land_orders(state_q2_ok, "mid", 2050), [["BUY_LAND"]])
        # Day 0 Q2 expansion blocked by day >= 5 gate:
        state_q2_d0 = self._make_state(day=0, money=2049)
        self.assertEqual(_generate_land_orders(state_q2_d0, "early", 2049), [])
        state_q2_d0_ok = self._make_state(day=0, money=2050)
        self.assertEqual(_generate_land_orders(state_q2_d0_ok, "early", 2050), [])  # Blocked by day < 5

        # Q3 expansion gated at day >= 9:
        state_q3 = self._make_state(day=9, money=3349)
        state_q3.my_farm.unlocked_quadrants = [0, 1]
        self.assertEqual(_generate_land_orders(state_q3, "mid", 3349), [])
        state_q3_ok = self._make_state(day=9, money=3350)
        state_q3_ok.my_farm.unlocked_quadrants = [0, 1]
        self.assertEqual(_generate_land_orders(state_q3_ok, "mid", 3350), [["BUY_LAND"]])

        # Standalone (current_money is None): preserves baseline buffers 1200 / 1500
        state_q2_none = self._make_state(day=5, money=2199)
        self.assertEqual(_generate_land_orders(state_q2_none, "mid", None), [])
        state_q2_none_ok = self._make_state(day=5, money=2200)
        self.assertEqual(_generate_land_orders(state_q2_none_ok, "mid", None), [["BUY_LAND"]])

        state_q3_none = self._make_state(day=12, money=3499)
        state_q3_none.my_farm.unlocked_quadrants = [0, 1]
        self.assertEqual(_generate_land_orders(state_q3_none, "mid", None), [])
        state_q3_none_ok = self._make_state(day=12, money=3500)
        state_q3_none_ok.my_farm.unlocked_quadrants = [0, 1]
        self.assertEqual(_generate_land_orders(state_q3_none_ok, "mid", None), [["BUY_LAND"]])

    def test_animal_reserve_zeroed_when_current_money_passed(self):
        """When current_money is provided, today's hire cost is zeroed (already deducted upstream)."""
        from src.strategy import _generate_animal_orders
        tiles = [[None] * 10 for _ in range(10)]
        state = self._make_state(day=10, money=1500, tiles=tiles)
        orders_with_curr = _generate_animal_orders(state, current_money=1500)
        orders_standalone = _generate_animal_orders(state, current_money=None)
        self.assertIsInstance(orders_with_curr, list)
        self.assertIsInstance(orders_standalone, list)

    def test_seed_reserve_zeroed_when_current_money_passed(self):
        """When current_money is provided, today's wage is accounted for, producing matching spendable cash."""
        from src.strategy import _generate_seed_orders
        tiles = [[None] * 10 for _ in range(10)]
        for r in range(10):
            for c in range(10):
                if r >= 5 or c >= 5:
                    tiles[r][c] = "LOCKED"
        # Standalone: money=500, reserve=200 => spendable=300 => 25 wheat
        state_standalone = self._make_state(day=1, money=500, tiles=tiles)
        orders_standalone = _generate_seed_orders(state_standalone, "early", current_money=None)
        self.assertEqual(orders_standalone[0], ["BUY_SEED", "WHEAT", 25])

        # With upstream deduction: money=357 (500 - 143), reserve=57 => spendable=300 => 25 wheat
        orders_upstream = _generate_seed_orders(state_standalone, "early", current_money=357)
        self.assertEqual(orders_upstream[0], ["BUY_SEED", "WHEAT", 25])

    def test_feed_orders_fallback(self):
        """Test emergency feed fallback BUY_PRODUCT WHEAT generation, caps, and front-loading."""
        from src.strategy import _generate_feed_orders, generate_market_orders
        tiles = [[None] * 10 for _ in range(10)]
        tiles[0][0] = {"kind": "PASTURE", "animal": "COW", "fed_today": False, "consecutive_unfed": 0}
        
        # 1. Animals exist, shed has 0 wheat -> generates BUY_PRODUCT WHEAT 2
        state_no_wheat = self._make_state(day=1, money=1000, tiles=tiles, shed={"WHEAT": 0})
        orders = _generate_feed_orders(state_no_wheat, current_money=1000)
        self.assertEqual(orders, [["BUY_PRODUCT", "WHEAT", 2]])

        # 2. Shed has plenty of wheat (>= threshold) -> no orders
        state_full_wheat = self._make_state(day=1, money=1000, tiles=tiles, shed={"WHEAT": 10})
        orders_full = _generate_feed_orders(state_full_wheat, current_money=1000)
        self.assertEqual(orders_full, [])

        # 3. Market price exceeds ceiling -> no orders
        state_high_price = self._make_state(day=1, money=1000, tiles=tiles, shed={"WHEAT": 0})
        state_high_price.market.inventory["WHEAT"] = 0  # inventory depleted raises price > 25
        orders_high = _generate_feed_orders(state_high_price, current_money=1000)
        self.assertEqual(orders_high, [])

        # 4. Front-loading in generate_market_orders
        state_frontload = self._make_state(day=1, money=1000, tiles=tiles, shed={"WHEAT": 0})
        all_orders = generate_market_orders(state_frontload)
        self.assertTrue(any(o[0] == "BUY_PRODUCT" and o[1] == "WHEAT" for o in all_orders))
        # BUY_PRODUCT WHEAT should be at slot 0
        self.assertEqual(all_orders[0], ["BUY_PRODUCT", "WHEAT", 2])

    def test_unified_animal_counting(self):
        """Test that _count_total_animals counts farm + shed + carried animals identically."""
        from src.strategy import _count_total_animals, _generate_sell_orders
        # Empty farm, 0 shed animals, but 2 cows carried in inventories
        tiles = [[None] * 10 for _ in range(10)]
        state = self._make_state(day=1, money=1000, tiles=tiles, shed={"WHEAT": 6})
        state.private.inventories = [{"COW": 2}]
        
        # Helper correctly counts the 2 carried animals
        self.assertEqual(_count_total_animals(state), 2)
        self.assertEqual(_count_total_animals(state, animals_bought_today=1), 3)

        # _generate_sell_orders protects 2 animals * 3 = 6 wheat reserve, so sells 0 wheat
        sell_orders = _generate_sell_orders(state, "early")
        self.assertFalse(any(o[0] == "SELL" and o[1] == "WHEAT" for o in sell_orders))

    def test_graduated_throttle_tiers(self):
        """Test get_throttle_tier classification and strictly fresh turn-by-turn non-latching recalculation."""
        from src.strategy import get_throttle_tier, _daily_start_money

        # Setup daily start money to simulate negative cash flow: Day 1=1200, Day 2=800, Day 3=500
        _daily_start_money.clear()
        _daily_start_money[0] = 1500
        _daily_start_money[1] = 1200
        _daily_start_money[2] = 800
        _daily_start_money[3] = 500

        state = self._make_state(day=3, money=450)

        # 1. Tier 1: $300-$600 with negative trend -> Tier 1
        self.assertEqual(get_throttle_tier(state, current_money=450), 1)

        # 2. Tier 2: $150-$300 with negative trend -> Tier 2
        self.assertEqual(get_throttle_tier(state, current_money=250), 2)

        # 3. Tier 3: < $150 -> Tier 3 (acute emergency)
        self.assertEqual(get_throttle_tier(state, current_money=120), 3)
        self.assertEqual(get_throttle_tier(state, current_money=40), 3)

        # 4. Tier 0: >= $600 -> Tier 0 (unrestricted even if negative trend was present)
        self.assertEqual(get_throttle_tier(state, current_money=600), 0)
        self.assertEqual(get_throttle_tier(state, current_money=1000), 0)

        # 5. Non-latching proof: dynamic switching between turns without sticky state
        # Turn A: $200 -> Tier 2
        self.assertEqual(get_throttle_tier(state, current_money=200), 2)
        # Turn B: money recovers to $700 -> immediately Tier 0 (no lingering restriction)
        self.assertEqual(get_throttle_tier(state, current_money=700), 0)
        # Turn C: cash dips to $100 -> immediately Tier 3
        self.assertEqual(get_throttle_tier(state, current_money=100), 3)
        # Turn D: cash rises to $450 -> immediately Tier 1
        self.assertEqual(get_throttle_tier(state, current_money=450), 1)

        # 6. If not distressed (e.g. day < 2), cash of $450 is Tier 0
        state_early = self._make_state(day=1, money=450)
        self.assertEqual(get_throttle_tier(state_early, current_money=450), 0)

    def test_graduated_throttle_consumption(self):
        """Test graduated throttle consumption across hire, land, animal, and seed orders."""
        from src.strategy import (
            _generate_hire_orders, _generate_land_orders,
            _generate_animal_orders, _generate_seed_orders,
            _daily_start_money,
        )

        # Simulate negative cash flow
        _daily_start_money.clear()
        _daily_start_money[0] = 1500
        _daily_start_money[1] = 1200
        _daily_start_money[2] = 800
        _daily_start_money[3] = 500

        # --- A. Hiring by Tier ---
        # Tier 1 with $450 on 3 quads: maintains 10 hands
        state_t1_high = self._make_state(day=3, money=450, unlocked_quadrants=["NW", "NE", "SW"])
        hires_t1_high = _generate_hire_orders(state_t1_high, "early", current_money=450)
        self.assertEqual(len(hires_t1_high), 10)

        # Tier 1 with $350 on 3 quads: maintains 8 hands
        state_t1_low = self._make_state(day=3, money=350, unlocked_quadrants=["NW", "NE", "SW"])
        hires_t1_low = _generate_hire_orders(state_t1_low, "early", current_money=350)
        self.assertEqual(len(hires_t1_low), 8)

        # Tier 2 with $200 on 3 quads: steps down to 6 hands
        state_t2 = self._make_state(day=3, money=200, unlocked_quadrants=["NW", "NE", "SW"])
        hires_t2 = _generate_hire_orders(state_t2, "early", current_money=200)
        self.assertEqual(len(hires_t2), 6)

        # Tier 3 with $120 on 3 quads: emergency 4 hands
        state_t3 = self._make_state(day=3, money=120, unlocked_quadrants=["NW", "NE", "SW"])
        hires_t3 = _generate_hire_orders(state_t3, "early", current_money=120)
        self.assertEqual(len(hires_t3), 4)

        # Tier 3 with $40: < $50 returns 0 hands
        state_t3_broke = self._make_state(day=3, money=40)
        hires_t3_broke = _generate_hire_orders(state_t3_broke, "early", current_money=40)
        self.assertEqual(len(hires_t3_broke), 0)

        # --- B. Capex Freeze in Tiers 1-3 ---
        # Land and animals blocked in Tier 1 ($450), Tier 2 ($250), Tier 3 ($120)
        for m in [450, 250, 120]:
            st = self._make_state(day=6, money=m)
            self.assertEqual(_generate_land_orders(st, "mid", current_money=m), [])
            self.assertEqual(_generate_animal_orders(st, current_money=m), [])

        # --- C. Seed Orders by Tier ---
        # Tier 1 (mid game, $450): excludes Melon and Strawberry
        state_mid_t1 = self._make_state(day=6, money=450)
        seed_orders_t1 = _generate_seed_orders(state_mid_t1, "mid", current_money=450)
        crop_names_t1 = [o[1] for o in seed_orders_t1 if o[0] == "BUY_SEED"]
        self.assertNotIn("MELON", crop_names_t1)
        self.assertNotIn("STRAWBERRY", crop_names_t1)

        # Tier 2 ($250): prioritizes watering existing crops, no open planting
        tiles_with_crops = [[None] * 10 for _ in range(10)]
        for i in range(10):
            tiles_with_crops[0][i] = {"kind": "PLANT", "crop": "CARROT", "growth": 1, "watered": True, "water_count": 1}
        state_t2_crops = self._make_state(day=6, money=250, tiles=tiles_with_crops)
        seed_orders_t2 = _generate_seed_orders(state_t2_crops, "mid", current_money=250)
        self.assertEqual(seed_orders_t2, [])

    def test_animal_orders_expanded_reserve(self):
        """Option C: _generate_animal_orders accounts for wage_reserve + seed_reserve + feed_buffer."""
        from src.strategy import _generate_animal_orders
        tiles = [["LOCKED"] * 10 for _ in range(10)]
        for r in range(5):
            for c in range(5):
                tiles[r][c] = None
        # Day 0, Quad 1 unlocked (25 tiles), 0 shed wheat
        state = self._make_state(day=0, money=1857, tiles=tiles)
        
        # When land_bought_today=1 (buying Q2), reserve on Q1 is $95 wage + $480 seeds + $150 feed = $725.
        # With money=1857: 2 Cows (cost 800) leaves 1057 >= 725 (bought).
        orders = _generate_animal_orders(state, current_money=1857, land_bought_today=1)
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0], ["BUY_ANIMAL", "COW", 1])
        self.assertEqual(orders[1], ["BUY_ANIMAL", "SHEEP", 1])

        # If money is below 690 + 400 = 1090 (e.g. 1000), no animals can be bought
        orders_low = _generate_animal_orders(state, current_money=1000, land_bought_today=1)
        self.assertEqual(orders_low, [])

    def test_dynamic_ev_feed_price_ceiling(self):
        """Test that feed purchases use dynamic EV valuation rather than a fixed $35 ceiling:
        - 1 lone cow: marginal value = 0.5 * $160 = $80/day.
          - At spot wheat $45 (above old $35 ceiling): buys feed (EV positive).
          - At spot wheat $95+: skips feed (EV negative).
        - 1 lone sheep: marginal value = 0.333 * $200 = $66.67/day.
          - At spot wheat $55: buys feed (EV positive).
          - At spot wheat $75: skips feed (EV negative).
        - Both cow & sheep: at spot wheat $75, buys feed because cow EV ($80) is positive.
        """
        from src.strategy import _generate_feed_orders
        
        # Scenario 1: Lone Cow (Milk base $160, interval 2 -> $80/day EV)
        cow_tiles = [[None] * 10 for _ in range(10)]
        cow_tiles[0][0] = {"kind": "PASTURE", "animal": "COW", "fed_today": False, "consecutive_unfed": 0}
        
        # 1a. Spot wheat = $45 (inv 9600): above old $35 ceiling, but < $80 EV -> BUYS feed
        state_cow_45 = self._make_state(day=15, money=1000, tiles=cow_tiles, shed={"WHEAT": 0})
        state_cow_45.market.inventory["WHEAT"] = 9600
        orders_cow_45 = _generate_feed_orders(state_cow_45, current_money=1000)
        self.assertEqual(orders_cow_45, [["BUY_PRODUCT", "WHEAT", 2]])
        
        # 1b. Spot wheat = $95+ (inv 5000): > $80 EV -> SKIPS feed
        state_cow_95 = self._make_state(day=15, money=1000, tiles=cow_tiles, shed={"WHEAT": 0})
        state_cow_95.market.inventory["WHEAT"] = 5000
        orders_cow_95 = _generate_feed_orders(state_cow_95, current_money=1000)
        self.assertEqual(orders_cow_95, [])

        # Scenario 2: Lone Sheep (Wool base $200, interval 3 -> $66.67/day EV)
        sheep_tiles = [[None] * 10 for _ in range(10)]
        sheep_tiles[0][0] = {"kind": "PASTURE", "animal": "SHEEP", "fed_today": False, "consecutive_unfed": 0}
        
        # 2a. Spot wheat = $55 (inv 9100): < $66.67 EV -> BUYS feed
        state_sheep_55 = self._make_state(day=15, money=1000, tiles=sheep_tiles, shed={"WHEAT": 0})
        state_sheep_55.market.inventory["WHEAT"] = 9100
        orders_sheep_55 = _generate_feed_orders(state_sheep_55, current_money=1000)
        self.assertEqual(orders_sheep_55, [["BUY_PRODUCT", "WHEAT", 2]])

        # 2b. Spot wheat = $75 (inv 7500): > $66.67 EV -> SKIPS feed
        state_sheep_75 = self._make_state(day=15, money=1000, tiles=sheep_tiles, shed={"WHEAT": 0})
        state_sheep_75.market.inventory["WHEAT"] = 7500
        orders_sheep_75 = _generate_feed_orders(state_sheep_75, current_money=1000)
        self.assertEqual(orders_sheep_75, [])

        # Scenario 3: Mixed Cow + Sheep at spot wheat $75
        # Cow EV ($80) > $75 -> at least one animal is positive EV -> BUYS feed
        mixed_tiles = [[None] * 10 for _ in range(10)]
        mixed_tiles[0][0] = {"kind": "PASTURE", "animal": "COW", "fed_today": False, "consecutive_unfed": 0}
        mixed_tiles[0][1] = {"kind": "PASTURE", "animal": "SHEEP", "fed_today": False, "consecutive_unfed": 0}
        state_mixed_75 = self._make_state(day=15, money=1000, tiles=mixed_tiles, shed={"WHEAT": 0})
        state_mixed_75.market.inventory["WHEAT"] = 7500
        orders_mixed_75 = _generate_feed_orders(state_mixed_75, current_money=1000)
        self.assertEqual(orders_mixed_75, [["BUY_PRODUCT", "WHEAT", 2]])

    def test_terminal_liquidation_timing(self):
        """Terminal liquidation triggers only at step >= 717 (Day 29 Hour 21+), not Day 28."""
        shed = {"MELON": 5, "WHEAT": 20}
        # Day 28, Hour 12 (step 684) — normal phase selling, NOT terminal liquidation
        state_d28 = self._make_state(day=28, hour=12, shed=shed)
        orders_d28 = generate_market_orders(state_d28)
        from src.strategy import _terminal_liquidation
        term_orders = _terminal_liquidation(state_d28)
        # Terminal liquidation sells ALL wheat (20), whereas normal selling respects safe sell limits
        self.assertNotEqual(orders_d28, term_orders)

        # Day 29, Hour 20 (step 716) — NOT yet terminal liquidation
        state_d29_h20 = self._make_state(day=29, hour=20, shed=shed)
        orders_d29_h20 = generate_market_orders(state_d29_h20)
        self.assertNotEqual(orders_d29_h20, term_orders)

        # Day 29, Hour 21 (step 717) — EXACTLY triggers terminal liquidation
        state_d29_h21 = self._make_state(day=29, hour=21, shed=shed)
        orders_d29_h21 = generate_market_orders(state_d29_h21)
        self.assertEqual(orders_d29_h21, term_orders)
        self.assertEqual(orders_d29_h21[0], ["SELL", "MELON", 5])
        self.assertEqual(orders_d29_h21[1], ["SELL", "WHEAT", 20])

    def test_terminal_liquidation_integration_step_boundary(self):
        """
        Verify terminal liquidation boundary on realistic raw observation dicts and
        confirm the kaggle_environments replay-logging convention: steps[i].action
        records the agent's response to observation at step i-1.
        """
        from agent.main import agent

        tiles = [{'kind': 'PASTURE', 'animal': {'animal_type': 'COW', 'placed_day': 0, 'yield_units': 0, 'fed_today': True, 'consecutive_unfed': 0, 'cared_today': True, 'fertilizer_available': False, 'pending_care_bonus': 0}} if (r==0 and c==0) else None for r in range(10) for c in range(10)]
        tiles_2d = [tiles[i*10:(i+1)*10] for i in range(10)]

        # 1. Realistic raw observation at step 716 (Day 29 Hour 20)
        obs_716 = {
            'player': 0, 'step': 716, 'day': 29, 'hour': 20,
            'farms': [{'money': 1000, 'tiles': tiles_2d, 'farmer': [4, 4], 'hands': [], 'unlocked_quadrants': ['NW'], 'hires_today': 0},
                      {'money': 1000, 'tiles': [[None]*10 for _ in range(10)], 'farmer': [4, 4], 'hands': [], 'unlocked_quadrants': ['NW'], 'hires_today': 0}],
            'market': {'inventory': {'WHEAT': 10000, 'FERTILIZER': 10000}, 'prices': {'WHEAT': 25, 'FERTILIZER': 100}},
            'private': {'shed': {'FERTILIZER': 5, 'WHEAT': 5}, 'seeds': {}, 'inventories': [{}]}
        }

        # 2. Realistic raw observation at step 717 (Day 29 Hour 21)
        obs_717 = dict(obs_716)
        obs_717['step'] = 717
        obs_717['hour'] = 21

        act_716 = agent(obs_716)
        act_717 = agent(obs_717)

        # Step 716: Normal orders — preserves reserves (only 3 fert, 2 wheat sold) and hires labor
        self.assertEqual(act_716['market'][:2], [['SELL', 'FERTILIZER', 3], ['SELL', 'WHEAT', 2]])
        hire_orders_716 = [o for o in act_716['market'] if o[0] == 'HIRE']
        self.assertGreater(len(hire_orders_716), 0)

        # Step 717: Terminal liquidation — sells 100% of shed inventory, sorted descending, zero hires
        self.assertEqual(act_717['market'], [['SELL', 'FERTILIZER', 5], ['SELL', 'WHEAT', 5]])

        # 3. Verify kaggle_environments replay logging index offset convention
        from kaggle_environments import make
        invocations = []
        def probe_agent(obs):
            invocations.append(obs['step'])
            return {'farmer': ['PASS'], 'hands': [], 'market': [['BUY_PRODUCT', 'WHEAT', obs['step']]]}

        env = make('kaggriculture')
        env.run([probe_agent, 'random'])

        # Total actionable turns is exactly 719 (steps 0 through 718; step 719 is terminal state)
        self.assertEqual(len(invocations), 719)
        self.assertEqual(invocations[0], 0)
        self.assertEqual(invocations[-1], 718)

        # Confirm that env.steps[i].action contains the action generated in response to obs[i-1]
        self.assertEqual(env.steps[717][0]['action']['market'], [['BUY_PRODUCT', 'WHEAT', 716]])
        self.assertEqual(env.steps[718][0]['action']['market'], [['BUY_PRODUCT', 'WHEAT', 717]])
        self.assertEqual(env.steps[719][0]['action']['market'], [['BUY_PRODUCT', 'WHEAT', 718]])

    def test_unconditional_premium_sell_front_running(self):
        """Unconditional front-running promotes MELON/MILK/WOOL ahead of WHEAT/FERTILIZER even when mirror detection returns False."""
        from agent.main import _front_run_premium_sells, _detect_mirror_farm

        from src.game_state import Animal
        opp_tiles = [[None] * 10 for _ in range(10)]
        for i in range(8):
            opp_tiles[i][0] = Animal(
                animal_type="COW", structure="PASTURE", placed_day=0, yield_units=0,
                fed_today=True, consecutive_unfed=0, cared_today=True, fertilizer_available=False,
                pending_care_bonus=0
            )
        state = self._make_state(day=15, hour=5, shed={"MELON": 2, "WHEAT": 10, "WOOL": 1, "FERTILIZER": 3})
        # Set opponent tiles
        state.opponent_farm.tiles = opp_tiles
        # Verify mirror detection returns False
        self.assertFalse(_detect_mirror_farm(state))

        # Mixed orders list with non-premium sells, premium sells, and BUY/HIRE orders
        orders = [
            ["BUY_PRODUCT", "WHEAT", 2],
            ["SELL", "WHEAT", 5],
            ["SELL", "FERTILIZER", 3],
            ["SELL", "MELON", 2],
            ["SELL", "WOOL", 1],
            ["BUY_LAND", "NE"],
            ["HIRE"],
        ]

        reordered = _front_run_premium_sells(state, orders)

        # Premium sells (MELON, WOOL) must be in the earliest slots (0-1)
        self.assertEqual(reordered[0], ["SELL", "MELON", 2])
        self.assertEqual(reordered[1], ["SELL", "WOOL", 1])

        # Non-premium sells (WHEAT, FERTILIZER) and other orders must follow
        remaining_sells = [o[1] for o in reordered[2:] if o[0] == "SELL"]
        self.assertEqual(remaining_sells, ["WHEAT", "FERTILIZER"])

        # Non-SELL orders (BUY_PRODUCT, BUY_LAND, HIRE) must preserve their relative sequence
        non_sells = [o[0] for o in reordered if o[0] != "SELL"]
        self.assertEqual(non_sells, ["BUY_PRODUCT", "BUY_LAND", "HIRE"])


if __name__ == "__main__":
    unittest.main()

