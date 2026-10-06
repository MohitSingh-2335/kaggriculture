"""
Unit and regression tests for Opponent Pipeline Projection & Calibration Guard.
Verifies tile observation, harvest tracking, dump ratio calibration (unestablished / hoarder / quick dumper),
pipeline projection timing (1-2 turns ahead), and integration with strategy sell orders.
"""

import unittest
from collections import defaultdict
from src.opponent_tracker import OpponentTracker, get_opponent_tracker, reset_opponent_tracker
from src.game_state import GameState, Plant, Animal, Farm, Market, Town, PrivateState
from src.strategy import generate_market_orders


def _make_dummy_farm(tiles=None, money=1000):
    if tiles is None:
        tiles = [[None] * 10 for _ in range(10)]
    return Farm(
        money=money,
        tiles=tiles,
        farmer_pos=(4, 4),
        hands_positions=[],
        unlocked_quadrants=["NW"],
        hires_today=0,
    )


def _make_dummy_state(day=0, hour=0, step=0, opp_tiles=None, shed=None, market_inv=None):
    if shed is None:
        shed = {}
    if market_inv is None:
        market_inv = {"MELON": 10000, "MILK": 10000, "WOOL": 10000, "WHEAT": 10000}

    my_farm = _make_dummy_farm()
    opp_farm = _make_dummy_farm(tiles=opp_tiles)
    market = Market(inventory=market_inv, prices={k: 50 for k in market_inv})
    town = Town(unlocked_shops=[])
    private = PrivateState(shed=shed, seeds={}, inventories=[{}])

    return GameState(
        player=0,
        day=day,
        hour=hour,
        step=step,
        my_farm=my_farm,
        opponent_farm=opp_farm,
        market=market,
        town=town,
        private=private,
    )


class TestOpponentPipeline(unittest.TestCase):
    def setUp(self):
        reset_opponent_tracker()
        self.tracker = OpponentTracker()

    def test_tile_harvest_detection(self):
        """Verify that tile yield decreases are correctly identified as opponent harvests."""
        # Step 100: Opponent has 1 Melon with 6 units and 1 Cow with 1 unit
        tiles_step100 = [[None] * 10 for _ in range(10)]
        tiles_step100[0][0] = Plant(
            crop="MELON", planted_day=0, watered_today=True, consecutive_unwatered=0,
            yield_units=6, max_lifespan_step=-1, fertilized_until_day=-1, x=0, y=0
        )
        tiles_step100[0][1] = Animal(
            animal_type="COW", structure="PASTURE", placed_day=0, yield_units=1,
            fed_today=True, consecutive_unfed=0, cared_today=True, fertilizer_available=False,
            pending_care_bonus=0, x=1, y=0
        )
        state_100 = _make_dummy_state(day=4, hour=4, step=100, opp_tiles=tiles_step100)
        self.tracker.update({}, state=state_100)

        self.assertEqual(self.tracker.harvested_by_opp["MELON"], 0)
        self.assertEqual(self.tracker.harvested_by_opp["MILK"], 0)

        # Step 101: Opponent harvested the melon (tile becomes None) and milked the cow (yield_units -> 0)
        tiles_step101 = [[None] * 10 for _ in range(10)]
        tiles_step101[0][0] = None
        tiles_step101[0][1] = Animal(
            animal_type="COW", structure="PASTURE", placed_day=0, yield_units=0,
            fed_today=True, consecutive_unfed=0, cared_today=True, fertilizer_available=False,
            pending_care_bonus=0, x=1, y=0
        )
        state_101 = _make_dummy_state(day=4, hour=5, step=101, opp_tiles=tiles_step101)
        self.tracker.update({}, state=state_101)

        self.assertEqual(self.tracker.harvested_by_opp["MELON"], 6)
        self.assertEqual(self.tracker.harvested_by_opp["MILK"], 1)

    def test_calibration_guard_states(self):
        """Verify calibration transitions: UNESTABLISHED -> QUICK_DUMPER -> HOARDER."""
        # 1. 0 harvests: UNESTABLISHED
        ratio, status = self.tracker.get_dump_ratio("MELON")
        self.assertEqual(status, "UNESTABLISHED")

        # 2. 3 harvests (< 4): still UNESTABLISHED
        self.tracker.harvested_by_opp["MELON"] = 3
        self.tracker.total_sales["MELON"] = 3
        ratio, status = self.tracker.get_dump_ratio("MELON")
        self.assertEqual(status, "UNESTABLISHED")

        # 3. 6 harvests and 6 sales (ratio = 1.0 >= 0.60): QUICK_DUMPER
        self.tracker.harvested_by_opp["MELON"] = 6
        self.tracker.total_sales["MELON"] = 6
        ratio, status = self.tracker.get_dump_ratio("MELON")
        self.assertEqual(status, "QUICK_DUMPER")
        self.assertAlmostEqual(ratio, 1.0)

        # 4. 20 harvests and 2 sales (ratio = 0.10 < 0.60): HOARDER
        self.tracker.harvested_by_opp["MELON"] = 20
        self.tracker.total_sales["MELON"] = 2
        ratio, status = self.tracker.get_dump_ratio("MELON")
        self.assertEqual(status, "HOARDER")
        self.assertAlmostEqual(ratio, 0.10)

    def test_pipeline_projection_firing_and_suppression(self):
        """
        Verify that pipeline projection fires 1-2 turns ahead of harvest when QUICK_DUMPER,
        and is strictly blocked when UNESTABLISHED or HOARDER.
        """
        # Opponent has 6 MELON tiles planted on Day 0, maturing Day 10
        opp_tiles = [[None] * 10 for _ in range(10)]
        for i in range(6):
            opp_tiles[0][i] = Plant(
                crop="MELON", planted_day=0, watered_today=True, consecutive_unwatered=0,
                yield_units=6, max_lifespan_step=-1, fertilized_until_day=-1, x=i, y=0
            )

        # Day 9 Hour 22 (step 238): 2 turns ahead of Day 10 Hour 0 harvest
        state_day9_h22 = _make_dummy_state(day=9, hour=22, step=238, opp_tiles=opp_tiles)

        # Case A: UNESTABLISHED (no prior harvest history)
        pipe_dump, vol = self.tracker.project_pipeline_dump("MELON", 9, 22, 238, state=state_day9_h22)
        self.assertFalse(pipe_dump)
        self.assertEqual(vol, 0)
        will_preempt, _, reason = self.tracker.evaluate_preemption("MELON", 9, 22, 238, state=state_day9_h22)
        self.assertFalse(will_preempt)
        self.assertEqual(reason, "none")

        # Case B: HOARDER (harvested 10, sold 0)
        self.tracker.harvested_by_opp["MELON"] = 10
        self.tracker.total_sales["MELON"] = 0
        pipe_dump, vol = self.tracker.project_pipeline_dump("MELON", 9, 22, 238, state=state_day9_h22)
        self.assertFalse(pipe_dump)
        self.assertEqual(vol, 0)
        will_preempt, _, reason = self.tracker.evaluate_preemption("MELON", 9, 22, 238, state=state_day9_h22)
        self.assertFalse(will_preempt)
        self.assertEqual(reason, "none")

        # Case C: QUICK_DUMPER (harvested 10, sold 10)
        self.tracker.total_sales["MELON"] = 10
        pipe_dump, vol = self.tracker.project_pipeline_dump("MELON", 9, 22, 238, state=state_day9_h22)
        self.assertTrue(pipe_dump)
        self.assertGreater(vol, 0)
        will_preempt, pre_vol, reason = self.tracker.evaluate_preemption("MELON", 9, 22, 238, state=state_day9_h22)
        self.assertTrue(will_preempt)
        self.assertEqual(reason, "pipeline")
        self.assertGreaterEqual(pre_vol, 6)

    def test_pipeline_preemption_sell_order_integration(self):
        """
        Verify that strategy.py generates preemptive SELL orders ahead of harvest turn
        ONLY when calibration guard passes as QUICK_DUMPER.
        """
        global_tracker = get_opponent_tracker()
        global_tracker.reset()

        # Opponent has 6 Melons maturing Day 10
        opp_tiles = [[None] * 10 for _ in range(10)]
        for i in range(6):
            opp_tiles[0][i] = Plant(
                crop="MELON", planted_day=0, watered_today=True, consecutive_unwatered=0,
                yield_units=6, max_lifespan_step=-1, fertilized_until_day=-1, x=i, y=0
            )

        # We have 5 Melons in shed ready to sell, but market is oversupplied so normal sell does not trigger
        state = _make_dummy_state(
            day=9, hour=22, step=238, opp_tiles=opp_tiles, shed={"MELON": 5},
            market_inv={"MELON": 10200, "MILK": 10000, "WOOL": 10000, "WHEAT": 10000}
        )

        # Phase 1: Uncalibrated -> Preemption should NOT fire
        orders_uncal = generate_market_orders(state)
        sell_melons_uncal = [o for o in orders_uncal if o[0] == "SELL" and o[1] == "MELON"]
        self.assertEqual(len(sell_melons_uncal), 0)

        # Phase 2: Hoarder -> Preemption should NOT fire even with history
        global_tracker.harvested_by_opp["MELON"] = 12
        global_tracker.total_sales["MELON"] = 1  # dump ratio = 1/12 = 0.08 < 0.60
        orders_hoarder = generate_market_orders(state)
        sell_melons_hoarder = [o for o in orders_hoarder if o[0] == "SELL" and o[1] == "MELON"]
        self.assertEqual(len(sell_melons_hoarder), 0)

        # Phase 3: Quick Dumper -> Preemption MUST fire!
        global_tracker.total_sales["MELON"] = 12  # dump ratio = 1.0 >= 0.60
        orders_dumper = generate_market_orders(state)
        sell_melons_dumper = [o for o in orders_dumper if o[0] == "SELL" and o[1] == "MELON"]
        self.assertGreater(len(sell_melons_dumper), 0)
        self.assertEqual(sell_melons_dumper[0][0], "SELL")
        self.assertEqual(sell_melons_dumper[0][1], "MELON")
        self.assertGreater(sell_melons_dumper[0][2], 0)

    def test_fallback_to_reactive_rhythm(self):
        """
        Verify that when pipeline projection does not fire,
        evaluate_preemption seamlessly falls back to reactive rhythm detection.
        """
        # Opponent has no imminent crop harvest (empty farm)
        state = _make_dummy_state(day=5, hour=0, step=120)

        # Train a reactive Hour 1 dump rhythm for WOOL
        self.tracker.total_sales["WOOL"] = 20
        self.tracker.hourly_sales[1]["WOOL"] = 15

        # At Hour 0 (upcoming hour is 1): pipeline has 0 volume, but reactive rhythm fires
        will_preempt, vol, reason = self.tracker.evaluate_preemption("WOOL", 5, 0, 120, state=state)
        self.assertTrue(will_preempt)
        self.assertEqual(reason, "reactive")
        self.assertGreater(vol, 0)


if __name__ == "__main__":
    unittest.main()
