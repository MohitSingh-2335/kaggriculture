"""
Unit and regression tests for Opponent Observation Tracker & Ledger Debt Tracking.
"""

import unittest
from types import SimpleNamespace
from collections import defaultdict

from src.opponent_tracker import OpponentTracker, get_opponent_tracker, reset_opponent_tracker
from src.constants import ALL_PRODUCTS


class MockTown:
    def __init__(self, unlocked_shops=None):
        self.unlocked_shops = unlocked_shops or []


class MockMarket:
    def __init__(self, inventory=None):
        self.inventory = inventory or {p: 10000 for p in ALL_PRODUCTS}


class MockObs:
    def __init__(self, step=0, market=None, town=None):
        self.step = step
        self.market = market or MockMarket()
        self.town = town or MockTown()


class TestOpponentTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = OpponentTracker()

    def test_reset_on_step_zero(self):
        """Verify tracker resets state when step == 0."""
        obs0 = MockObs(step=0)
        self.tracker.update(obs0, [])
        self.tracker.total_sales["MILK"] = 50

        # Next turn step 0 (e.g. new episode)
        obs_new = MockObs(step=0)
        self.tracker.update(obs_new, [])
        self.assertEqual(self.tracker.total_sales["MILK"], 0)
        self.assertEqual(self.tracker.last_step, 0)

    def test_reset_on_step_decrease(self):
        """Verify tracker resets state if step < last_step (sub-episode or test reset)."""
        obs10 = MockObs(step=10)
        self.tracker.update(obs10, [])
        self.tracker.total_sales["WOOL"] = 25

        obs5 = MockObs(step=5)
        self.tracker.update(obs5, [])
        self.assertEqual(self.tracker.total_sales["WOOL"], 0)
        self.assertEqual(self.tracker.last_step, 5)

    def test_opponent_sell_inference_no_town(self):
        """
        At step 1 (not a town shop step % 4 or center % 24),
        if we sell 5 MILK and market inv increases by 15 MILK,
        opponent sold 10 MILK.
        """
        # Step 1
        inv1 = {p: 10000 for p in ALL_PRODUCTS}
        obs1 = MockObs(step=1, market=MockMarket(inv1))
        self.tracker.update(obs1, our_submitted_orders=[])

        # Step 2: inv has increased by 15 for MILK; our step 1 submitted orders had 5 MILK
        inv2 = dict(inv1)
        inv2["MILK"] += 15
        obs2 = MockObs(step=2, market=MockMarket(inv2))
        self.tracker.update(obs2, our_submitted_orders=[["SELL", "MILK", 5]])

        # Step 1 was day 0 hour 1. Opponent net should be 15 - 5 = 10 sold.
        self.assertEqual(self.tracker.history[0][1]["MILK"]["sold"], 10)
        self.assertEqual(self.tracker.hourly_sales[1]["MILK"], 10)
        self.assertEqual(self.tracker.total_sales["MILK"], 10)

    def test_opponent_sell_inference_with_town_shop_consumption(self):
        """
        At step 4 (town shop tick step % 4 == 0), Bakery consumes 1 WHEAT, 1 EGG.
        If we sell 0 WHEAT and market inv remains 10000,
        opponent must have sold 1 WHEAT to replace Bakery's consumption!
        """
        # Step 4 with Bakery unlocked
        town = MockTown(["BAKERY"])
        inv4 = {p: 10000 for p in ALL_PRODUCTS}
        obs4 = MockObs(step=4, market=MockMarket(inv4), town=town)
        self.tracker.update(obs4, our_submitted_orders=[])

        # Step 5: market inventory still 10000
        inv5 = dict(inv4)
        obs5 = MockObs(step=5, market=MockMarket(inv5), town=town)
        self.tracker.update(obs5, our_submitted_orders=[])

        # Delta = 0 - 0 + 1 = 1 WHEAT sold by opponent
        self.assertEqual(self.tracker.history[0][4]["WHEAT"]["sold"], 1)
        self.assertEqual(self.tracker.history[0][4]["EGG"]["sold"], 1)
        self.assertEqual(self.tracker.total_sales["WHEAT"], 1)

    def test_pattern_detection_hour_1_rhythm(self):
        """Verify detect_liquidation_rhythm detects recurring dumps at Hour 1."""
        # Simulate opponent selling 10 MILK at Hour 1 on Day 1, Day 2, and Day 3
        for day in range(1, 4):
            self.tracker.history[day][1]["MILK"]["sold"] = 10
            self.tracker.hourly_sales[1]["MILK"] += 10
            self.tracker.total_sales["MILK"] += 10

        # At Day 4 Hour 0, upcoming hour is Hour 1
        will_dump, est_vol = self.tracker.detect_liquidation_rhythm("MILK", current_day=4, current_hour=0)
        self.assertTrue(will_dump)
        self.assertGreaterEqual(est_vol, 5)

        # At Day 4 Hour 5, upcoming hour is Hour 6 (no dump expected)
        will_dump_h6, _ = self.tracker.detect_liquidation_rhythm("MILK", current_day=4, current_hour=5)
        self.assertFalse(will_dump_h6)

    def test_ledger_debt_prevents_double_selling_regression(self):
        """
        REGRESSION TEST: Confirm that shifting a sale earlier and recording
        ledger debt prevents selling that same inventory on its original schedule.
        """
        # Scenario:
        # Day 5 Hour 0 (step = 5 * 24 + 0 = 120):
        # Shed has 10 WOOL.
        # Opponent is detected to dump WOOL at Hour 1 (step = 121).
        # We pre-emptively sell 10 WOOL at Hour 0.
        current_step = 120
        orig_step = 121
        shed_wool = 10

        # Check initial debt is 0
        self.assertEqual(self.tracker.get_ledger_debt("WOOL"), 0)

        # Shift sale to current_step (Hour 0)
        shift_qty = shed_wool
        self.tracker.record_preemptive_sale("WOOL", shift_qty, current_step, orig_step)

        # Confirm ledger debt is recorded
        self.assertEqual(self.tracker.get_ledger_debt("WOOL"), 10)

        # Now advance to orig_step (Hour 1, step 121)
        # In a real environment, our 10 wool was sold from shed, so shed has 0.
        # But even if shed somehow had 10 wool (e.g. harvest landed or test condition):
        shed_wool_at_orig = 10
        effective_sellable = max(0, shed_wool_at_orig - self.tracker.get_ledger_debt("WOOL"))

        # MUST BE 0! The debt prevents selling the 10 wool again!
        self.assertEqual(effective_sellable, 0, "Ledger debt must offset shed quantity to prevent double selling!")

        # Now advance past orig_step to Hour 2 (step 122)
        # Debt should expire
        self.tracker.clear_expired_debt(122)
        self.assertEqual(self.tracker.get_ledger_debt("WOOL"), 0, "Debt must clear after the original schedule window passes")

        # After debt is cleared, fresh produce in shed can be sold normally
        fresh_wool = 4
        effective_fresh = max(0, fresh_wool - self.tracker.get_ledger_debt("WOOL"))
        self.assertEqual(effective_fresh, 4)


if __name__ == "__main__":
    unittest.main()
