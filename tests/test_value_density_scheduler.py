"""
Unit and regression tests for Value-Density Task Assignment in src/scheduler.py.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_state import GameState, Animal, Plant
from src.scheduler import (
    Task, assign_tasks, _score_task_value, reset_scheduler_state,
    PRIORITY_FEED, PRIORITY_WATER, PRIORITY_HARVEST_ANIMAL, PRIORITY_HARVEST_PLANT,
)


class TestValueDensityScheduler(unittest.TestCase):
    """Test value-density task assignment logic and task valuation."""

    def setUp(self):
        reset_scheduler_state()

    def _make_state(self, day=15, hour=0, farmer_pos=(4, 4), hands=None, tiles=None, inventories=None, shed=None, market_prices=None):
        if tiles is None:
            tiles = [[None] * 10 for _ in range(10)]
        if hands is None:
            hands = []
        if inventories is None:
            inventories = [{} for _ in range(len(hands) + 1)]
        if shed is None:
            shed = {}

        prices = {"WHEAT": 25, "MILK": 160, "WOOL": 200, "MELON": 250, "STRAWBERRY": 150}
        if market_prices:
            prices.update(market_prices)

        obs = {
            "player": 0,
            "day": day,
            "hour": hour,
            "step": day * 24 + hour,
            "farms": [
                {
                    "money": 5000,
                    "tiles": tiles,
                    "farmer": list(farmer_pos),
                    "hands": list(hands),
                    "unlocked_quadrants": ["NW", "NE", "SW"],
                    "hires_today": len(hands),
                },
                {
                    "money": 5000,
                    "tiles": [[None] * 10 for _ in range(10)],
                    "farmer": [4, 4],
                    "hands": [],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                },
            ],
            "market": {
                "inventory": {"WHEAT": 10000, "MILK": 10000, "WOOL": 10000, "MELON": 10000, "STRAWBERRY": 10000},
                "prices": prices,
            },
            "town": {"unlocked_shops": []},
            "private": {
                "shed": shed,
                "seeds": {},
                "inventories": inventories,
            },
        }
        return GameState.from_obs(obs)

    def test_task_scoring_values(self):
        """Verify task scoring reflects economic values of products and actions."""
        tiles = [[None] * 10 for _ in range(10)]
        # Cow at (0, 0)
        tiles[0][0] = {
            "kind": "PASTURE", "animal": "COW", "placed_day": 5, "yield_units": 1,
            "fed_today": True, "consecutive_unfed": 0, "cared_today": False,
            "fertilizer_available": False, "pending_care_bonus": 0,
        }
        # Melon plant at (1, 1)
        tiles[1][1] = {
            "kind": "PLANT", "crop": "MELON", "planted_day": 5, "watered_today": False,
            "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": -1,
            "fertilized_until_day": -1,
        }

        state = self._make_state(day=15, tiles=tiles)

        t_harvest_cow = Task(priority=PRIORITY_HARVEST_ANIMAL, action=["HARVEST"], target_pos=(0, 0))
        val_harvest = _score_task_value(t_harvest_cow, state)
        self.assertGreaterEqual(val_harvest, 160.0, "Harvesting cow must reflect milk spot/base price ($160)")

        t_water_melon = Task(priority=PRIORITY_WATER, action=["WATER"], target_pos=(1, 1))
        val_water = _score_task_value(t_water_melon, state)
        self.assertGreaterEqual(val_water, 30.0, "Watering melon must reflect crop value")

    def test_value_density_prioritizes_high_density_task(self):
        """
        Verify that with SCHEDULER_ASSIGNMENT=value_density:
        A high-value task near Hand 1 is prioritized over distant tasks,
        and value competes directly with distance.
        """
        os.environ["SCHEDULER_ASSIGNMENT"] = "value_density"
        try:
            tiles = [[None] * 10 for _ in range(10)]
            # Cow harvest at (1, 0)
            tiles[0][1] = {
                "kind": "PASTURE", "animal": "COW", "placed_day": 5, "yield_units": 1,
                "fed_today": True, "consecutive_unfed": 0, "cared_today": False,
                "fertilizer_available": False, "pending_care_bonus": 0,
            }
            # Wheat water at (8, 8)
            tiles[8][8] = {
                "kind": "PLANT", "crop": "WHEAT", "planted_day": 10, "watered_today": False,
                "consecutive_unwatered": 0, "yield_units": 0, "max_lifespan_step": -1,
                "fertilized_until_day": -1,
            }

            # Hand 1 is at (0, 0), 1 step from cow at (1, 0)
            state = self._make_state(
                day=15,
                farmer_pos=(9, 9),
                hands=[(0, 0)],
                tiles=tiles,
            )

            tasks = [
                Task(priority=PRIORITY_WATER, action=["WATER"], target_pos=(8, 8), description="Water wheat"),
                Task(priority=PRIORITY_HARVEST_ANIMAL, action=["HARVEST"], target_pos=(1, 0), description="Harvest cow"),
            ]

            farmer_act, hands_act = assign_tasks(state, tasks)
            # Hand 0 at (0, 0) must move EAST towards cow at (1, 0) due to high value density
            self.assertEqual(hands_act[0], ["EAST"], "Value density must send Hand 1 east to harvest cow")

        finally:
            os.environ.pop("SCHEDULER_ASSIGNMENT", None)

    def test_regression_feed_starvation_wins_value_density(self):
        """
        Regression test for feed-delivery starvation defect:
        Scenario:
        - Hand 0 carrying WHEAT at (1, 1).
        - Sheep at (1, 2) (1 tile away) with consecutive_unfed=1 (acute danger: 1 day from death).
        - Mature melon plant at (2, 1) (1 tile away) with harvest worth $200.
        Both tasks are at equal Manhattan distance (1 tile) from Hand 0.
        Assert:
        1. FEED task value reflects asset preservation (>= $400) and beats $200 harvest.
        2. In value-density task assignment, FEED wins the match over HARVEST even when HARVEST is listed first.
        3. Hand 0 moves SOUTH toward the endangered sheep at (1, 2).
        """
        os.environ["SCHEDULER_ASSIGNMENT"] = "value_density"
        try:
            tiles = [[None] * 10 for _ in range(10)]
            # Endangered sheep at (1, 2)
            tiles[2][1] = {
                "kind": "PASTURE", "animal": "SHEEP", "placed_day": 5, "yield_units": 0,
                "fed_today": False, "consecutive_unfed": 1, "cared_today": False,
                "fertilizer_available": False, "pending_care_bonus": 0,
            }
            # Mature melon at (2, 1)
            tiles[1][2] = {
                "kind": "PLANT", "crop": "MELON", "planted_day": 5, "watered_today": True,
                "consecutive_unwatered": 0, "yield_units": 1, "max_lifespan_step": -1,
                "fertilized_until_day": -1,
            }

            # Hand 0 at (1, 1), carrying 1 wheat
            state = self._make_state(
                day=15,
                hour=12,
                farmer_pos=(9, 9),
                hands=[(1, 1)],
                tiles=tiles,
                inventories=[{}, {"WHEAT": 1}],
                market_prices={"MELON": 200, "WOOL": 160},
            )

            t_feed = Task(priority=PRIORITY_FEED, action=["FEED"], target_pos=(1, 2), description="Feed Sheep at (1,2)")
            t_harvest = Task(priority=PRIORITY_HARVEST_PLANT, action=["HARVEST"], target_pos=(2, 1), description="Harvest Melon at (2,1)")

            val_feed = _score_task_value(t_feed, state)
            val_harvest = _score_task_value(t_harvest, state)

            self.assertGreaterEqual(val_feed, 400.0, "Feed task for unfed=1 animal must score >= $400 for asset preservation")
            self.assertEqual(val_harvest, 200.0, "Melon harvest must score exactly $200")
            self.assertGreater(val_feed, val_harvest, "Feed asset preservation value must exceed $200 harvest")

            # Place harvest first in task list to verify list order does NOT override value density
            tasks = [t_harvest, t_feed]
            farmer_act, hands_act = assign_tasks(state, tasks)

            self.assertEqual(t_feed.assigned_to, 1, "Hand 0 (index 1) must be assigned to FEED task via value density")
            self.assertEqual(hands_act[0], ["SOUTH"], "Hand 0 at (1, 1) must move SOUTH toward sheep at (1, 2), not EAST toward melon")

        finally:
            os.environ.pop("SCHEDULER_ASSIGNMENT", None)

    def test_regression_pickup_wheat_starvation_urgency(self):
        """
        Verify that when an animal is in acute danger (consecutive_unfed=1),
        a hand without wheat near the shed prioritizes PICKUP WHEAT over a nearby $150 harvest.
        """
        os.environ["SCHEDULER_ASSIGNMENT"] = "value_density"
        try:
            tiles = [[None] * 10 for _ in range(10)]
            # Endangered cow at (1, 4)
            tiles[4][1] = {
                "kind": "PASTURE", "animal": "COW", "placed_day": 3, "yield_units": 0,
                "fed_today": False, "consecutive_unfed": 1, "cared_today": False,
                "fertilizer_available": False, "pending_care_bonus": 0,
            }
            # Strawberry harvest at (3, 3) (worth $150)
            tiles[3][3] = {
                "kind": "PLANT", "crop": "STRAWBERRY", "planted_day": 5, "watered_today": True,
                "consecutive_unwatered": 0, "yield_units": 1, "max_lifespan_step": -1,
                "fertilized_until_day": -1,
            }

            # Hand 0 at (3, 4) — 1 step from shed at (4, 4), and 1 step from strawberry at (3, 3)
            # Hand 0 has empty inventory, shed has 10 wheat
            state = self._make_state(
                day=15,
                hour=14,
                farmer_pos=(9, 9),
                hands=[(3, 4)],
                tiles=tiles,
                shed={"WHEAT": 10},
                inventories=[{}, {}],
                market_prices={"STRAWBERRY": 150},
            )

            t_pickup = Task(priority=PRIORITY_FEED, action=["PICKUP", "WHEAT", "1"], target_pos=(4, 4), description="Pickup wheat")
            t_harvest = Task(priority=PRIORITY_HARVEST_PLANT, action=["HARVEST"], target_pos=(3, 3), description="Harvest Strawberry")

            val_pickup = _score_task_value(t_pickup, state)
            val_harvest = _score_task_value(t_harvest, state)

            self.assertGreaterEqual(val_pickup, 400.0, "Pickup wheat must inherit acute asset preservation value (>= $400)")
            self.assertEqual(val_harvest, 150.0, "Strawberry harvest must score $150")

            tasks = [t_harvest, t_pickup]
            farmer_act, hands_act = assign_tasks(state, tasks)

            self.assertEqual(t_pickup.assigned_to, 1, "Hand 0 must be assigned to PICKUP WHEAT over $150 harvest")
            self.assertEqual(hands_act[0], ["EAST"], "Hand 0 at (3, 4) must move EAST to shed at (4, 4)")

        finally:
            os.environ.pop("SCHEDULER_ASSIGNMENT", None)



if __name__ == "__main__":
    unittest.main()
