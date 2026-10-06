"""
Unit tests for scheduler task generation and assignment.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_state import GameState, Plant, Animal, Weed, EmptyStructure, Farm
from src.scheduler import (
    generate_tasks, assign_tasks, Task, reset_scheduler_state,
    PRIORITY_FEED, PRIORITY_WATER, PRIORITY_HARVEST_ANIMAL, PRIORITY_HARVEST_PLANT,
    PRIORITY_COLLECT_FERTILIZER, PRIORITY_PLANT, PRIORITY_BUILD, PRIORITY_PLACE,
    PRIORITY_DROP, PRIORITY_CARE, PRIORITY_DIG,
)


class TestScheduler(unittest.TestCase):
    """Test task generation, priorities, BUILD/PLACE pipeline, and weed priority."""

    def setUp(self):
        reset_scheduler_state()

    def _make_state(self, day=6, hour=0, money=3000, shed=None, seeds=None, tiles=None, farmer_pos=(4, 4), inventories=None, step=None, hands=None):
        if tiles is None:
            tiles = [[None] * 10 for _ in range(10)]
        if shed is None:
            shed = {}
        if seeds is None:
            seeds = {}
        if inventories is None:
            inventories = [{}]
        if hands is None:
            hands = []
        if step is None:
            step = day * 24 + hour

        obs = {
            "player": 0,
            "day": day,
            "hour": hour,
            "step": step,
            "farms": [
                {
                    "money": money,
                    "tiles": tiles,
                    "farmer": list(farmer_pos),
                    "hands": list(hands),
                    "unlocked_quadrants": ["NW"],
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
                "inventories": inventories,
            },
        }
        return GameState.from_obs(obs)

    def test_priority_ordering(self):
        """Verify priority constants are correctly ordered."""
        self.assertLess(PRIORITY_FEED, PRIORITY_WATER)
        self.assertLess(PRIORITY_WATER, PRIORITY_BUILD)
        self.assertLess(PRIORITY_BUILD, PRIORITY_HARVEST_ANIMAL)
        self.assertLess(PRIORITY_HARVEST_ANIMAL, PRIORITY_HARVEST_PLANT)
        self.assertLess(PRIORITY_HARVEST_PLANT, PRIORITY_COLLECT_FERTILIZER)
        self.assertLess(PRIORITY_COLLECT_FERTILIZER, PRIORITY_PLANT)
        self.assertEqual(PRIORITY_PLANT, PRIORITY_PLACE)
        self.assertLess(PRIORITY_PLANT, PRIORITY_DROP)
        self.assertLess(PRIORITY_DROP, PRIORITY_CARE)
        self.assertEqual(PRIORITY_DIG, 9)

    def test_build_then_place_sequence_end_to_end(self):
        """
        Test the 3-step animal pipeline:
        Step 1: Animal in shed, no structure -> BUILD_PASTURE generated.
        Step 2: Animal in shed, empty structure exists, unit inv empty -> PICKUP from shed.
        Step 3: Animal in unit inv, empty structure exists -> PLACE animal on structure.
        """
        # Step 1: Animal in shed, no structure on farm
        state_step1 = self._make_state(shed={"COW": 1}, farmer_pos=(0, 0))
        tasks_step1 = generate_tasks(state_step1)
        build_tasks = [t for t in tasks_step1 if t.priority == PRIORITY_BUILD and t.action[0] == "BUILD_PASTURE"]
        self.assertGreaterEqual(len(build_tasks), 1)

        # Step 2: Empty structure on farm, animal in shed, unit inv empty
        tiles_step2 = [[None] * 10 for _ in range(10)]
        tiles_step2[2][2] = {"kind": "PASTURE"}
        state_step2 = self._make_state(shed={"COW": 1}, tiles=tiles_step2, farmer_pos=(4, 4), inventories=[{}])
        tasks_step2 = generate_tasks(state_step2)
        pickup_tasks = [t for t in tasks_step2 if t.priority == PRIORITY_PLACE and t.action[0] == "PICKUP"]
        self.assertGreaterEqual(len(pickup_tasks), 1)
        self.assertEqual(pickup_tasks[0].action[1], "COW")

        # When farmer is at shed (4, 4), assign_tasks returns PICKUP
        farmer_act, _ = assign_tasks(state_step2, tasks_step2)
        self.assertEqual(farmer_act[0], "PICKUP")
        self.assertEqual(farmer_act[1], "COW")

        # Step 3: Animal in unit inventory, empty structure exists on farm
        state_step3 = self._make_state(shed={}, tiles=tiles_step2, farmer_pos=(2, 2), inventories=[{"COW": 1}])
        tasks_step3 = generate_tasks(state_step3)
        place_tasks = [t for t in tasks_step3 if t.priority == PRIORITY_PLACE and t.action[0] == "PLACE"]
        self.assertGreaterEqual(len(place_tasks), 1)
        self.assertEqual(place_tasks[0].action[1], "COW")
        self.assertEqual(place_tasks[0].target_pos, (2, 2))

        # When farmer is at structure (2, 2), assign_tasks returns PLACE COW
        farmer_act, _ = assign_tasks(state_step3, tasks_step3)
        self.assertEqual(farmer_act, ["PLACE", "COW"])

    def test_reactive_weed_repair_state_machine(self):
        """
        Test the reactive weed-repair state machine:
        Turn 1: Farmer at (2, 3) wants to PLANT WHEAT, but (2, 3) has a Weed.
                assign_tasks issues DIG and remembers intended action.
        Turn 2: Tile (2, 3) is now clear (None).
                assign_tasks automatically resumes remembered PLANT WHEAT action.
        """
        tiles_turn1 = [[None] * 10 for _ in range(10)]
        tiles_turn1[3][2] = {"kind": "WEED"}  # Weed at (x=2, y=3)

        state_turn1 = self._make_state(tiles=tiles_turn1, farmer_pos=(2, 3), step=1)
        plant_task = Task(priority=PRIORITY_PLANT, action=["PLANT", "WHEAT"], target_pos=(2, 3))
        
        # Turn 1: Farmer at (2, 3) with weed -> issues DIG
        farmer_act1, _ = assign_tasks(state_turn1, [plant_task])
        self.assertEqual(farmer_act1, ["DIG"])

        # Turn 2: Tile is cleared (None)
        tiles_turn2 = [[None] * 10 for _ in range(10)]
        state_turn2 = self._make_state(tiles=tiles_turn2, farmer_pos=(2, 3), step=2)

        # Turn 2: Farmer resumes remembered PLANT WHEAT action
        farmer_act2, _ = assign_tasks(state_turn2, [])
        self.assertEqual(farmer_act2, ["PLANT", "WHEAT"])

    def test_fair_pasture_allocation_round_robin(self):
        """
        Test that when both COW and SHEEP are in shed and pasture structures exist,
        PASTURE structures are allocated round-robin across species rather than
        COW greedily starving SHEEP.
        """
        tiles = [[None] * 10 for _ in range(10)]
        tiles[1][1] = {"kind": "PASTURE"}
        tiles[1][2] = {"kind": "PASTURE"}

        # Shed has 5 cows and 5 sheep, but only 2 pastures exist on farm
        state = self._make_state(shed={"COW": 5, "SHEEP": 5}, tiles=tiles)
        tasks = generate_tasks(state)

        pickup_tasks = [t for t in tasks if t.priority == PRIORITY_PLACE and t.action[0] == "PICKUP"]
        self.assertEqual(len(pickup_tasks), 2)

        # In round-robin, one pickup is COW and one pickup is SHEEP
        picked_species = [t.action[1] for t in pickup_tasks]
        self.assertIn("COW", picked_species)
        self.assertIn("SHEEP", picked_species)

    def test_shed_to_farm_placement_priority_dynamic_targets(self):
        """
        Test that shed-to-farm placement priority dynamically tracks TARGET_COWS
        and TARGET_SHEEP imported from src.strategy, rather than hardcoded 6.0/8.0.
        """
        import src.scheduler as sched
        orig_cows, orig_sheep = sched.TARGET_COWS, sched.TARGET_SHEEP

        try:
            # 1 empty pasture structure on farm
            tiles = [[None] * 10 for _ in range(10)]
            tiles[1][1] = {"kind": "PASTURE"}

            # Farm currently has 5 cows and 3 sheep
            tiles[0][0] = {"kind": "PASTURE", "animal": "COW"}
            tiles[0][1] = {"kind": "PASTURE", "animal": "COW"}
            tiles[0][2] = {"kind": "PASTURE", "animal": "COW"}
            tiles[0][3] = {"kind": "PASTURE", "animal": "COW"}
            tiles[0][4] = {"kind": "PASTURE", "animal": "COW"}
            tiles[2][0] = {"kind": "PASTURE", "animal": "SHEEP"}
            tiles[2][1] = {"kind": "PASTURE", "animal": "SHEEP"}
            tiles[2][2] = {"kind": "PASTURE", "animal": "SHEEP"}

            # Shed has both 1 COW and 1 SHEEP
            state = self._make_state(shed={"COW": 1, "SHEEP": 1}, tiles=tiles)

            # Case A: Default targets (8 Cows, 6 Sheep)
            # cows = 5/8 = 0.625, sheep = 3/6 = 0.500 -> SHEEP is further behind (0.500 < 0.625)
            sched.TARGET_COWS = 8
            sched.TARGET_SHEEP = 6
            tasks = generate_tasks(state)
            pickup = [t for t in tasks if t.priority == PRIORITY_PLACE and t.action[0] == "PICKUP"]
            self.assertEqual(len(pickup), 1)
            self.assertEqual(pickup[0].action[1], "SHEEP", "Under 8C/6S targets, SHEEP is further behind and must be placed first")

            # Case B: Reconfigured targets (12 Cows, 4 Sheep)
            # cows = 5/12 = 0.417, sheep = 3/4 = 0.750 -> COW is now further behind (0.417 < 0.750)
            sched.TARGET_COWS = 12
            sched.TARGET_SHEEP = 4
            tasks = generate_tasks(state)
            pickup = [t for t in tasks if t.priority == PRIORITY_PLACE and t.action[0] == "PICKUP"]
            self.assertEqual(len(pickup), 1)
            self.assertEqual(pickup[0].action[1], "COW", "Under 12C/4S targets, COW is further behind and must be placed first")

        finally:
            sched.TARGET_COWS = orig_cows
            sched.TARGET_SHEEP = orig_sheep

    def test_shortest_path_route_following(self):
        """
        Regression test: Verify a hand follows an actual shortest BFS path (cached turn-by-turn)
        rather than the old single-step diagonal-biased heuristic (which always chose horizontal
        when abs(dx) == abs(dy)).
        """
        import src.scheduler as sched
        from src.actions import move_toward, path_to

        reset_scheduler_state()
        start_pos = (4, 4)
        target_pos = (1, 1)

        # Confirm that old heuristic has diagonal bias (dx=-3, dy=-3 -> chose WEST)
        old_heuristic_step = move_toward(start_pos, target_pos)
        self.assertEqual(old_heuristic_step, "WEST", "Old heuristic had horizontal bias on diagonals")

        # Initial turn: Hand at (4, 4)
        state1 = self._make_state(
            farmer_pos=(9, 9),
            hands=[start_pos],
            inventories=[{}, {}],
        )
        task = Task(priority=PRIORITY_HARVEST_PLANT, action=["HARVEST"], target_pos=target_pos)
        farmer_act, hands_act = assign_tasks(state1, [task])

        # BFS explores NORTH first in DIRECTIONS order:
        expected_bfs_path = path_to(start_pos, target_pos)
        first_step = expected_bfs_path[0]
        self.assertEqual(first_step, "NORTH")
        self.assertEqual(hands_act[0], [first_step])
        self.assertNotEqual(hands_act[0], [old_heuristic_step])

        # Verify persistent path is cached for unit 1
        self.assertIn(1, sched._unit_paths)
        self.assertEqual(sched._unit_paths[1]["target_pos"], target_pos)
        self.assertEqual(sched._unit_paths[1]["path"], expected_bfs_path[1:])

        # Next turn: Hand moves to (4, 3), continuing along cached route without recomputing
        state2 = self._make_state(
            farmer_pos=(9, 9),
            hands=[(4, 3)],
            inventories=[{}, {}],
        )
        task2 = Task(priority=PRIORITY_HARVEST_PLANT, action=["HARVEST"], target_pos=target_pos)
        farmer_act2, hands_act2 = assign_tasks(state2, [task2])

        second_step = expected_bfs_path[1]
        self.assertEqual(hands_act2[0], [second_step])
        self.assertEqual(sched._unit_paths[1]["path"], expected_bfs_path[2:])

    def test_greedy_assignment_matches_original_order(self):
        """
        Regression test for Item 4:
        Confirm assignment order matches the original greedy nearest-hand-first algorithm's
        behavior, while movement follows BFS shortest paths.
        """
        reset_scheduler_state()
        # Hand 1 at (1, 0), Hand 2 at (5, 0)
        # Task A at (2, 0), Task B at (0, 0)
        # Greedy behavior:
        # Task A (first in list) is paired to nearest available hand -> Hand 1 (dist = |1-2| = 1).
        # Task B (second in list) is left with Hand 2 -> (dist = |5-0| = 5).
        state = self._make_state(
            farmer_pos=(9, 9),
            hands=[(1, 0), (5, 0)],
            inventories=[{}, {}, {}],
        )
        task_a = Task(priority=PRIORITY_HARVEST_PLANT, action=["HARVEST"], target_pos=(2, 0), description="Task A")
        task_b = Task(priority=PRIORITY_HARVEST_PLANT, action=["HARVEST"], target_pos=(0, 0), description="Task B")

        farmer_act, hands_act = assign_tasks(state, [task_a, task_b])

        # Task A should be assigned to Hand 1 (nearest available to Task A)
        # Task B should be assigned to Hand 2
        self.assertEqual(task_a.assigned_to, 1, "Task A should be greedily matched to Hand 1")
        self.assertEqual(task_b.assigned_to, 2, "Task B should be greedily matched to Hand 2")

        # Hand 1 at (1, 0) moves EAST toward (2, 0)
        # Hand 2 at (5, 0) moves WEST toward (0, 0)
        self.assertEqual(hands_act[0], ["EAST"])
        self.assertEqual(hands_act[1], ["WEST"])

    def test_terminal_recall_drops_inventory_by_step_716(self):
        """
        Regression test:
        1. At step >= 711 (Day 29 Hour 15), PLANT and WATER task generation is cancelled.
        2. A hand carrying inventory at step 711, several tiles from the shed (e.g. (4, 2)),
           is forced to path directly to the nearest shed tile (4, 4), overriding other tasks,
           and executes DROP by step 716 so inventory is sellable for step 717 terminal liquidation.
        """
        from src.strategy import get_planting_targets, get_build_targets, generate_market_orders

        reset_scheduler_state()

        # Place an unwatered plant at (1, 1) and weed at (0, 0)
        tiles = [[None] * 10 for _ in range(10)]
        tiles[1][1] = {
            "crop": "MELON", "age": 8, "watered_today": False,
            "consecutive_unwatered": 0, "yield_units": 0, "fertilized": False,
        }
        tiles[0][0] = "WEED"

        # Step 711: Day 29, Hour 15
        # Hand 1 is at (4, 2) (distance 2 from shed tile (4, 4)) carrying 2 MELON
        state_711 = self._make_state(
            day=29, hour=15, step=711,
            farmer_pos=(9, 9),
            hands=[(4, 2)],
            tiles=tiles,
            inventories=[{}, {"MELON": 2}],
            seeds={"MELON": 5},
        )

        # 1. Confirm PLANT and WATER task generation are completely cancelled at step 711
        tasks_711 = generate_tasks(state_711)
        water_tasks = [t for t in tasks_711 if t.action == ["WATER"]]
        plant_tasks = [t for t in tasks_711 if t.action and t.action[0] == "PLANT"]
        build_tasks = [t for t in tasks_711 if t.action and t.action[0].startswith("BUILD")]
        self.assertEqual(water_tasks, [], "Water tasks must be cancelled at step >= 711")
        self.assertEqual(plant_tasks, [], "Plant tasks must be cancelled at step >= 711")
        self.assertEqual(build_tasks, [], "Build tasks must be cancelled at step >= 711")

        # Confirm strategy targets are also empty
        self.assertEqual(get_planting_targets(state_711), [])
        self.assertEqual(get_build_targets(state_711), [])

        # 2. Hand carrying inventory at step 711 paths directly south toward shed (4, 4)
        farmer_act, hands_act = assign_tasks(state_711, tasks_711)
        self.assertEqual(hands_act[0], ["SOUTH"], "Step 711: Hand at (4, 2) must move SOUTH toward shed (4, 4)")

        # Step 712: Day 29, Hour 16
        # Hand 1 has moved to (4, 3), still carrying 2 MELON
        state_712 = self._make_state(
            day=29, hour=16, step=712,
            farmer_pos=(9, 9),
            hands=[(4, 3)],
            tiles=tiles,
            inventories=[{}, {"MELON": 2}],
        )
        tasks_712 = generate_tasks(state_712)
        farmer_act, hands_act = assign_tasks(state_712, tasks_712)
        self.assertEqual(hands_act[0], ["SOUTH"], "Step 712: Hand at (4, 3) must move SOUTH onto shed tile (4, 4)")

        # Step 713: Day 29, Hour 17
        # Hand 1 has arrived at shed tile (4, 4), still carrying 2 MELON
        state_713 = self._make_state(
            day=29, hour=17, step=713,
            farmer_pos=(9, 9),
            hands=[(4, 4)],
            tiles=tiles,
            inventories=[{}, {"MELON": 2}],
        )
        tasks_713 = generate_tasks(state_713)
        farmer_act, hands_act = assign_tasks(state_713, tasks_713)
        self.assertEqual(hands_act[0], ["DROP"], "Step 713: Hand at (4, 4) must execute DROP to empty inventory into shed")

        # Step 717: Day 29, Hour 21
        # Dropped produce is now in shed and liquidated by terminal liquidation market orders
        state_717 = self._make_state(
            day=29, hour=21, step=717,
            farmer_pos=(9, 9),
            hands=[(4, 4)],
            shed={"MELON": 2},
            inventories=[{}, {}],
        )
        market_orders = generate_market_orders(state_717)
        self.assertIn(["SELL", "MELON", 2], market_orders, "Step 717: Dropped melons must be sold by terminal liquidation")


if __name__ == "__main__":
    unittest.main()


