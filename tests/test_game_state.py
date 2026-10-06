"""
Unit tests for game state parsing.
"""

import sys
import os
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_state import GameState, Plant, Animal, Weed, EmptyStructure, Farm


class TestGameStateParsing(unittest.TestCase):
    """Test parsing raw observation dicts into typed GameState objects."""

    def _make_obs(self, **overrides):
        """Create a minimal valid observation dict."""
        obs = {
            "player": 0,
            "day": 5,
            "hour": 12,
            "farms": [
                {
                    "money": 3000,
                    "tiles": [[None] * 10 for _ in range(10)],
                    "farmer": [4, 4],
                    "hands": [],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                },
                {
                    "money": 2800,
                    "tiles": [[None] * 10 for _ in range(10)],
                    "farmer": [4, 4],
                    "hands": [],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                },
            ],
            "market": {
                "inventory": {"WHEAT": 10000, "CARROT": 10000},
                "prices": {"WHEAT": 25, "CARROT": 35},
            },
            "town": {
                "unlocked_shops": [],
            },
            "private": {
                "shed": {"WHEAT": 5},
                "seeds": {"WHEAT": 2},
                "inventories": [{}],
            },
        }
        obs.update(overrides)
        return obs

    def test_basic_parsing(self):
        """Test that a basic observation parses without error."""
        obs = self._make_obs()
        state = GameState.from_obs(obs)
        self.assertEqual(state.player, 0)
        self.assertEqual(state.day, 5)
        self.assertEqual(state.hour, 12)
        self.assertAlmostEqual(state.my_money, 3000)
        self.assertAlmostEqual(state.opponent_money, 2800)

    def test_turn_calculation(self):
        """Test absolute turn number."""
        obs = self._make_obs(day=5, hour=12)
        state = GameState.from_obs(obs)
        self.assertEqual(state.turn, 5 * 24 + 12)
        self.assertEqual(state.turns_remaining, 720 - (5 * 24 + 12))

    def test_plant_parsing(self):
        """Test that plant tiles are parsed into Plant objects."""
        obs = self._make_obs()
        obs["farms"][0]["tiles"][3][2] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 3,
            "watered_today": False,
            "consecutive_unwatered": 1,
            "yield_units": 0,
            "max_lifespan_step": -1,
            "fertilized_until_day": -1,
        }
        state = GameState.from_obs(obs)
        plants = state.my_farm.all_plants()
        self.assertEqual(len(plants), 1)
        self.assertEqual(plants[0].crop, "WHEAT")
        self.assertEqual(plants[0].x, 2)
        self.assertEqual(plants[0].y, 3)
        self.assertTrue(plants[0].needs_water)
        self.assertFalse(plants[0].is_harvestable)

    def test_animal_parsing(self):
        """Test that animal tiles are parsed into Animal objects."""
        obs = self._make_obs()
        obs["farms"][0]["tiles"][1][1] = {
            "kind": "COOP",
            "animal": "GOOSE",
            "placed_day": 2,
            "yield_units": 2,
            "fed_today": True,
            "consecutive_unfed": 0,
            "cared_today": False,
            "fertilizer_available": True,
            "pending_care_bonus": 1,
        }
        state = GameState.from_obs(obs)
        animals = state.my_farm.all_animals()
        self.assertEqual(len(animals), 1)
        self.assertEqual(animals[0].animal_type, "GOOSE")
        self.assertTrue(animals[0].is_harvestable)
        self.assertFalse(animals[0].needs_feed)
        self.assertTrue(animals[0].needs_care)
        self.assertTrue(animals[0].can_collect_fertilizer)

    def test_weed_parsing(self):
        """Test that weed tiles are parsed."""
        obs = self._make_obs()
        obs["farms"][0]["tiles"][0][0] = {"kind": "WEED"}
        state = GameState.from_obs(obs)
        weeds = state.my_farm.all_weeds()
        self.assertEqual(len(weeds), 1)
        self.assertEqual(weeds[0].x, 0)
        self.assertEqual(weeds[0].y, 0)

    def test_locked_tiles(self):
        """Test that LOCKED tiles are preserved."""
        obs = self._make_obs()
        obs["farms"][0]["tiles"][0][5] = "LOCKED"
        state = GameState.from_obs(obs)
        tile = state.my_farm.tile_at(5, 0)
        self.assertEqual(tile, "LOCKED")

    def test_empty_structure_parsing(self):
        """Test empty coop/pasture without animal."""
        obs = self._make_obs()
        obs["farms"][0]["tiles"][2][2] = {
            "kind": "COOP",
            "animal": None,
        }
        state = GameState.from_obs(obs)
        structures = state.my_farm.empty_structures()
        self.assertEqual(len(structures), 1)
        self.assertEqual(structures[0].structure, "COOP")

    def test_private_state(self):
        """Test private state parsing."""
        obs = self._make_obs()
        state = GameState.from_obs(obs)
        self.assertEqual(state.private.shed_count("WHEAT"), 5)
        self.assertEqual(state.private.seed_count("WHEAT"), 2)
        self.assertEqual(state.private.total_seeds(), 2)

    def test_market_parsing(self):
        """Test market state parsing."""
        obs = self._make_obs()
        state = GameState.from_obs(obs)
        self.assertEqual(state.market.price_of("WHEAT"), 25)
        self.assertEqual(state.market.inventory_of("WHEAT"), 10000)

    def test_player_1_perspective(self):
        """Test that player=1 swaps farms correctly."""
        obs = self._make_obs(player=1)
        state = GameState.from_obs(obs)
        self.assertEqual(state.player, 1)
        # Player 1's farm should be farms[1]
        self.assertAlmostEqual(state.my_money, 2800)
        self.assertAlmostEqual(state.opponent_money, 3000)


class TestFarmHelpers(unittest.TestCase):
    """Test Farm helper methods."""

    def test_empty_tiles_with_locked(self):
        """Ensure LOCKED tiles are not included in empty_tiles."""
        obs = {
            "player": 0, "day": 0, "hour": 0,
            "farms": [
                {
                    "money": 3000,
                    "tiles": [
                        [None, None, None, None, None, "LOCKED", "LOCKED", "LOCKED", "LOCKED", "LOCKED"],
                    ] + [[None] * 5 + ["LOCKED"] * 5 for _ in range(4)] + \
                    [["LOCKED"] * 10 for _ in range(5)],
                    "farmer": [0, 0],
                    "hands": [],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                },
                {
                    "money": 3000,
                    "tiles": [[None] * 10 for _ in range(10)],
                    "farmer": [0, 0],
                    "hands": [],
                    "unlocked_quadrants": ["NW"],
                    "hires_today": 0,
                },
            ],
            "market": {"inventory": {}, "prices": {}},
            "town": {"unlocked_shops": []},
            "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        }
        state = GameState.from_obs(obs)
        empty = state.my_farm.empty_tiles()
        # NW quadrant has 25 tiles (5x5), all None
        self.assertEqual(len(empty), 25)


if __name__ == "__main__":
    unittest.main()
