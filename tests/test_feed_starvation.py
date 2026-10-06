"""
Regression tests for Feed EV starvation edge case.
Verifies that:
1. When animals face starvation risk (consecutive_unfed >= 1 or shed_wheat == 0) and liquidity
   is available, feed is purchased even when market wheat price exceeds the normal daily yield
   EV ceiling, protecting unamortized animal asset value.
2. In normal, non-starvation conditions, the standard dynamic EV ceiling continues to govern feed
   purchases to prevent overpaying for feed.
3. Liquidity constraints are strictly respected even under starvation override.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game_state import GameState, Animal
from src.strategy import _generate_feed_orders, generate_market_orders
from src.market_model import calculate_price


class TestFeedStarvation(unittest.TestCase):
    """Regression test suite for Feed EV starvation override."""

    def _make_state(
        self,
        day=18,
        hour=0,
        money=5000,
        wheat_inventory=9600,  # yields $45/wheat (above normal EV ceiling of ~$25)
        consecutive_unfed=1,
        shed_wheat=0,
        milk_price=50,
    ):
        tiles = [[None] * 10 for _ in range(10)]
        # Place a cow at (0, 0)
        tiles[0][0] = {
            "kind": "PASTURE",
            "animal": "COW",
            "placed_day": 7,
            "yield_units": 0,
            "fed_today": False,
            "consecutive_unfed": consecutive_unfed,
            "cared_today": False,
            "fertilizer_available": False,
            "pending_care_bonus": 0,
        }

        obs = {
            "player": 0,
            "day": day,
            "hour": hour,
            "step": day * 24 + hour,
            "farms": [
                {
                    "money": money,
                    "tiles": tiles,
                    "farmer": [4, 4],
                    "hands": [],
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
                "inventory": {"WHEAT": wheat_inventory, "COW": 10000, "MILK": 10000},
                "prices": {"WHEAT": calculate_price("WHEAT", wheat_inventory), "COW": 400, "MILK": milk_price},
            },
            "town": {"unlocked_shops": []},
            "private": {
                "shed": {"WHEAT": shed_wheat} if shed_wheat > 0 else {},
                "seeds": {},
                "inventories": [{}],
            },
        }
        return GameState.from_obs(obs)

    def test_starvation_override_purchases_expensive_wheat(self):
        """
        Scenario:
        - Spot wheat price is $45 (wheat market inventory = 9600).
        - Dynamic daily milk EV ceiling for Cow is ~ $25 (1.0 / interval * $50 = $25).
        - Cow is at consecutive_unfed = 1 with 0 wheat in shed (starvation risk).
        - Player holds $5,000 liquidity.
        
        Expected:
        - Starvation asset-preservation override activates.
        - Feed order IS generated despite unit_price ($45) > normal EV ceiling ($25).
        """
        state = self._make_state(
            day=18,
            consecutive_unfed=1,
            shed_wheat=0,
            wheat_inventory=9600,
            money=5000,
        )
        spot_wheat = calculate_price("WHEAT", 9600)
        self.assertEqual(spot_wheat, 45, "Wheat spot price should be $45 at inv=9600")

        feed_orders = _generate_feed_orders(state)
        self.assertTrue(len(feed_orders) > 0, "Feed orders must be generated during starvation risk")
        self.assertEqual(feed_orders[0][0], "BUY_PRODUCT")
        self.assertEqual(feed_orders[0][1], "WHEAT")
        self.assertGreater(feed_orders[0][2], 0)

    def test_normal_conditions_respect_daily_yield_ev_ceiling(self):
        """
        Counterfactual check:
        - Spot wheat price is $45 (above $25 daily yield EV ceiling).
        - Cow is healthy: consecutive_unfed = 0, shed has 5 wheat buffer.
        - Player holds $5,000 liquidity.

        Expected:
        - Starvation override does NOT activate.
        - Feed order is NOT generated because $45 exceeds the daily yield EV ceiling ($25).
        """
        state = self._make_state(
            day=18,
            consecutive_unfed=0,
            shed_wheat=5,
            wheat_inventory=9600,
            money=5000,
        )
        feed_orders = _generate_feed_orders(state)
        self.assertEqual(feed_orders, [], "Normal conditions must respect EV ceiling and not overpay for feed")

    def test_liquidity_guard_prevents_purchases_without_cash(self):
        """
        Scenario:
        - Cow faces starvation risk (consecutive_unfed = 1, shed_wheat = 0).
        - Wheat spot price is $45.
        - Player cash is only $30 (insufficient to afford even 1 wheat at $45).

        Expected:
        - Liquidity guard aborts buy loop; no orders generated.
        """
        state = self._make_state(
            day=18,
            consecutive_unfed=1,
            shed_wheat=0,
            wheat_inventory=9600,
            money=30,
        )
        feed_orders = _generate_feed_orders(state)
        self.assertEqual(feed_orders, [], "Insufficient liquidity must prevent buy orders")

    def test_full_market_orders_integration_includes_feed_order(self):
        """
        Verify that generate_market_orders(state) includes the starvation-overridden feed order
        in full agent execution.
        """
        state = self._make_state(
            day=18,
            consecutive_unfed=1,
            shed_wheat=0,
            wheat_inventory=9600,
            money=5000,
        )
        orders = generate_market_orders(state)
        wheat_orders = [o for o in orders if o[0] == "BUY_PRODUCT" and o[1] == "WHEAT"]
        self.assertTrue(len(wheat_orders) > 0, "generate_market_orders must output WHEAT buy order")


if __name__ == "__main__":
    unittest.main()
