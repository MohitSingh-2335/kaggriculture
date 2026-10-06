"""
Unit tests for market price model.

Verifies the price function against known values from the competition spec table:

Resource     | Base | P(I0−T) | P(I0+T) | P(I0+2T)
-------------|------|---------|---------|----------
Wheat        |  25  |  $45    |  $20    |  $19
Carrot       |  35  |  $70    |  $10    |  $1
Tomato       |  60  |  $84    |  $24    |  $9
Strawberry   | 120  | $204    |  $1     |  $1
Melon        | 250  | $300    |  $1     |  $1
Egg          |  50  |  $70    |  $40    |  $39
Milk         | 160  | $256    |  $1     |  $1
Wool         | 200  | $240    |  $1     |  $1
Fertilizer   | 100  | $140    |  $60    |  $20
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.market_model import calculate_price, predict_sell_revenue, predict_buy_cost
from src.constants import MARKET_PARAMS, MARKET_I0


class TestPriceFunction(unittest.TestCase):
    """Test the price function against the spec table values."""

    def test_base_prices_at_I0(self):
        """At I0 (10000), price should equal base."""
        for product, params in MARKET_PARAMS.items():
            price = calculate_price(product, MARKET_I0)
            self.assertEqual(price, params["base"],
                           f"{product}: expected {params['base']} at I0, got {price}")

    def test_wheat_scarcity(self):
        """Wheat at I0-T=9600 should be $45."""
        T = MARKET_PARAMS["WHEAT"]["T"]
        price = calculate_price("WHEAT", MARKET_I0 - T)
        self.assertEqual(price, 45, f"Wheat P(I0-T): expected $45, got ${price}")

    def test_wheat_glut(self):
        """Wheat at I0+T=10400 should be $20."""
        T = MARKET_PARAMS["WHEAT"]["T"]
        price = calculate_price("WHEAT", MARKET_I0 + T)
        self.assertEqual(price, 20, f"Wheat P(I0+T): expected $20, got ${price}")

    def test_wheat_glut_2T(self):
        """Wheat at I0+2T=10800 should be $19."""
        T = MARKET_PARAMS["WHEAT"]["T"]
        price = calculate_price("WHEAT", MARKET_I0 + 2 * T)
        self.assertEqual(price, 19, f"Wheat P(I0+2T): expected $19, got ${price}")

    def test_carrot_scarcity(self):
        """Carrot at I0-T should be $70."""
        T = MARKET_PARAMS["CARROT"]["T"]
        price = calculate_price("CARROT", MARKET_I0 - T)
        self.assertEqual(price, 70, f"Carrot P(I0-T): expected $70, got ${price}")

    def test_carrot_glut(self):
        """Carrot at I0+T should be $10."""
        T = MARKET_PARAMS["CARROT"]["T"]
        price = calculate_price("CARROT", MARKET_I0 + T)
        self.assertEqual(price, 10, f"Carrot P(I0+T): expected $10, got ${price}")

    def test_carrot_glut_2T(self):
        """Carrot at I0+2T should be $1."""
        T = MARKET_PARAMS["CARROT"]["T"]
        price = calculate_price("CARROT", MARKET_I0 + 2 * T)
        self.assertEqual(price, 1, f"Carrot P(I0+2T): expected $1, got ${price}")

    def test_tomato_scarcity(self):
        """Tomato at I0-T should be $84."""
        T = MARKET_PARAMS["TOMATO"]["T"]
        price = calculate_price("TOMATO", MARKET_I0 - T)
        self.assertEqual(price, 84, f"Tomato P(I0-T): expected $84, got ${price}")

    def test_tomato_glut(self):
        """Tomato at I0+T should be $24."""
        T = MARKET_PARAMS["TOMATO"]["T"]
        price = calculate_price("TOMATO", MARKET_I0 + T)
        self.assertEqual(price, 24, f"Tomato P(I0+T): expected $24, got ${price}")

    def test_tomato_glut_2T(self):
        """Tomato at I0+2T should be $9."""
        T = MARKET_PARAMS["TOMATO"]["T"]
        price = calculate_price("TOMATO", MARKET_I0 + 2 * T)
        self.assertEqual(price, 9, f"Tomato P(I0+2T): expected $9, got ${price}")

    def test_strawberry_scarcity(self):
        """Strawberry at I0-T should be $204."""
        T = MARKET_PARAMS["STRAWBERRY"]["T"]
        price = calculate_price("STRAWBERRY", MARKET_I0 - T)
        self.assertEqual(price, 204, f"Strawberry P(I0-T): expected $204, got ${price}")

    def test_strawberry_glut(self):
        """Strawberry at I0+T should be $1."""
        T = MARKET_PARAMS["STRAWBERRY"]["T"]
        price = calculate_price("STRAWBERRY", MARKET_I0 + T)
        self.assertEqual(price, 1, f"Strawberry P(I0+T): expected $1, got ${price}")

    def test_melon_scarcity(self):
        """Melon at I0-T should be $300."""
        T = MARKET_PARAMS["MELON"]["T"]
        price = calculate_price("MELON", MARKET_I0 - T)
        self.assertEqual(price, 300, f"Melon P(I0-T): expected $300, got ${price}")

    def test_melon_glut(self):
        """Melon at I0+T should be $1."""
        T = MARKET_PARAMS["MELON"]["T"]
        price = calculate_price("MELON", MARKET_I0 + T)
        self.assertEqual(price, 1, f"Melon P(I0+T): expected $1, got ${price}")

    def test_egg_scarcity(self):
        """Egg at I0-T should be $70."""
        T = MARKET_PARAMS["EGG"]["T"]
        price = calculate_price("EGG", MARKET_I0 - T)
        self.assertEqual(price, 70, f"Egg P(I0-T): expected $70, got ${price}")

    def test_egg_glut(self):
        """Egg at I0+T should be $40."""
        T = MARKET_PARAMS["EGG"]["T"]
        price = calculate_price("EGG", MARKET_I0 + T)
        self.assertEqual(price, 40, f"Egg P(I0+T): expected $40, got ${price}")

    def test_egg_glut_2T(self):
        """Egg at I0+2T should be $39."""
        T = MARKET_PARAMS["EGG"]["T"]
        price = calculate_price("EGG", MARKET_I0 + 2 * T)
        self.assertEqual(price, 39, f"Egg P(I0+2T): expected $39, got ${price}")

    def test_milk_scarcity(self):
        """Milk at I0-T should be $256."""
        T = MARKET_PARAMS["MILK"]["T"]
        price = calculate_price("MILK", MARKET_I0 - T)
        self.assertEqual(price, 256, f"Milk P(I0-T): expected $256, got ${price}")

    def test_milk_glut(self):
        """Milk at I0+T should be $1."""
        T = MARKET_PARAMS["MILK"]["T"]
        price = calculate_price("MILK", MARKET_I0 + T)
        self.assertEqual(price, 1, f"Milk P(I0+T): expected $1, got ${price}")

    def test_wool_scarcity(self):
        """Wool at I0-T should be $240."""
        T = MARKET_PARAMS["WOOL"]["T"]
        price = calculate_price("WOOL", MARKET_I0 - T)
        self.assertEqual(price, 240, f"Wool P(I0-T): expected $240, got ${price}")

    def test_wool_glut(self):
        """Wool at I0+T should be $1."""
        T = MARKET_PARAMS["WOOL"]["T"]
        price = calculate_price("WOOL", MARKET_I0 + T)
        self.assertEqual(price, 1, f"Wool P(I0+T): expected $1, got ${price}")

    def test_fertilizer_scarcity(self):
        """Fertilizer at I0-T should be $140."""
        T = MARKET_PARAMS["FERTILIZER"]["T"]
        price = calculate_price("FERTILIZER", MARKET_I0 - T)
        self.assertEqual(price, 140, f"Fertilizer P(I0-T): expected $140, got ${price}")

    def test_fertilizer_glut(self):
        """Fertilizer at I0+T should be $60."""
        T = MARKET_PARAMS["FERTILIZER"]["T"]
        price = calculate_price("FERTILIZER", MARKET_I0 + T)
        self.assertEqual(price, 60, f"Fertilizer P(I0+T): expected $60, got ${price}")

    def test_fertilizer_glut_2T(self):
        """Fertilizer at I0+2T should be $20."""
        T = MARKET_PARAMS["FERTILIZER"]["T"]
        price = calculate_price("FERTILIZER", MARKET_I0 + 2 * T)
        self.assertEqual(price, 20, f"Fertilizer P(I0+2T): expected $20, got ${price}")

    def test_price_floor(self):
        """Price should never go below $1."""
        for product in MARKET_PARAMS:
            price = calculate_price(product, MARKET_I0 + 100000)
            self.assertGreaterEqual(price, 1, f"{product} price went below $1")


class TestSellRevenuePrediction(unittest.TestCase):
    """Test sell revenue prediction."""

    def test_single_unit_sell(self):
        """Selling 1 unit should give exactly the current price."""
        revenue, prices = predict_sell_revenue("WHEAT", 1, MARKET_I0)
        self.assertEqual(revenue, 25)
        self.assertEqual(len(prices), 1)

    def test_multi_unit_sell_decreasing_price(self):
        """Selling multiple units should show price potentially decreasing."""
        revenue, prices = predict_sell_revenue("WHEAT", 10, MARKET_I0)
        self.assertEqual(len(prices), 10)
        # First price should be 25 (at I0)
        self.assertEqual(prices[0], 25)
        # Total revenue should be positive
        self.assertGreater(revenue, 0)

    def test_zero_quantity(self):
        """Selling 0 units should give 0 revenue."""
        revenue, prices = predict_sell_revenue("WHEAT", 0, MARKET_I0)
        self.assertEqual(revenue, 0)
        self.assertEqual(len(prices), 0)


class TestBuyCostPrediction(unittest.TestCase):
    """Test buy cost prediction."""

    def test_single_unit_buy(self):
        """Buying 1 unit should cost the post-buy price."""
        cost, prices = predict_buy_cost("WHEAT", 1, MARKET_I0)
        self.assertEqual(len(prices), 1)
        self.assertGreater(cost, 0)

    def test_zero_quantity(self):
        """Buying 0 units should cost 0."""
        cost, prices = predict_buy_cost("WHEAT", 0, MARKET_I0)
        self.assertEqual(cost, 0)
        self.assertEqual(len(prices), 0)


if __name__ == "__main__":
    unittest.main()
