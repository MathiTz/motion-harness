import pytest
from cart import Cart


def test_valid_item_still_works():
    c = Cart()
    c.add_item("widget", 2.5, 3)
    assert c.total() == 7.5


def test_negative_price_raises():
    c = Cart()
    with pytest.raises(ValueError):
        c.add_item("widget", -1, 1)


def test_zero_or_negative_qty_raises():
    c = Cart()
    with pytest.raises(ValueError):
        c.add_item("widget", 1, 0)
    with pytest.raises(ValueError):
        c.add_item("widget", 1, -2)


def test_non_integer_qty_raises():
    c = Cart()
    with pytest.raises(ValueError):
        c.add_item("widget", 1, 1.5)
