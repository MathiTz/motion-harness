from utils import get_last_n


def test_returns_exactly_n_items():
    assert get_last_n([1, 2, 3, 4, 5], 2) == [4, 5]


def test_n_larger_than_list_returns_whole_list():
    assert get_last_n([1, 2], 5) == [1, 2]


def test_n_zero_returns_empty():
    assert get_last_n([1, 2, 3], 0) == []


def test_preserves_order():
    assert get_last_n(["a", "b", "c", "d"], 3) == ["b", "c", "d"]
