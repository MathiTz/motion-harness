from counter import Batch


def test_first_batch_counts_its_own_label():
    b = Batch()
    b.record("a")
    b.record("a")
    assert b.count("a") == 2


def test_second_batch_starts_empty():
    b = Batch()
    assert b.count("a") == 0  # fails when run after the first test: seen[] was shared
