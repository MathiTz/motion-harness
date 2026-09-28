from limiter import RateLimiter


def test_allows_up_to_max_calls_within_window():
    rl = RateLimiter(max_calls=3, window_seconds=10)
    assert rl.allow(0.0) is True
    assert rl.allow(1.0) is True
    assert rl.allow(2.0) is True
    assert rl.allow(3.0) is False  # 4th call within the window


def test_allows_again_once_old_calls_fall_out_of_the_window():
    rl = RateLimiter(max_calls=2, window_seconds=10)
    assert rl.allow(0.0) is True
    assert rl.allow(1.0) is True
    assert rl.allow(2.0) is False
    assert rl.allow(11.5) is True  # the call at t=0 is now outside the 10s window
