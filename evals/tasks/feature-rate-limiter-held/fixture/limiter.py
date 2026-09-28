class RateLimiter:
    def __init__(self, max_calls, window_seconds):
        self.max_calls = max_calls
        self.window_seconds = window_seconds

    def allow(self, now):
        return True
