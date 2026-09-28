def get_last_n(items, n):
    """Return the last n items of items, in original order."""
    if n <= 0:
        return []
    return items[-n - 1:]
