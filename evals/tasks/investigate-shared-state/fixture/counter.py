class Batch:
    """Collects labeled counts across calls."""

    def __init__(self, seen=[]):  # bug: mutable default shared across every Batch()
        self.seen = seen

    def record(self, label):
        self.seen.append(label)

    def count(self, label):
        return self.seen.count(label)
