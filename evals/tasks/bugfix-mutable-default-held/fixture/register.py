class Roster:
    def __init__(self):
        self.people = {}

    def add(self, name, tags=[]):
        tags.append("member")
        self.people[name] = tags
