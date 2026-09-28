from register import Roster


def test_each_person_gets_their_own_tags():
    r = Roster()
    r.add("alice", ["admin"])
    r.add("bob")
    assert r.people["bob"] == ["member"]
    assert r.people["alice"] == ["admin", "member"]


def test_default_tags_do_not_leak_between_people():
    r = Roster()
    r.add("carol")
    r.add("dave")
    assert r.people["carol"] == ["member"]
    assert r.people["dave"] == ["member"]
