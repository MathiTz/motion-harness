import math

from shapes import Circle, Square, Triangle


def test_circle_area():
    assert math.isclose(Circle(2).area(), math.pi * 4, rel_tol=1e-6)


def test_square_area():
    assert Square(3).area() == 9


def test_triangle_area():
    assert math.isclose(Triangle(4, 5).area(), 10.0, rel_tol=1e-6)
