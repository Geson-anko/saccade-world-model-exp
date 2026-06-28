"""Behaviour spec for ``exp.types.size``.

Pins the contract of ``size_2d_to_tuple``: an int becomes a square, and
a tuple passes through preserving (h, w) order. These are unit tests
over a pure function (no torch, no I/O).
"""

from exp.types.size import size_2d_to_tuple


def test_int_becomes_square():
    # A scalar size means a square: (n, n).
    assert size_2d_to_tuple(4) == (4, 4)


def test_tuple_passthrough_preserves_hw_order():
    # A non-symmetric tuple pins (h, w) order so a swap would be caught.
    assert size_2d_to_tuple((2, 5)) == (2, 5)
