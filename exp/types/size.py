__all__ = ["Size2d", "size_2d_to_tuple"]

type Size2d = int | tuple[int, int]  # PEP 695 (Python 3.13)


def size_2d_to_tuple(size: Size2d) -> tuple[int, int]:
    """整数なら正方 (size, size)、tuple はそのまま (h, w) で返す。"""
    if isinstance(size, int):
        return (size, size)
    return size
