from typing import Any, Sequence

TSlices = tuple[slice | int, ...]


def parse(s: str) -> TSlices:
    """
    Interpret slice strings formatted similarly to Python's slicing syntax.

    >>> parse("")
    ()
    >>> parse(":,::,1:,-1:,2::,-2:")
    (slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(-1, None, None), slice(2, None, None), slice(-2, None, None))
    >>> parse(":1,:-1,:2:,:-2:")
    (slice(None, 1, None), slice(None, -1, None), slice(None, 2, None), slice(None, -2, None))
    >>> parse("::1,::-1")
    (slice(None, None, 1), slice(None, None, -1))
    >>> parse("1:2,1:2:,3::4,:5:6")
    (slice(1, 2, None), slice(1, 2, None), slice(3, None, 4), slice(None, 5, 6))
    >>> parse("1:2:3,4,-4")
    (slice(1, 2, 3), 4, -4)
    >>> parse(": , :5, -10:")
    (slice(None, None, None), slice(None, 5, None), slice(-10, None, None))
    """
    s = "".join(s.split())
    if any(c not in "0123456789:,-" for c in s):
        raise ValueError(f"Invalid characters in slice string: {s}")
    return tuple(
        slice(*(int(num) if num else None for num in part.split(":"))) if ":" in part else int(part)
        for part in s.split(",")
        if s
    )


def format(slices: TSlices) -> str:
    """
    Formats a tuple containing slice objects and integers back into a simplified string
    representation similar to Python's slicing syntax.

    >>> format(())
    ''
    >>> format((slice(None, None, None), slice(1, None, None), slice(-1, None, None)))
    ':,1:,-1:'
    >>> format((slice(None, 1, None), slice(None, -1, None)))
    ':1,:-1'
    >>> format((slice(None, None, 1), slice(None, None, -1)))
    '::1,::-1'
    >>> format((slice(1, 2, None), slice(3, None, 4), slice(None, 5, 6)))
    '1:2,3::4,:5:6'
    >>> format((slice(1, 2, 3), 4, -4))
    '1:2:3,4,-4'
    >>> format((slice(None, None, None), slice(None, 5, None), slice(-10, None, None)))
    ':,:5,-10:'
    """
    results = []
    for s in slices:
        if isinstance(s, slice):
            if s.step is not None:
                x = ":".join("" if x is None else str(x) for x in (s.start, s.stop, s.step))
            else:
                x = ":".join("" if x is None else str(x) for x in (s.start, s.stop))
        else:
            x = str(s)
        results.append(x)
    return ",".join(results)


def apply(obj: Sequence[Any], slices: TSlices) -> tuple[Any]:
    """
    Applies a series of slices and integer indices to a sequence object

    >>> apply(range(10), (slice(2, 5, None), slice(-3, None, None)))
    (2, 3, 4, 7, 8, 9)
    """
    results = []
    for s in slices:
        if isinstance(s, slice):
            results.extend(obj[s])
        else:
            results.append(obj[s])
    return tuple(results)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
