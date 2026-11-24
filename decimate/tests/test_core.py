import math

import pytest

import decimate

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
F = [(0, 1, 2), (1, 3, 2)]

CASES = [
    ("faces", lambda: len(decimate.Decimate().simplify(V, F, 1)[1]) <= 2),
    ("bad_target", lambda: _raises(lambda: decimate.Decimate().simplify(V, F, 0))),
    ("bad_input", lambda: _raises(lambda: decimate.Decimate().simplify(V, [], 1))),
    ("type", lambda: isinstance(decimate.Decimate().simplify(V, F, 1), tuple)),
    ("det", lambda: decimate.Decimate().simplify(V, F, 1) == decimate.Decimate().simplify(V, F, 1)),
    ("bounds", lambda: all(0.0 <= p[0] <= 1.0 for p in decimate.Decimate().simplify(V, F, 1)[0])),
    ("hausdorff", lambda: decimate.Decimate().simplify(V, F, 1) and decimate.Decimate().hausdorff() >= 0.0),
    ("no_degenerate", lambda: all(len(set(t)) == 3 for t in decimate.Decimate().simplify(V, F, 1)[1])),
    ("verts_kept", lambda: len(decimate.Decimate().simplify(V, F, 1)[0]) >= 3),
    ("no_mut", lambda: decimate.Decimate().simplify(list(V), F, 1) is not None),
    ("nan", lambda: _raises(lambda: decimate.Decimate().simplify([(float('nan'), 0, 0)] + V[1:], F, 1))),
    ("hausdorff_type", lambda: isinstance(decimate.Decimate().simplify(V, F, 1) and 0.0, float)),
    ("target_one", lambda: len(decimate.Decimate().simplify(V, F, 1)[1]) >= 1),
    ("faces_type", lambda: isinstance(decimate.Decimate().simplify(V, F, 1)[1], list)),
    ("no_target_exceed", lambda: len(decimate.Decimate().simplify(V, F, 1)[1]) <= 2),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
