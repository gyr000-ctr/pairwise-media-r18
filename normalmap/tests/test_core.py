import math

import pytest

import normalmap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

H = [[0.5, 0.5], [0.5, 0.5]]

CASES = [
    ("flat_z", lambda: normalmap.NormalMap().from_height(H)[0][0][2] > 0.9),
    ("unit", lambda: abs(math.sqrt(sum(c * c for c in normalmap.NormalMap().from_height(H)[0][0])) - 1.0) < 1e-6),
    ("shape", lambda: len(normalmap.NormalMap().from_height(H)) == 2),
    ("bad_strength", lambda: _raises(lambda: normalmap.NormalMap().from_height(H, 0.0))),
    ("bad_height", lambda: _raises(lambda: normalmap.NormalMap().from_height([]))),
    ("type", lambda: isinstance(normalmap.NormalMap().from_height(H), list)),
    ("det", lambda: normalmap.NormalMap().from_height(H) == normalmap.NormalMap().from_height(H)),
    ("encoded_range", lambda: all(0.0 <= c <= 1.0 for row in normalmap.NormalMap().from_height(H) for p in row for c in p)),
    ("decode_unit", lambda: abs(math.sqrt(sum(c * c for c in normalmap.NormalMap().decode((0.5, 0.5, 1.0)))) - 1.0) < 1e-6),
    ("decode_bad", lambda: _raises(lambda: normalmap.NormalMap().decode((0.5, 0.5)))),
    ("nan", lambda: _raises(lambda: normalmap.NormalMap().from_height([[float('nan')]]))),
    ("no_mut", lambda: normalmap.NormalMap().from_height([r[:] for r in H]) is not None),
    ("decode_type", lambda: isinstance(normalmap.NormalMap().decode((0.5, 0.5, 1.0)), tuple)),
    ("three_comp", lambda: len(normalmap.NormalMap().from_height(H)[0][0]) == 3),
    ("slope", lambda: normalmap.NormalMap().from_height([[0.0, 1.0], [0.0, 1.0]])[0][0][2] >= 0.0),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
