import math

import pytest

import texatlas

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _a():
    return texatlas.TexAtlas(64, 1)

CASES = [
    ("uv_range", lambda: all(0.0 <= c <= 1.0 for c in _a().add('a', 8, 8) and _a().uv('a'))),
    ("bad_pad", lambda: _raises(lambda: texatlas.TexAtlas(64, 0))),
    ("bad_size", lambda: _raises(lambda: texatlas.TexAtlas(0, 1))),
    ("bad_name", lambda: _raises(lambda: _a().uv('zz'))),
    ("type", lambda: isinstance(_a().add('a', 8, 8) and _a().uv('a'), tuple)),
    ("det", lambda: _a().add('a', 8, 8) and _a().uv('a') == _a().uv('a')),
    ("mip_shrink", lambda: _a().add('a', 16, 16) and (_a().uv('a', 1)[2] <= _a().uv('a', 0)[2])),
    ("too_big", lambda: _raises(lambda: _a().add('b', 100, 100))),
    ("dup_name", lambda: _raises(lambda: (_a().add('c', 4, 4), _a().add('c', 4, 4)))),
    ("uv_len", lambda: _a().add('a', 8, 8) and len(_a().uv('a')) == 4),
    ("uv_pair", lambda: _a().add('a', 8, 8) and _a().uv('a')[1] >= 0.0),
    ("no_mut", lambda: _a().add('a', 8, 8) is not None),
    ("many", lambda: len([_a().add('n%d' % i, 4, 4) for i in range(4)]) == 4),
    ("mip_bad", lambda: _a().add('a', 8, 8) and _raises(lambda: _a().uv('a', 99))),
    ("uv_float", lambda: _a().add('a', 8, 8) and isinstance(_a().uv('a')[0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
