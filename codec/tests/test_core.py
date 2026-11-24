import math

import pytest

import codec

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.0, 0.1, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1]

CASES = [
    ("roundtrip_len", lambda: len(codec.Adpcm().decode(codec.Adpcm().encode(S), len(S))) == len(S)),
    ("bits_per_sample", lambda: len(codec.Adpcm().encode(S)) == 4),
    ("first_sample", lambda: abs(codec.Adpcm().decode(codec.Adpcm().encode(S), len(S))[0] - S[0]) < 1e-9),
    ("bad_empty", lambda: _raises(lambda: codec.Adpcm().encode([]))),
    ("bad_data", lambda: _raises(lambda: codec.Adpcm().decode([], 4))),
    ("bad_n", lambda: _raises(lambda: codec.Adpcm().decode(codec.Adpcm().encode(S), 0))),
    ("type", lambda: isinstance(codec.Adpcm().encode(S), list)),
    ("det", lambda: codec.Adpcm().encode(S) == codec.Adpcm().encode(S)),
    ("range", lambda: all(abs(v) <= 1.0 for v in codec.Adpcm().decode(codec.Adpcm().encode(S), len(S)))),
    ("nan", lambda: _raises(lambda: codec.Adpcm().encode([float('nan')]))),
    ("byte_range", lambda: all(0 <= b <= 255 for b in codec.Adpcm().encode(S))),
    ("no_mut", lambda: codec.Adpcm().encode(list(S)) is not None),
    ("decode_type", lambda: isinstance(codec.Adpcm().decode(codec.Adpcm().encode(S), len(S)), list)),
    ("long", lambda: len(codec.Adpcm().decode(codec.Adpcm().encode([0.1] * 1000), 1000)) == 1000),
    ("size_ratio", lambda: len(codec.Adpcm().encode([0.1] * 1000)) == 500),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
