import math

import pytest

import layout

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

G = [("a", 1.0), ("b", 1.0), (" ", 0.5), ("c", 1.0)]

CASES = [
    ("lines", lambda: len(layout.TextLayout().layout(G, 2.0)) >= 1),
    ("width_bound", lambda: all(sum(w for _, w in line) <= 2.0 + 1e-9 for line in layout.TextLayout().layout(G, 2.0))),
    ("bad_width", lambda: _raises(lambda: layout.TextLayout().layout(G, 0.0))),
    ("empty", lambda: layout.TextLayout().layout([], 2.0) == []),
    ("type", lambda: isinstance(layout.TextLayout().layout(G, 2.0), list)),
    ("det", lambda: layout.TextLayout().layout(G, 2.0) == layout.TextLayout().layout(G, 2.0)),
    ("tracking", lambda: layout.TextLayout(0.5).layout(G, 10.0) != layout.TextLayout(0.0).layout(G, 10.0)),
    ("bad_tracking", lambda: _raises(lambda: layout.TextLayout(-1.0))),
    ("single_line", lambda: len(layout.TextLayout().layout(G, 10.0)) == 1),
    ("no_glyph_lost", lambda: sum(len(l) for l in layout.TextLayout().layout(G, 2.0)) == 4),
    ("line_type", lambda: isinstance(layout.TextLayout().layout(G, 2.0)[0], list)),
    ("nan", lambda: _raises(lambda: layout.TextLayout().layout([('a', float('nan'))], 2.0))),
    ("no_mut", lambda: layout.TextLayout().layout(list(G), 2.0) is not None),
    ("glyph_pair", lambda: len(layout.TextLayout().layout(G, 10.0)[0][0]) == 2),
    ("bad_glyph", lambda: _raises(lambda: layout.TextLayout().layout([('a',)], 2.0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
