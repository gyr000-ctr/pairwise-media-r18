"""扩展约束的独立回归（仅用标准库编写，任何环境都可收集运行）。

这些用例把题面中的量化指标逐条固化：
Sobel/中心差分 ≤1e-6、TBN 正交归一 ≤1e-6、雅可比球面 ≤2°、
mip 逐级重归一、法线重建高度相对误差 ≤5%、编码往返 ≤1/255、
非法强度有可区分的异常类型，以及并发调用安全。
"""

import math
import threading

import normalmap


NM = normalmap.NormalMap()


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _linear_ramp(h=6, w=9, ax=0.37, by=-0.51, c=1.25):
    return [[ax * x + by * y + c for x in range(w)] for y in range(h)]


def _max(a, b):
    return max(abs(a[i] - b[i]) for i in range(len(a)))


def _mean_angle(nmap, amap):
    total = 0.0
    count = 0
    for row_n, row_a in zip(nmap, amap):
        for n, a in zip(row_n, row_a):
            cosv = (n[0] * a[0] + n[1] * a[1] + n[2] * a[2])
            ln = math.sqrt(sum(c * c for c in n))
            la = math.sqrt(sum(c * c for c in a))
            cosv = max(-1.0, min(1.0, cosv / (ln * la)))
            total += math.degrees(math.acos(cosv))
            count += 1
    return total / count


def _transformed_sphere(size, radius, j00, j01, j10, j11):
    c = (size - 1) / 2.0
    height = [[0.0] * size for _ in range(size)]
    analytic = [[None] * size for _ in range(size)]
    for row in range(size):
        for col in range(size):
            u = col - c
            v = row - c
            x = j00 * u + j01 * v
            y = j10 * u + j11 * v
            z2 = radius * radius - x * x - y * y
            z = math.sqrt(z2) if z2 > 0.0 else 1e-6
            height[row][col] = z
            norm = math.sqrt(x * x + y * y + z * z)
            analytic[row][col] = [x / norm, y / norm, z / norm]
    return height, analytic


def _relative_error(original, reconstructed):
    h = len(original)
    w = len(original[0])
    mo = sum(sum(r) for r in original) / (h * w)
    mr = sum(sum(r) for r in reconstructed) / (h * w)
    oc = [[original[y][x] - mo for x in range(w)] for y in range(h)]
    rc = [[reconstructed[y][x] - mr for x in range(w)] for y in range(h)]
    num = sum(oc[y][x] * rc[y][x] for y in range(h) for x in range(w))
    den = sum(v * v for r in rc for v in r)
    scale = num / den if den > 1e-30 else 1.0
    err2 = sum((oc[y][x] - scale * rc[y][x]) ** 2
               for y in range(h) for x in range(w))
    norm2 = sum(v * v for r in oc for v in r)
    return math.sqrt(err2 / norm2)


# ---------------------------------------------------------------------------
# 1. Sobel 与中心差分对照
# ---------------------------------------------------------------------------
def test_sobel_matches_central_on_linear_field():
    hmap = _linear_ramp()
    sx, sy = normalmap.sobel(hmap)
    cx, cy = normalmap.central_difference(hmap)
    worst = 0.0
    for r1, r2 in zip(sx, cx):
        worst = max(worst, _max(r1, r2))
    for r1, r2 in zip(sy, cy):
        worst = max(worst, _max(r1, r2))
    assert worst <= 1e-6


def test_sobel_x_unit_slope():
    # h = 2x 时归一化 Sobel gx 应恰为 2
    hmap = [[2.0 * x for x in range(7)] for _ in range(5)]
    gx = normalmap.sobel_x(hmap)
    assert all(abs(gx[r][c] - 2.0) < 1e-9 for r in range(1, 4) for c in range(1, 6))


# ---------------------------------------------------------------------------
# 2. 单位化与切线空间正交归一
# ---------------------------------------------------------------------------
def test_flat_is_positive_z_and_unit():
    out = NM.from_height([[0.5, 0.5], [0.5, 0.5]])
    for row in out:
        for x, y, z in row:
            assert z > 0.9
            assert abs(math.sqrt(x * x + y * y + z * z) - 1.0) < 1e-9


def test_all_normals_unit_on_slopes():
    hmap = _linear_ramp(12, 16)
    out = NM.from_height(hmap, 2.0)
    for row in out:
        for x, y, z in row:
            assert abs(math.sqrt(x * x + y * y + z * z) - 1.0) < 1e-9


def test_tangent_basis_orthonormal():
    for n in [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.3, -0.7, 0.648074)]:
        t, b, nv = normalmap.orthonormal_basis(n)
        basis = (t, b, nv)
        for i in range(3):
            assert abs(math.sqrt(sum(c * c for c in basis[i])) - 1.0) < 1e-9
            for j in range(i):
                dot = sum(basis[i][k] * basis[j][k] for k in range(3))
                assert abs(dot) < 1e-9


def test_frame_from_gradients_orthonormal():
    hmap = _linear_ramp(8, 8)
    gx, gy = normalmap.sobel(hmap)
    tmap, bmap, nmap = normalmap.frame_from_gradients(gx, gy)
    for y in range(8):
        for x in range(8):
            t, b, n = tmap[y][x], bmap[y][x], nmap[y][x]
            assert abs(sum(t[k] * b[k] for k in range(3))) < 1e-6
            assert abs(sum(t[k] * n[k] for k in range(3))) < 1e-6
            assert abs(sum(b[k] * n[k] for k in range(3))) < 1e-6
            for v in (t, b, n):
                assert abs(math.sqrt(sum(c * c for c in v)) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 3. UV 旋转 / 缩放的雅可比修正（解析球面 ≤2°）
# ---------------------------------------------------------------------------
def test_jacobian_correction_matches_sphere():
    size = 40
    radius = 10.0
    step = 2.4 / (size - 1)
    for theta, kx, ky in [(0.0, 1.0, 1.0), (0.6, 2.0, 2.0),
                          (-0.9, 1.6, 0.7)]:
        c, s = math.cos(theta), math.sin(theta)
        dxx, dxy = c * step * kx, -s * step * ky
        dyx, dyy = s * step * kx, c * step * ky
        height, analytic = _transformed_sphere(
            size, radius, dxx, dxy, dyx, dyy)
        j = ((dxx, dxy), (dyx, dyy))
        nmap = NM.from_height(height, 1.0, method="central", jacobian=j)
        assert _mean_angle(nmap, analytic) < 2.0


def test_missing_jacobian_on_scaled_grid_is_worse():
    size = 40
    radius = 10.0
    sx, sy = 2.0 * 2.4 / (size - 1), 3.0 * 2.4 / (size - 1)
    height, analytic = _transformed_sphere(size, radius, sx, 0.0, 0.0, sy)
    corrected = NM.from_height(height, 1.0, method="central",
                               jacobian=((sx, 0.0), (0.0, sy)))
    raw = NM.from_height(height, 1.0, method="central")
    good = _mean_angle(corrected, analytic)
    bad = _mean_angle(raw, analytic)
    assert good < 2.0
    assert bad > good + 3.0


# ---------------------------------------------------------------------------
# 4. mip 链逐级重归一
# ---------------------------------------------------------------------------
def test_mip_chain_every_level_unit_and_descending():
    hmap = _linear_ramp(32, 32)
    nmap = NM.from_height(hmap)
    chain = normalmap.mip_chain(nmap)
    assert len(chain[0]) == 32 and len(chain[0][0]) == 32
    assert (len(chain[-1]), len(chain[-1][0])) == (1, 1)
    for level in chain:
        for row in level:
            for x, y, z in row:
                assert abs(math.sqrt(x * x + y * y + z * z) - 1.0) < 1e-9


def test_mip_level_shapes():
    nmap = NM.from_height(_linear_ramp(16, 16))
    assert len(normalmap.mip_level(nmap, 2)) == 4


# ---------------------------------------------------------------------------
# 5. 由法线重建高度
# ---------------------------------------------------------------------------
def _smooth_height(n=24):
    return [[0.7 * math.sin(x / 4.0) + 0.5 * math.cos(y / 5.0)
             + 0.2 * math.sin((x + y) / 7.0)
             for x in range(n)] for y in range(n)]


def test_reconstruct_height_poisson():
    hmap = _smooth_height()
    nmap = NM.from_height(hmap, 1.0, method="central")
    rec = NM.reconstruct_height(nmap, 1.0, method="poisson")
    assert _relative_error(hmap, rec) < 0.05


def test_reconstruct_height_rows():
    hmap = _smooth_height()
    nmap = NM.from_height(hmap, 1.0, method="central")
    rec = normalmap.integrate_rows(nmap, 1.0)
    assert _relative_error(hmap, rec) < 0.05


def test_bidirectional_roundtrip_preserves_shape():
    hmap = _smooth_height(12)
    nmap = normalmap.height_to_normals(hmap)
    back = normalmap.normal_to_height(nmap)
    assert len(back) == 12 and len(back[0]) == 12


# ---------------------------------------------------------------------------
# 6. 编码 [0,1] 往返
# ---------------------------------------------------------------------------
def test_encode_values_in_unit_interval():
    rgb = normalmap.encode((0.0, 0.0, 1.0))
    assert rgb == (0.5, 0.5, 1.0)
    assert all(0.0 <= c <= 1.0 for c in rgb)


def test_quantized_roundtrip_within_one_255():
    for n in [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
              (0.577350269, -0.577350269, 0.577350269)]:
        rgb = normalmap.encode(n)
        quantized = tuple(round(c * 255.0) / 255.0 for c in rgb)
        assert max(abs(a - b) for a, b in zip(rgb, quantized)) <= 1.0 / 255 + 1e-12


def test_byte_roundtrip():
    for n in [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0)]:
        b8 = normalmap.encode8(n)
        back = normalmap.decode8(b8)
        assert all(abs(a - b) <= 1.0 / 255 + 1e-9 for a, b in zip(n, back))


def test_decode_is_unit_tuple():
    out = normalmap.decode((0.5, 0.5, 1.0))
    assert isinstance(out, tuple)
    assert abs(math.sqrt(sum(c * c for c in out)) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 7. 非法强度可区分报错 + 非有限输入分类
# ---------------------------------------------------------------------------
def test_invalid_strength_type_distinct():
    for bad in (0.0, -0.5, float("nan"), float("inf"), "strong", None, True):
        try:
            NM.from_height([[0.1, 0.2]], bad)
        except normalmap.InvalidStrengthError:
            pass
        else:
            raise AssertionError("非法强度未报错: %r" % (bad,))
    assert issubclass(normalmap.InvalidStrengthError, ValueError)
    assert normalmap.InvalidStrengthError is not normalmap.InvalidHeightError


def test_nonfinite_input_classified():
    for bad in ([[float("nan")]], [[float("inf")]], [[-float("inf")]]):
        try:
            NM.from_height(bad)
        except normalmap.NonFiniteError:
            pass
        else:
            raise AssertionError("非有限输入未报 NonFiniteError")


def test_bad_height_is_value_error():
    for bad in ([], [[]], [[1.0, 2.0], [3.0]], "not a map"):
        try:
            NM.from_height(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("非法高度图未报错: %r" % (bad,))


# ---------------------------------------------------------------------------
# 8. 并发安全：共享实例与独立实例结果一致
# ---------------------------------------------------------------------------
def test_concurrent_calls_safe():
    hmap = _smooth_height(20)
    expected = NM.from_height(hmap)
    errors = []

    def worker():
        try:
            for _ in range(20):
                assert NM.from_height(hmap) == expected
                assert NM.mip_chain(expected)[1] == normalmap.mip_level(expected, 1)
        except Exception as exc:  # pragma: no cover - 并发诊断
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
