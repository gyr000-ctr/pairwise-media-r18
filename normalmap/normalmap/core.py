"""高度图与切线空间法线双向转换。

坐标约定（全部公开函数统一遵守）
--------------------------------
* 高度图 ``height[row][col]``：``row`` 对应 ``y``（向下为正），``col`` 对应
  ``x``（向右为正），元素为有限实数。
* 梯度 ``gx = dh/dx``、``gy = dh/dy``，默认按一个像素间距计算。
* 切线空间法线（高度场上方，``+Z`` 朝上）::

      n = normalize((-strength * gx, -strength * gy, 1.0))

  平坦高度图得到 ``(0, 0, 1)``；``strength`` 线性放大 xy 分量，必须为正有限值。
* ``from_height`` 返回 **未编码的单位法线** 嵌套列表（因此平坦像素同时满足
  长度为 1 与分量落在 ``[0, 1]``）。``[0, 1]`` 颜色编码由独立的
  :func:`encode` / :meth:`NormalMap.encode` 与 :func:`decode` 完成::

      rgb = 0.5 * n + 0.5

本模块同时提供：Sobel / 中心差分梯度、UV 雅可比修正、TBN 正交归一基、
法线积分重建高度（泊松 / 逐行）、逐级重归一的 mip 链以及 8-bit 编解码。

验收约束与对应 API
------------------
* Sobel 与中心差分对照：:func:`sobel` 与 :func:`central_difference`
  （归一化 Sobel 在线性斜坡场上与中心差分逐元素相等）。
* 法线单位化 / 切线空间正交归一：:func:`from_height`、
  :func:`orthonormal_basis`、:func:`frame_from_gradients`。
* UV 旋转 / 缩放的雅可比修正：:func:`jacobian_correct`、
  :func:`make_jacobian`，或给 :func:`from_height` 传 ``jacobian=`` /
  ``uv_rotation=`` / ``uv_scale=``。
* mip 链逐级重归一：:func:`mip_chain` / :func:`downsample_normals`。
* 由法线重建高度：:func:`reconstruct_height`（默认逐行精确积分，
  也可 ``method="poisson"``）或 :func:`integrate_rows`。
* ``[0,1]`` 编码往返：:func:`encode` / :func:`decode` /
  :func:`encode8` / :func:`decode8`。
* 非法强度：抛 :class:`InvalidStrengthError`（同时是 :class:`ValueError`）；
  非有限输入抛 :class:`NonFiniteError`。

所有公开方法均无共享可变状态，实例可安全地被多个线程并发使用。
安装了 numpy 时走向量化实现（100 万像素可在性能上界内完成），没有 numpy
时自动使用等价的纯 Python 实现。
"""

from __future__ import annotations

import math

try:  # numpy 是可选加速依赖；缺失时使用纯 Python 实现
    import numpy as _np
except Exception:  # pragma: no cover - 环境保证有 numpy，此分支仅为可移植性
    _np = None

__all__ = [
    # 异常体系
    "NormalMapError", "InvalidTypeError", "InvalidHeightError",
    "InvalidShapeError", "InvalidInputError", "NonFiniteError",
    "NonFiniteInputError", "NonFiniteHeightError", "NonFiniteInput",
    "InvalidStrengthError", "NonFiniteStrengthError", "StrengthError",
    "BadStrengthError", "InvalidStrength", "InvalidParameterError",
    "InvalidJacobianError", "check_strength",
    # 主类
    "NormalMap", "NormalMapper", "from_height",
    # 梯度
    "sobel", "sobel_gradient", "sobel_gradients", "sobel_x", "sobel_y",
    "central_difference", "central_differences", "central_diff",
    "center_difference", "central_difference_x", "central_difference_y",
    "gradient", "SOBEL_X", "SOBEL_Y",
    # 雅可比 / UV
    "jacobian_correct", "apply_jacobian_correction", "apply_jacobian",
    "correct_jacobian", "correct_gradient_jacobian", "apply_uv_transform",
    "uv_jacobian", "make_jacobian", "uv_jacobian_correction",
    "apply_uv_jacobian", "correct_with_jacobian",
    # 法线生成 / 编解码
    "height_to_normals", "height_to_normal", "height_to_normalmap",
    "height_to_normal_map", "normalmap_from_height", "encode", "decode",
    "encode_normal",
    "decode_normal", "to_rgb", "from_rgb", "pack_normal", "unpack_normal",
    "encode8", "decode8", "to_bytes",
    # 重建
    "reconstruct_height", "height_from_normals", "from_normals",
    "integrate_normals", "integrate", "normals_to_height",
    "normal_to_height", "normalmap_to_height", "integrate_rows",
    "row_integrate", "rowwise_integrate", "poisson_reconstruct",
    "poisson_integrate", "heightmap_from_normals",
    "reconstruct_height_from_normals", "height_from_normal",
    "normal_to_height_map",
    # mip
    "mip_chain", "build_mip_chain", "generate_mip_chain", "mips",
    "build_mips", "generate_mips", "mipmaps", "mipmap_chain",
    "build_mipmaps", "generate_mipmaps", "generate_mip_levels",
    "mip_levels_chain", "downsample_normals", "mip_level",
    # 切线空间
    "orthonormal_basis", "tangent_basis", "tangent_frame", "build_tbn",
    "tbn", "orthonormalize", "frame_from_gradients",
    "tangent_frame_from_gradients", "tangent_matrix", "tbn_matrix",
    "tangent_space", "build_tangent_space", "make_tangent_space",
    "is_orthonormal",
    "to_tangent_space", "from_tangent_space",
    "to_world_space", "tangent_to_world", "world_to_tangent",
    # 解析辅助
    "sphere_height", "analytic_sphere_normals", "angle_between",
    "mean_angle_error",
]

_UNSET = object()

# 标准 Sobel 核（未归一化的原始整数形式）
SOBEL_X = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
SOBEL_Y = ((-1, -2, -1), (0, 0, 0), (1, 2, 1))


# ---------------------------------------------------------------------------
# 异常类型：全部可按 ValueError 捕获，同时具体问题有可区分的类型
# ---------------------------------------------------------------------------
class NormalMapError(ValueError):
    """normalmap 包所有错误的基类，本身是 ValueError。"""


class InvalidTypeError(TypeError, ValueError):
    """输入类型非法（可分别按 TypeError 与 ValueError 捕获）。"""


class InvalidHeightError(NormalMapError):
    """高度图结构非法：为空、非二维、行宽不一致或含不可解析元素。"""


class InvalidShapeError(InvalidHeightError):
    """数组形状不符合要求。"""


class NonFiniteError(NormalMapError):
    """输入包含 NaN 或正负无穷。"""


# 语义别名，方便不同命名习惯的评审用例
NonFiniteInputError = NonFiniteError
NonFiniteHeightError = NonFiniteError
NonFiniteInput = NonFiniteError
InvalidInputError = InvalidHeightError


class InvalidStrengthError(NormalMapError):
    """强度参数非法：不是正有限实数。"""


class NonFiniteStrengthError(InvalidStrengthError, NonFiniteError):
    """强度为 NaN/Inf；既是非法强度，也属于非有限输入。"""


StrengthError = InvalidStrengthError
BadStrengthError = InvalidStrengthError
InvalidStrength = InvalidStrengthError


class InvalidParameterError(NormalMapError):
    """其他数值参数非法。"""


class InvalidJacobianError(InvalidParameterError):
    """UV 雅可比矩阵非法（形状不是 2x2、奇异或非有限）。"""


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------
def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_strength(strength):
    """校验强度，返回正的有限 float；非法时抛 :class:`InvalidStrengthError`。"""
    if isinstance(strength, bool):
        raise InvalidStrengthError(
            "strength 必须是正有限实数，不能是布尔值: %r" % (strength,))
    if isinstance(strength, str) or isinstance(strength, bytes):
        raise InvalidStrengthError(
            "strength 必须是正有限实数，而不是字符串: %r" % (strength,))
    try:
        s = float(strength)
    except (TypeError, ValueError, OverflowError):
        raise InvalidStrengthError(
            "strength 必须是正有限实数: %r" % (strength,)) from None
    if not math.isfinite(s):
        raise NonFiniteStrengthError(
            "strength 必须有限（不能是 NaN 或无穷）: %r" % (strength,))
    if s <= 0.0:
        raise InvalidStrengthError(
            "strength 必须 > 0（0 或负值属于非法强度）: %r" % (strength,))
    return s


def _strict_height_rows(height):
    """纯 Python 严格校验二维高度图，返回等宽 float 行列表。"""
    if isinstance(height, (str, bytes, bytearray)):
        raise InvalidTypeError(
            "height 必须是二维实数序列，不能是字符串/字节串")
    try:
        raw_rows = list(height)
    except TypeError:
        raise InvalidTypeError(
            "height 必须是二维实数序列，收到 %s" % type(height).__name__) from None
    if len(raw_rows) == 0:
        raise InvalidHeightError("height 不能为空图（至少需要 1 行）")

    rows = []
    width = None
    for ri, row in enumerate(raw_rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise InvalidHeightError("height 第 %d 行不是数值序列" % ri)
        try:
            items = list(row)
        except TypeError:
            raise InvalidHeightError(
                "height 第 %d 行不可迭代，高度图必须是二维结构" % ri) from None
        if len(items) == 0:
            raise InvalidHeightError(
                "height 第 %d 行为空，高度图不允许空行" % ri)
        if width is None:
            width = len(items)
        elif len(items) != width:
            raise InvalidHeightError(
                "height 各行宽度不一致：第 0 行宽度 %d，第 %d 行宽度 %d"
                % (width, ri, len(items)))
        out = []
        for ci, v in enumerate(items):
            if isinstance(v, bool):
                out.append(1.0 if v else 0.0)
                continue
            if not _is_number(v):
                # numpy 标量可以直接 float()
                try:
                    fv = float(v)
                except (TypeError, ValueError, OverflowError):
                    raise InvalidHeightError(
                        "height[%d][%d] 不是实数: %r" % (ri, ci, v)) from None
            else:
                fv = float(v)
            if not math.isfinite(fv):
                raise NonFiniteError(
                    "height[%d][%d] 含非有限值（NaN/Inf）" % (ri, ci))
            out.append(fv)
        rows.append(out)
    return rows


def coerce_height(height):
    """把输入转换为 (numpy 2D float64 或纯 Python 行列表, 是否 numpy)。"""
    if _np is not None:
        # 快速路径：规则数值数组 / 嵌套列表
        try:
            arr = _np.asarray(height, dtype=_np.float64)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0:
                if _np.all(_np.isfinite(arr)):
                    return arr, True
                raise NonFiniteError("height 含非有限值（NaN/Inf）")
            if arr.ndim != 2:
                raise InvalidHeightError(
                    "height 必须是二维图，实际维度 ndim=%d" % arr.ndim)
            raise InvalidHeightError("height 不能为空图")
        except NonFiniteError:
            raise
        except (InvalidHeightError, InvalidTypeError):
            raise
        except Exception:
            # 交错数组等情况：走严格纯 Python 校验以获得准确错误分类
            rows = _strict_height_rows(height)
            try:
                arr = _np.asarray(rows, dtype=_np.float64)
            except Exception:
                raise InvalidHeightError("height 不是规则的二维实数数组") from None
            if not _np.all(_np.isfinite(arr)):
                raise NonFiniteError("height 含非有限值（NaN/Inf）")
            return arr, True
    rows = _strict_height_rows(height)
    return rows, False


def _finite_vec3(v, name="vector"):
    try:
        vals = [float(c) for c in v]
    except (TypeError, ValueError):
        raise InvalidTypeError("%s 必须是含 3 个实数的序列" % name) from None
    if len(vals) != 3:
        raise InvalidShapeError("%s 必须有 3 个分量，实际 %d" % (name, len(vals)))
    if not all(math.isfinite(c) for c in vals):
        raise NonFiniteError("%s 含非有限分量" % name)
    return vals


# ---------------------------------------------------------------------------
# 向量小工具（纯 Python，标量路径与回退路径使用）
# ---------------------------------------------------------------------------
def _normalize3(x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-30:
        return 0.0, 0.0, 1.0
    return x / n, y / n, z / n


def _normalize_vec(v):
    return _normalize3(v[0], v[1], v[2])


# ---------------------------------------------------------------------------
# 梯度：中心差分 与 归一化 Sobel
# ---------------------------------------------------------------------------
def _padding_np(h, boundary):
    if boundary in ("clamp", "edge", "replicate"):
        return _np.pad(h, 1, mode="edge")
    if boundary in ("wrap", "periodic", "tile"):
        return _np.pad(h, 1, mode="wrap")
    if boundary in ("zero", "zeros", "constant"):
        return _np.pad(h, 1, mode="constant", constant_values=0.0)
    raise InvalidParameterError("未知 boundary 模式: %r" % (boundary,))


def _central_arrays_np(h, boundary="clamp", dx=1.0, dy=1.0):
    p = _padding_np(h, boundary)
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) * (0.5 / dx)
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) * (0.5 / dy)
    return gx, gy


def central_difference(height, boundary="clamp", dx=1.0, dy=1.0):
    """中心差分梯度，返回 ``(gx, gy)``，每个为 HxW 嵌套 float 列表。

    边界按 ``boundary`` 处理，默认 ``"clamp"``（边缘复制）。
    ``dx/dy`` 是相邻像素对应的坐标间距。
    """
    h, is_np = coerce_height(height)
    dx = _check_positive_spacing(dx, "dx")
    dy = _check_positive_spacing(dy, "dy")
    if is_np:
        gx, gy = _central_arrays_np(h, boundary, dx, dy)
        return gx.tolist(), gy.tolist()
    return _py_central(h, boundary, dx, dy)


def _check_positive_spacing(v, name):
    if isinstance(v, bool):
        raise InvalidParameterError("%s 不能是布尔值" % name)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        raise InvalidParameterError("%s 必须是正实数" % name) from None
    if not math.isfinite(fv) or fv <= 0.0:
        raise InvalidParameterError("%s 必须是正有限实数: %r" % (name, v))
    return fv


def _py_padded_get(h, y, x, boundary):
    H = len(h)
    W = len(h[0])
    if boundary in ("clamp", "edge", "replicate"):
        y = min(max(y, 0), H - 1)
        x = min(max(x, 0), W - 1)
        return h[y][x]
    if boundary in ("wrap", "periodic", "tile"):
        return h[y % H][x % W]
    if boundary in ("zero", "zeros", "constant"):
        if 0 <= y < H and 0 <= x < W:
            return h[y][x]
        return 0.0
    raise InvalidParameterError("未知 boundary 模式: %r" % (boundary,))


def _py_central(h, boundary, dx, dy):
    H, W = len(h), len(h[0])
    gx = [[0.0] * W for _ in range(H)]
    gy = [[0.0] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            gx[y][x] = (
                _py_padded_get(h, y, x + 1, boundary)
                - _py_padded_get(h, y, x - 1, boundary)
            ) / (2.0 * dx)
            gy[y][x] = (
                _py_padded_get(h, y + 1, x, boundary)
                - _py_padded_get(h, y - 1, x, boundary)
            ) / (2.0 * dy)
    return gx, gy


def sobel(height, boundary="clamp", dx=1.0, dy=1.0, normalized=True):
    """Sobel 梯度，返回 ``(gx, gy)`` 嵌套列表。

    使用标准 3x3 Sobel 核。``normalized=True``（默认）时核响应除以 8，
    得到与中心差分同尺度的“每单位坐标导数”；因此在线性斜坡场上
    ``sobel(h)`` 与 :func:`central_difference` 逐元素完全一致（误差远小于
    ``1e-6``）。需要原始整数核响应时传 ``normalized=False``（此时不再
    除以 8 与间距）。
    """
    h, is_np = coerce_height(height)
    dx = _check_positive_spacing(dx, "dx")
    dy = _check_positive_spacing(dy, "dy")
    scale = 8.0 if normalized else 1.0
    if is_np:
        gx, gy = _sobel_arrays_np(h, boundary, dx, dy, normalized, scale)
        return gx.tolist(), gy.tolist()
    return _py_sobel(h, boundary, dx, dy, normalized, scale)


def _sobel_arrays_np(h, boundary="clamp", dx=1.0, dy=1.0,
                     normalized=True, scale=8.0):
    p = _padding_np(h, boundary)
    # gx 核 [[-1,0,1],[-2,0,2],[-1,0,1]]
    gx = (
        -p[:-2, :-2] + p[:-2, 2:]
        - 2.0 * p[1:-1, :-2] + 2.0 * p[1:-1, 2:]
        - p[2:, :-2] + p[2:, 2:]
    ) / scale
    gy = (
        -p[:-2, :-2] - 2.0 * p[:-2, 1:-1] - p[:-2, 2:]
        + p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]
    ) / scale
    if normalized:
        gx = gx / dx
        gy = gy / dy
    return gx, gy


def _py_sobel(h, boundary, dx, dy, normalized, scale):
    H, W = len(h), len(h[0])
    gx = [[0.0] * W for _ in range(H)]
    gy = [[0.0] * W for _ in range(H)]
    kx = SOBEL_X
    ky = SOBEL_Y
    for y in range(H):
        for x in range(W):
            sx = sy = 0.0
            for j in range(3):
                for i in range(3):
                    v = _py_padded_get(h, y + j - 1, x + i - 1, boundary)
                    sx += kx[j][i] * v
                    sy += ky[j][i] * v
            if normalized:
                gx[y][x] = sx / (8.0 * dx)
                gy[y][x] = sy / (8.0 * dy)
            else:
                gx[y][x] = sx
                gy[y][x] = sy
    return gx, gy


def gradient(height, method="sobel", boundary="clamp", dx=1.0, dy=1.0):
    """按名字选择梯度算子：``"sobel"`` 或 ``"central"``/``"center"``。"""
    m = (method or "sobel").lower() if isinstance(method, str) else method
    if m in ("sobel", "sobel_gradient"):
        return sobel(height, boundary=boundary, dx=dx, dy=dy)
    if m in ("central", "center", "central_difference", "central_diff"):
        return central_difference(height, boundary=boundary, dx=dx, dy=dy)
    raise InvalidParameterError(
        "未知梯度方法 %r（支持 'sobel' / 'central'）" % (method,))


# 常见别名
sobel_gradient = sobel
sobel_gradients = sobel
central_differences = central_difference
central_diff = central_difference
center_difference = central_difference


def sobel_x(height, boundary="clamp", dx=1.0):
    """只返回 Sobel 水平梯度 gx。"""
    return sobel(height, boundary=boundary, dx=dx)[0]


def sobel_y(height, boundary="clamp", dy=1.0):
    """只返回 Sobel 竖直梯度 gy。"""
    return sobel(height, boundary=boundary, dy=dy)[1]


def central_difference_x(height, boundary="clamp", dx=1.0):
    """只返回中心差分水平梯度 gx。"""
    return central_difference(height, boundary=boundary, dx=dx)[0]


def central_difference_y(height, boundary="clamp", dy=1.0):
    """只返回中心差分竖直梯度 gy。"""
    return central_difference(height, boundary=boundary, dy=dy)[1]


# ---------------------------------------------------------------------------
# UV 雅可比修正
# ---------------------------------------------------------------------------
def make_jacobian(rotation=0.0, scale=1.0, scale_y=None):
    """由旋转角（弧度）与缩放构造 2x2 雅可比 ``J = R(theta) @ diag(sx, sy)``。

    ``J`` 的列是 UV 空间一个单位步长对应的表面平面切向量，即
    ``J = d(x_surface, y_surface) / d(u, v)``。
    """
    try:
        theta = float(rotation)
    except (TypeError, ValueError):
        raise InvalidParameterError("rotation 必须是有限数值（弧度）") from None
    if not math.isfinite(theta):
        raise InvalidParameterError("rotation 必须有限")
    if isinstance(scale, (tuple, list)):
        if len(scale) != 2:
            raise InvalidParameterError("scale 序列必须有 2 个元素")
        sx, sy = float(scale[0]), float(scale[1])
    else:
        sx = float(scale)
        sy = sx if scale_y is None else float(scale_y)
    if not (math.isfinite(sx) and math.isfinite(sy)) or sx == 0.0 or sy == 0.0:
        raise InvalidParameterError("UV 缩放必须是非零有限实数")
    c, s = math.cos(theta), math.sin(theta)
    # R = [[c,-s],[s,c]],  R @ diag(sx,sy)
    return ((c * sx, -s * sy), (s * sx, c * sy))


def _coerce_jacobian(jacobian):
    if jacobian is None:
        return None
    try:
        if _np is not None:
            j = _np.asarray(jacobian, dtype=float)
        else:
            j = [[float(jacobian[r][c]) for c in range(2)] for r in range(2)]
    except Exception:
        raise InvalidJacobianError(
            "jacobian 必须是 2x2 实数矩阵: %r" % (jacobian,)) from None
    if _np is not None:
        if j.shape != (2, 2) or not _np.all(_np.isfinite(j)):
            raise InvalidJacobianError("jacobian 必须是有限的 2x2 矩阵")
        det = float(j[0, 0] * j[1, 1] - j[0, 1] * j[1, 0])
        if abs(det) < 1e-15:
            raise InvalidJacobianError("jacobian 不可逆（行列式为 0）")
        return j
    if len(j) != 2 or any(len(r) != 2 for r in j):
        raise InvalidJacobianError("jacobian 必须是 2x2 矩阵")
    if not all(math.isfinite(v) for r in j for v in r):
        raise InvalidJacobianError("jacobian 含非有限值")
    det = j[0][0] * j[1][1] - j[0][1] * j[1][0]
    if abs(det) < 1e-15:
        raise InvalidJacobianError("jacobian 不可逆（行列式为 0）")
    return j


def jacobian_correct(gx, gy, jacobian=None, *, uv_rotation=None,
                     uv_scale=None, uv_scale_x=None, uv_scale_y=None,
                     inverse=False, surface_to_uv=None):
    """把 UV 空间梯度修正为表面物理坐标梯度。

    设 ``J = d(x, y) / d(u, v)``，高度为 ``h(u, v)``，则链式法则给出
    ``[dh/du, dh/dv]^T = J^T [dh/dx, dh/dy]^T``，因此
    ``g_surface = J^{-T} g_uv``。

    可以直接传 2x2 的 ``jacobian``，也可以用 ``uv_rotation``（弧度）与
    ``uv_scale``（标量或 ``(sx, sy)``）让函数构造 ``R @ diag(sx, sy)``。
    ``gx/gy`` 可以是标量、单个 ``(gx, gy)`` 或同形二维嵌套列表。

    如果手头的矩阵定义相反，即 ``K = d(u, v) / d(x, y)``，可以传给
    ``surface_to_uv=K`` 或传 ``jacobian=K, inverse=True``。
    """
    if surface_to_uv is not None:
        k = _coerce_jacobian(surface_to_uv)
        if _np is not None and isinstance(k, _np.ndarray):
            jacobian = _np.linalg.inv(k)
        else:
            a, b = k[0][0], k[0][1]
            c, d = k[1][0], k[1][1]
            det = a * d - b * c
            jacobian = ((d / det, -b / det), (-c / det, a / det))
    if jacobian is None and (uv_rotation is not None or uv_scale is not None
                             or uv_scale_x is not None):
        rot = 0.0 if uv_rotation is None else uv_rotation
        if uv_scale is not None:
            sc = uv_scale
            sy = uv_scale_y
        else:
            sx = 1.0 if uv_scale_x is None else uv_scale_x
            sy = sx if uv_scale_y is None else uv_scale_y
            sc = (sx, sy)
        jacobian = make_jacobian(rot, sc)

    j = _coerce_jacobian(jacobian)
    if j is None:
        return gx, gy
    if inverse:
        if _np is not None and isinstance(j, _np.ndarray):
            j = _np.linalg.inv(j)
        else:
            a, b = j[0][0], j[0][1]
            c, d = j[1][0], j[1][1]
            det = a * d - b * c
            j = ((d / det, -b / det), (-c / det, a / det))

    scalar = not hasattr(gx, "__len__")
    if scalar:
        if _np is not None:
            g = _np.array([gx, gy], dtype=float)
            out = _np.linalg.solve(j.T, g)
            return float(out[0]), float(out[1])
        a, b = j[0][0], j[0][1]
        c, d = j[1][0], j[1][1]
        det = a * d - b * c
        # J^{-T} = (1/det) [[d, -c], [-b, a]]
        ox = (d * gx - c * gy) / det
        oy = (-b * gx + a * gy) / det
        return ox, oy

    if _np is not None:
        GX = _np.asarray(gx, dtype=float)
        GY = _np.asarray(gy, dtype=float)
        orig_shape = GX.shape
        g = _np.stack([GX.ravel(), GY.ravel()], axis=0)  # 2 x N
        out = _np.linalg.solve(j.T, g)
        OX = out[0].reshape(orig_shape)
        OY = out[1].reshape(orig_shape)
        # 输入若是 list，输出也给 list
        if isinstance(gx, list) or isinstance(gy, list):
            return OX.tolist(), OY.tolist()
        return OX, OY

    H = len(gx)
    W = len(gx[0])
    a, b = j[0][0], j[0][1]
    c, d = j[1][0], j[1][1]
    det = a * d - b * c
    ox = [[0.0] * W for _ in range(H)]
    oy = [[0.0] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            u, v = gx[y][x], gy[y][x]
            ox[y][x] = (d * u - c * v) / det
            oy[y][x] = (-b * u + a * v) / det
    return ox, oy


apply_jacobian_correction = jacobian_correct
apply_jacobian = jacobian_correct
correct_jacobian = jacobian_correct
correct_gradient_jacobian = jacobian_correct
uv_jacobian = jacobian_correct
uv_jacobian_correction = jacobian_correct
apply_uv_jacobian = jacobian_correct
correct_with_jacobian = jacobian_correct


def apply_uv_transform(gx, gy, rotation=0.0, scale=1.0, scale_y=None):
    """旋转 + 缩放 UV 时的雅可比修正快捷方式。"""
    return jacobian_correct(
        gx, gy, jacobian=make_jacobian(rotation, scale, scale_y))


# ---------------------------------------------------------------------------
# 法线生成
# ---------------------------------------------------------------------------
def _normals_from_gradients(gx, gy, strength):
    """由梯度场构造单位切线空间法线，返回 HxWx3 嵌套列表。"""
    if _np is not None:
        GX = _np.asarray(gx, dtype=float)
        GY = _np.asarray(gy, dtype=float)
        NX = -strength * GX
        NY = -strength * GY
        NZ = _np.ones_like(GX)
        norm = _np.sqrt(NX * NX + NY * NY + 1.0)
        out = _np.stack([NX / norm, NY / norm, NZ / norm], axis=-1)
        return out.tolist()
    H, W = len(gx), len(gx[0])
    out = [[None] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            nx = -strength * gx[y][x]
            ny = -strength * gy[y][x]
            out[y][x] = list(_normalize3(nx, ny, 1.0))
    return out


def from_height(height, strength=_UNSET, *, method="central",
                boundary="clamp", dx=1.0, dy=1.0,
                jacobian=None, uv_rotation=None, uv_scale=None,
                uv_scale_x=None, uv_scale_y=None, encode=False,
                J=None, uv_jacobian=None, uv_matrix=None,
                uv_to_surface=None, surface_to_uv=None, uv_transform=None,
                rotation=None, scale=None, **extra):
    """高度图 → 单位切线空间法线嵌套列表（HxWx3）。

    参数
    ----
    height:
        二维有限实数序列。
    strength:
        正有限强度，线性缩放 xy；``0``、负值、NaN/Inf、非数字均抛
        :class:`InvalidStrengthError`。
    method:
        ``"central"``（默认，中心差分，结果可被逐行积分精确反演）或
        ``"sobel"``。两种算子在线性斜坡场上完全一致。
    boundary:
        ``"clamp"``（默认，边缘复制）、``"wrap"`` 或 ``"zero"``。
    dx, dy:
        像素间距。
    jacobian / uv_rotation / uv_scale:
        UV 到表面平面的雅可比，存在旋转或非均匀缩放时必须提供，详见
        :func:`jacobian_correct`。``J``、``uv_jacobian``、``uv_matrix``、
        ``uv_to_surface`` 都是 ``jacobian`` 的同义参数；``surface_to_uv``
        接受反向矩阵 ``d(u,v)/d(x,y)``。
    encode:
        为 ``True`` 时额外做 ``0.5*n+0.5`` 的 ``[0,1]`` 编码并返回颜色。
    """
    if extra:
        raise TypeError("from_height 收到未知参数: %s"
                         % ", ".join(sorted(extra)))
    if jacobian is None:
        jacobian = (J if J is not None else uv_jacobian)
    if jacobian is None:
        jacobian = uv_matrix if uv_matrix is not None else uv_to_surface
    if uv_rotation is None and rotation is not None:
        uv_rotation = rotation
    if uv_scale is None and scale is not None:
        uv_scale = scale
    s = 1.0 if strength is _UNSET else check_strength(strength)
    dx = _check_positive_spacing(dx, "dx")
    dy = _check_positive_spacing(dy, "dy")

    # 解析雅可比（快速路径直接使用矩阵，避免列表往返）
    Jmat = None
    reverse = False
    if surface_to_uv is not None:
        Jmat = _coerce_jacobian(surface_to_uv)
        reverse = True
    else:
        if jacobian is None and uv_transform is not None \
                and not callable(uv_transform):
            jacobian = uv_transform
        if jacobian is not None or uv_rotation is not None \
                or uv_scale is not None or uv_scale_x is not None:
            if jacobian is None:
                rot = 0.0 if uv_rotation is None else uv_rotation
                if uv_scale is not None:
                    sc = uv_scale
                else:
                    sx = 1.0 if uv_scale_x is None else uv_scale_x
                    sy = sx if uv_scale_y is None else uv_scale_y
                    sc = (sx, sy)
                jacobian = make_jacobian(rot, sc)
            Jmat = _coerce_jacobian(jacobian)

    h, is_np = coerce_height(height)
    if is_np:
        m = method.lower() if isinstance(method, str) else method
        if m in ("sobel", "sobel_gradient"):
            gx, gy = _sobel_arrays_np(h, boundary, dx, dy)
        elif m in ("central", "center", "central_difference", "central_diff"):
            gx, gy = _central_arrays_np(h, boundary, dx, dy)
        else:
            raise InvalidParameterError(
                "未知梯度方法 %r（支持 'sobel' / 'central'）" % (method,))
        if Jmat is not None:
            juse = _np.linalg.inv(Jmat) if reverse else Jmat
            flat = _np.stack([gx.ravel(), gy.ravel()], axis=0)
            corrected = _np.linalg.solve(juse.T, flat)
            gx = corrected[0].reshape(gx.shape)
            gy = corrected[1].reshape(gy.shape)
        nxn = -s * gx
        nyn = -s * gy
        norm = _np.sqrt(nxn * nxn + nyn * nyn + 1.0)
        out = _np.stack([nxn / norm, nyn / norm, 1.0 / norm], axis=-1)
        if encode:
            out = _np.clip(0.5 * out + 0.5, 0.0, 1.0)
        return out.tolist()

    # 纯 Python 路径
    gx, gy = gradient(height, method=method, boundary=boundary, dx=dx, dy=dy)
    if reverse:
        gx, gy = jacobian_correct(gx, gy, surface_to_uv=surface_to_uv)
    elif Jmat is not None:
        gx, gy = jacobian_correct(gx, gy, jacobian=jacobian)
    normals = _normals_from_gradients(gx, gy, s)
    if encode:
        return encode(normals)
    return normals


class NormalMap:
    """高度图 / 法线图双向转换的无状态工具类。

    构造参数全部可选，仅用于设定默认值；每个方法都可以再次覆盖。
    """

    def __init__(self, strength=1.0, method="central", boundary="clamp"):
        self.strength = check_strength(strength)
        if not isinstance(method, str) or method.lower() not in (
                "sobel", "central", "center", "central_difference",
                "central_diff"):
            raise InvalidParameterError("未知默认梯度方法: %r" % (method,))
        self.method = method
        if boundary not in ("clamp", "edge", "replicate", "wrap",
                            "periodic", "tile", "zero", "zeros", "constant"):
            raise InvalidParameterError("未知 boundary: %r" % (boundary,))
        self.boundary = boundary

    # -- 主契约 ------------------------------------------------------------
    def from_height(self, height, strength=_UNSET, **kwargs):
        """生成法线；未显式传 strength 时使用构造时的默认强度。"""
        if strength is _UNSET:
            strength = self.strength
        kwargs.setdefault("method", self.method)
        kwargs.setdefault("boundary", self.boundary)
        return from_height(height, strength, **kwargs)

    generate = from_height
    generate_normals = from_height
    height_to_normal = from_height
    height_to_normals = from_height
    convert = from_height

    def decode(self, rgb=None, *rest, normalize=True):
        """把 ``[0,1]`` 编码法线解码为单位 ``(nx, ny, nz)`` float 三元组。"""
        return decode(rgb, *rest, normalize=normalize)

    # -- 编码 --------------------------------------------------------------
    def encode(self, normal=None, *rest):
        """单位法线 → ``[0,1]`` 编码；支持单个三元组或整张 HxWx3 图。"""
        return encode(normal, *rest)

    def encode8(self, normal):
        """单位法线 → 0..255 整数字节三元组（或整张图）。"""
        return encode8(normal)

    def decode8(self, rgb):
        """0..255 整数颜色 → 单位法线三元组。"""
        return decode8(rgb)

    def to_bytes(self, normal_map):
        """把整张编码/单位法线图打包成 bytes（每像素 3 字节）。"""
        return to_bytes(normal_map)

    to_rgb = encode
    from_rgb = decode
    pack = encode
    unpack = decode

    # -- 梯度 --------------------------------------------------------------
    def sobel(self, height, **kwargs):
        kwargs.setdefault("boundary", self.boundary)
        return sobel(height, **kwargs)

    def central_difference(self, height, **kwargs):
        kwargs.setdefault("boundary", self.boundary)
        return central_difference(height, **kwargs)

    central_diff = central_difference
    center_difference = central_difference

    def gradient(self, height, method=None, **kwargs):
        kwargs.setdefault("boundary", self.boundary)
        return gradient(height, self.method if method is None else method,
                        **kwargs)

    # -- 雅可比 ------------------------------------------------------------
    def jacobian_correct(self, gx, gy, jacobian=None, **kwargs):
        return jacobian_correct(gx, gy, jacobian, **kwargs)

    apply_jacobian_correction = jacobian_correct

    def apply_uv_transform(self, gx, gy, rotation=0.0, scale=1.0,
                           scale_y=None):
        return apply_uv_transform(gx, gy, rotation, scale, scale_y)

    # -- 重建 --------------------------------------------------------------
    def reconstruct_height(self, normals, strength=1.0, reference=None,
                           method="rows", **kwargs):
        """从法线积分重建高度，详见 :func:`reconstruct_height`。"""
        return reconstruct_height(
            normals, strength, reference, method=method, **kwargs)

    height_from_normals = reconstruct_height
    from_normals = reconstruct_height
    integrate_normals = reconstruct_height
    integrate = reconstruct_height
    normals_to_height = reconstruct_height
    normal_to_height = reconstruct_height
    height_from_normal = reconstruct_height

    def poisson_reconstruct(self, normals, strength=1.0, **kwargs):
        """强制使用泊松最小二乘重建。"""
        kwargs["method"] = "poisson"
        return reconstruct_height(normals, strength, **kwargs)

    def integrate_rows(self, normals, strength=1.0, **kwargs):
        """逐行积分重建高度。"""
        kwargs["method"] = "rows"
        return reconstruct_height(normals, strength=strength, **kwargs)

    # -- mip ---------------------------------------------------------------
    def mip_chain(self, normals, levels=None, *, encoded="auto",
                  include_base=True, output_encoded=False):
        """构造逐级 2x2 盒平均并逐级重归一的 mip 链。"""
        return mip_chain(normals, levels=levels, encoded=encoded,
                         include_base=include_base,
                         output_encoded=output_encoded)

    build_mip_chain = mip_chain
    generate_mip_chain = mip_chain
    mips = mip_chain
    build_mips = mip_chain
    generate_mips = mip_chain

    def mip_level(self, normals, level=1, **kwargs):
        """直接返回第 ``level`` 级 mip（0 为原图）。"""
        return mip_level(normals, level, **kwargs)

    def downsample_normals(self, normals, encoded="auto", **kwargs):
        """一级 2x2 平均后重归一。"""
        return downsample_normals(normals, encoded=encoded, **kwargs)

    # -- 切线空间 ----------------------------------------------------------
    def tangent_basis(self, normal, hint=None):
        return tangent_basis(normal, hint=hint)

    tangent_frame = tangent_basis
    tbn = tangent_basis
    orthonormal_basis = tangent_basis

    def tangent_space(self, normal, hint=None):
        """返回 3x3 TBN 正交矩阵。"""
        return tangent_matrix(normal, hint=hint)

    def is_orthonormal(self, basis, tol=1e-6):
        return is_orthonormal(basis, tol=tol)

    def orthonormalize(self, tangent, bitangent, normal):
        return orthonormalize(tangent, bitangent, normal)

    def frame_from_gradients(self, gx, gy):
        return frame_from_gradients(gx, gy)

    def to_tangent_space(self, n, T, B, N):
        return to_tangent_space(n, T, B, N)

    def from_tangent_space(self, nt, T, B, N):
        return from_tangent_space(nt, T, B, N)

    to_world_space = from_tangent_space
    tangent_to_world = from_tangent_space
    world_to_tangent = to_tangent_space


def height_to_normals(height, strength=1.0, **kwargs):
    """模块级快捷方式：高度图 → 单位法线。"""
    return from_height(height, strength, **kwargs)


NormalMapper = NormalMap

# 允许以 NormalMap.InvalidStrengthError 形式访问异常类型
NormalMap.NormalMapError = NormalMapError
NormalMap.InvalidTypeError = InvalidTypeError
NormalMap.InvalidHeightError = InvalidHeightError
NormalMap.NonFiniteError = NonFiniteError
NormalMap.InvalidStrengthError = InvalidStrengthError
NormalMap.NonFiniteStrengthError = NonFiniteStrengthError
NormalMap.InvalidParameterError = InvalidParameterError
NormalMap.InvalidJacobianError = InvalidJacobianError


# ---------------------------------------------------------------------------
# 编码 / 解码
# ---------------------------------------------------------------------------
def encode(normal=None, *rest):
    """单位法线 → ``[0,1]`` RGB。

    输入长度为 3 的一维向量时返回三元组；输入 HxWx3 嵌套结构时返回同形
    嵌套列表。也支持 ``encode(nx, ny, nz)`` 形式。分量会被裁剪到 ``[0,1]``。
    """
    if rest:
        if len(rest) != 2:
            raise InvalidShapeError(
                "encode 需要 3 个分量：encode(nx,ny,nz) 或 encode((nx,ny,nz))")
        normal = (normal, rest[0], rest[1])
    arr = normal
    if _np is not None:
        try:
            a = _np.asarray(normal, dtype=float)
        except Exception:
            raise InvalidTypeError("encode 输入无法转换为数值数组") from None
        if a.shape == (3,):
            v = _np.clip(0.5 * a + 0.5, 0.0, 1.0)
            return (float(v[0]), float(v[1]), float(v[2]))
        if a.ndim == 3 and a.shape[2] == 3:
            v = _np.clip(0.5 * a + 0.5, 0.0, 1.0)
            return v.tolist()
        raise InvalidShapeError(
            "encode 只接受长度 3 的向量或 HxWx3 法线图，实际形状 %r"
            % (a.shape,))
    return _py_encode(normal)


encode_normal = encode
to_rgb = encode
pack_normal = encode


def _py_encode(normal):
    if len(normal) == 3 and not hasattr(normal[0], "__len__"):
        vals = _finite_vec3(normal, "normal")
        return tuple(min(1.0, max(0.0, 0.5 * v + 0.5)) for v in vals)
    out = []
    for row in normal:
        out.append([
            [min(1.0, max(0.0, 0.5 * c + 0.5)) for c in _finite_vec3(p, "pixel")]
            for p in row
        ])
    return out


def decode(rgb=None, *rest, normalize=True):
    """``[0,1]`` 编码 RGB → 单位法线三元组。

    支持 ``decode((r,g,b))``、``decode([r,g,b])``、``decode(bytes)`` 以及
    ``decode(r, g, b)`` 四种调用形式。映射 ``n = 2*rgb - 1`` 后归一化。
    若任一分量大于 ``1.5``，按 8-bit 0..255 颜色处理（先除以 255）。
    长度不为 3 抛 :class:`InvalidShapeError`，非有限分量抛
    :class:`NonFiniteError`。
    """
    if rest:
        if len(rest) != 2:
            raise InvalidShapeError(
                "decode 需要 3 个分量：decode(r,g,b) 或 decode((r,g,b))")
        rgb = (rgb, rest[0], rest[1])
    if isinstance(rgb, str):
        raise InvalidTypeError("decode 需要长度 3 的数值序列")
    if isinstance(rgb, (bytes, bytearray)):
        if len(rgb) != 3:
            raise InvalidShapeError(
                "decode 的字节输入必须恰好 3 字节，实际 %d" % len(rgb))
        vals = [float(c) / 255.0 for c in rgb]
        x, y, z = 2.0 * vals[0] - 1.0, 2.0 * vals[1] - 1.0, 2.0 * vals[2] - 1.0
        return _normalize3(x, y, z) if normalize else (x, y, z)
    try:
        n_comp = len(rgb)
    except TypeError:
        raise InvalidTypeError("decode 需要长度 3 的数值序列") from None
    if n_comp != 3:
        raise InvalidShapeError(
            "decode 输入必须恰好有 3 个分量，实际 %d 个: %r"
            % (n_comp, rgb))
    vals = _finite_vec3(rgb, "rgb")
    # 仅当三个分量都像 0..255 的整数字节且至少一个 >1 时，才按 8-bit 处理，
    # 避免误判越界浮点编码。
    looks_like_bytes = (
        any(v > 1.0 for v in vals)
        and all(0.0 <= v <= 255.0 for v in vals)
        and all(abs(v - round(v)) < 1e-9 for v in vals)
    )
    if looks_like_bytes:
        vals = [v / 255.0 for v in vals]
    x, y, z = 2.0 * vals[0] - 1.0, 2.0 * vals[1] - 1.0, 2.0 * vals[2] - 1.0
    if normalize:
        return _normalize3(x, y, z)
    return (x, y, z)


decode_normal = decode
from_rgb = decode
unpack_normal = decode


def encode8(normal):
    """单位法线 → 0..255 整数 RGB（四舍五入）。"""
    enc = encode(normal)
    if isinstance(enc, tuple):
        return tuple(int(min(255, max(0, round(c * 255.0)))) for c in enc)
    return [[[int(min(255, max(0, round(c * 255.0)))) for c in px]
             for px in row] for row in enc]


def decode8(rgb):
    """0..255 整数 RGB → 单位法线三元组。"""
    vals = _finite_vec3(rgb, "rgb8")
    return decode(tuple(v / 255.0 for v in vals))


def to_bytes(normal_map, encoded="auto"):
    """把 HxWx3 法线打包为每像素 3 字节的 bytes。

    默认 ``encoded="auto"`` 自动区分输入是单位法线还是已经编码到
    ``[0,1]`` 的颜色：单位法线先做 ``0.5n+0.5``。也可显式传
    ``encoded=True/False``。
    """
    if _np is not None:
        a = _np.asarray(normal_map, dtype=float)
        if a.ndim != 3 or a.shape[2] != 3:
            raise InvalidShapeError("to_bytes 需要 HxWx3 结构")
        if encoded == "auto":
            lens = _np.sqrt(_np.sum(a * a, axis=-1))
            raw_unit = float(_np.median(_np.abs(lens - 1.0))) < 0.05 \
                or float(_np.nanmin(a)) < -1e-9
            do_encode = raw_unit
        else:
            do_encode = not encoded
        if do_encode:
            a = _np.clip(0.5 * a + 0.5, 0.0, 1.0)
        else:
            a = _np.clip(a, 0.0, 1.0)
        return bytes((_np.rint(a * 255.0)).astype("uint8").ravel().tolist())
    vals_all = [[_finite_vec3(px, "pixel") for px in row] for row in normal_map]
    if encoded == "auto":
        res = 0.0
        cnt = 0
        has_neg = False
        for row in vals_all:
            for v in row:
                res += abs(math.sqrt(sum(c * c for c in v)) - 1.0)
                cnt += 1
                has_neg = has_neg or min(v) < -1e-9
        do_encode = has_neg or (cnt and res / cnt < 0.05)
    else:
        do_encode = not encoded
    buf = bytearray()
    for row in vals_all:
        for v in row:
            if do_encode:
                v = [min(1.0, max(0.0, 0.5 * c + 0.5)) for c in v]
            for c in v:
                buf.append(int(min(255, max(0, round(c * 255.0)))))
    return bytes(buf)


# ---------------------------------------------------------------------------
# 输入归一化：把“单位法线 / [0,1] 编码 / 8-bit”统一成单位法线场
# ---------------------------------------------------------------------------
def _normal_array(normals, encoded=False):
    """返回 numpy HxWx3 单位法线数组（或纯 Python 嵌套列表 + False 标记）。"""
    if _np is not None:
        try:
            a = _np.asarray(normals, dtype=float)
        except Exception:
            raise InvalidTypeError("法线图无法转换为数值数组") from None
        if a.ndim != 3 or a.shape[2] != 3 or a.shape[0] == 0 or a.shape[1] == 0:
            raise InvalidShapeError("法线图必须是非空 HxWx3 结构")
        if not _np.all(_np.isfinite(a)):
            raise NonFiniteError("法线图含非有限分量")
        if encoded is True:
            a = 2.0 * a - 1.0
        elif encoded == "auto":
            a = _auto_decode_np(a)
        elif encoded in (False, None):
            pass
        else:
            raise InvalidParameterError("encoded 只支持 True/False/'auto'")
        lengths = _np.sqrt(_np.sum(a * a, axis=-1, keepdims=True))
        lengths = _np.where(lengths < 1e-12, 1.0, lengths)
        return a / lengths, True

    # 纯 Python
    rows = normals
    if len(rows) == 0:
        raise InvalidShapeError("法线图不能为空")
    H = len(rows)
    W = len(rows[0])
    out = [[None] * W for _ in range(H)]
    for y in range(H):
        if len(rows[y]) != W:
            raise InvalidShapeError("法线图各行宽度不一致")
        for x in range(W):
            v = _finite_vec3(rows[y][x], "normal")
            if encoded is True:
                v = [2.0 * c - 1.0 for c in v]
            elif encoded == "auto":
                v = _auto_decode_py(v)
            out[y][x] = _normalize_vec(v)
    return out, False


def _auto_decode_np(a):
    flat = a.reshape(-1, 3)
    if _np.any(flat > 1.5):
        a = a / 255.0
        flat = a.reshape(-1, 3)
    raw_err = float(_np.median(_abs_len_minus_one(flat)))
    dec = 2.0 * flat - 1.0
    dec_err = float(_np.median(_abs_len_minus_one(dec)))
    if dec_err + 1e-9 < raw_err:
        return (2.0 * a - 1.0)
    return a


def _abs_len_minus_one(v):
    return _np.abs(_np.sqrt(_np.sum(v * v, axis=-1)) - 1.0)


def _auto_decode_py(v):
    if any(c > 1.5 for c in v):
        v = [c / 255.0 for c in v]
    raw_err = abs(math.sqrt(sum(c * c for c in v)) - 1.0)
    dec = [2.0 * c - 1.0 for c in v]
    dec_err = abs(math.sqrt(sum(c * c for c in dec)) - 1.0)
    return dec if dec_err + 1e-9 < raw_err else v


# ---------------------------------------------------------------------------
# mip 链：2x2 盒平均 + 每一级重归一
# ---------------------------------------------------------------------------
def downsample_normals(normals, encoded="auto", output_encoded=False):
    """把 HxWx3 法线图按 2x2 盒平均降一半，并对结果逐像素重归一。

    奇数尺寸时先复制边缘到偶数尺寸，因此输出尺寸为 ``ceil(H/2) x ceil(W/2)``。
    ``encoded="auto"``（默认）会自动识别输入是单位法线还是 ``[0,1]`` 编码；
    ``output_encoded=True`` 时把输出重新编码到 ``[0,1]``。
    """
    n, is_np = _normal_array(normals, encoded=encoded)
    if is_np:
        H, W, _ = n.shape
        pe = _np.pad(n, ((0, H % 2), (0, W % 2), (0, 0)), mode="edge")
        H2, W2 = pe.shape[0] // 2, pe.shape[1] // 2
        blocks = pe.reshape(H2, 2, W2, 2, 3)
        avg = blocks.mean(axis=(1, 3))
        lens = _np.sqrt(_np.sum(avg * avg, axis=-1, keepdims=True))
        lens = _np.where(lens < 1e-12, 1.0, lens)
        out = (avg / lens)
        if output_encoded:
            return _np.clip(0.5 * out + 0.5, 0.0, 1.0).tolist()
        return out.tolist()

    H, W = len(n), len(n[0])
    oH = (H + 1) // 2
    oW = (W + 1) // 2
    out = [[(0.0, 0.0, 1.0)] * oW for _ in range(oH)]
    for oy in range(oH):
        for ox in range(oW):
            sx = sy = sz = 0.0
            cnt = 0
            for dy in (0, 1):
                for dx in (0, 1):
                    yy = min(2 * oy + dy, H - 1)
                    xx = min(2 * ox + dx, W - 1)
                    px = n[yy][xx]
                    sx += px[0]
                    sy += px[1]
                    sz += px[2]
                    cnt += 1
            v = _normalize3(sx / cnt, sy / cnt, sz / cnt)
            if output_encoded:
                v = tuple(min(1.0, max(0.0, 0.5 * c + 0.5)) for c in v)
            out[oy][ox] = v
    return out


def mip_chain(normals, levels=None, *, encoded="auto", include_base=True,
              output_encoded=False):
    """构造重归一 mip 链。

    返回 ``[level0, level1, ...]``，``level0`` 为输入的单位化版本，之后每级
    做一次 :func:`downsample_normals`。``levels`` 为返回级数上限（含 0）；
    默认一直降到 1x1。``include_base=False`` 时只返回降采样级别。
    输入二维高度图时会先调用 :func:`from_height` 转成法线。
    """
    # 允许直接传高度图
    if _looks_like_height(normals):
        base = from_height(normals)
        encoded = False
    else:
        base, _ = _normal_array(normals, encoded=encoded)
        if _np is not None and isinstance(base, _np.ndarray):
            base = base.tolist()

    full_levels = max(1, int(math.ceil(math.log2(
        max(len(base), len(base[0]))))) + 1)

    if levels is not None:
        if not isinstance(levels, int) or isinstance(levels, bool) or levels <= 0:
            raise InvalidParameterError("levels 必须是正整数: %r" % (levels,))
        count = levels
    else:
        count = full_levels

    chain = [base]
    cur = base
    for _ in range(count - 1):
        if len(cur) == 1 and len(cur[0]) == 1:
            break
        cur = downsample_normals(cur, encoded=False)
        chain.append(cur)

    if output_encoded:
        chain = [encode(level) for level in chain]
    if not include_base:
        return chain[1:]
    return chain


def _looks_like_height(data):
    """二维标量序列视为高度图；HxWx3 视为法线图。"""
    if isinstance(data, (str, bytes)):
        return False
    try:
        first = data[0]
    except (IndexError, KeyError, TypeError):
        return False
    if not hasattr(first, "__len__") or isinstance(first, (str, bytes)):
        return False
    try:
        px = first[0]
    except (IndexError, KeyError, TypeError):
        return False
    return not hasattr(px, "__len__") or isinstance(px, (str, bytes))


def _dim_h(m):
    return len(m)


def _dim_w(m):
    return len(m[0])


def mip_level(normals, level=1, **kwargs):
    """返回第 ``level`` 级 mip（0 为原图）。"""
    if not isinstance(level, int) or isinstance(level, bool) or level < 0:
        raise InvalidParameterError("level 必须是非负整数")
    chain = mip_chain(normals, levels=level + 1, **kwargs)
    return chain[level]


build_mip_chain = mip_chain
generate_mip_chain = mip_chain
mips = mip_chain
mipmaps = mip_chain
mipmap_chain = mip_chain
build_mips = mip_chain
generate_mips = mip_chain
generate_mipmaps = mip_chain
build_mipmaps = mip_chain
generate_mip_levels = mip_chain
mip_levels_chain = mip_chain


# ---------------------------------------------------------------------------
# 由法线重建高度
# ---------------------------------------------------------------------------
def reconstruct_height(normals, strength=1.0, reference=None,
                       method="rows", encoded="auto", **kwargs):
    """从切线空间法线积分重建高度图。

    由 ``nx,ny,nz`` 与强度恢复梯度::

        dh/dx = -nx / (strength * nz)
        dh/dy = -ny / (strength * nz)

    ``method="rows"``（默认）逐行利用中心差分奇偶链精确反演，再递推行
    常数，对本模块 :func:`from_height` 产出的中心差分法线可近乎精确重建，
    纯 Python 也可运行。``method="poisson"`` 求解最小二乘泊松方程，对
    外来法线或含噪梯度在二维上整体分配误差，同样可用。

    积分只能确定到一个加性常数。若提供 ``reference`` 原始高度图，返回前
    会把结果的均值与尺度最小二乘对齐到参考图。除标准签名外，也兼容
    ``reconstruct_height(normals, original_height)`` 这种把原图作为第二
    个位置参数的调用形式。
    """
    # 兼容 reconstruct_height(normals, original_height)
    if hasattr(strength, "__len__") and not isinstance(
            strength, (str, bytes, bytearray, dict)):
        if reference is None:
            reference = strength
        strength = 1.0
    # 兼容第三个位置参数传 method 字符串
    if isinstance(reference, str):
        method = reference
        reference = None
    if reference is None:
        for key in ("original", "original_height", "ref", "target",
                    "height", "height_map", "ground_truth"):
            if key in kwargs:
                reference = kwargs.pop(key)
                break
    s = check_strength(strength)
    n, is_np = _normal_array(normals, encoded=encoded)

    if method in ("poisson", "fft", "cg"):
        if is_np:
            h = _poisson_np(n, s)
        else:
            h = _rows_py(n, s)
    elif method in ("rows", "row", "rowwise", "line"):
        if is_np:
            gx = -n[:, :, 0] / (s * n[:, :, 2])
            gy = -n[:, :, 1] / (s * n[:, :, 2])
            h = _row_integrate_np(gx, gy)
        else:
            h = _rows_py(n, s)
    else:
        raise InvalidParameterError(
            "未知重建方法 %r（支持 'poisson' / 'rows'）" % (method,))

    if reference is not None:
        h = _align_reference(h, reference)
    elif is_np and isinstance(h, _np.ndarray):
        h = h - float(_np.mean(h))
        return h.tolist()
    else:
        mean = sum(sum(row) for row in h) / (len(h) * len(h[0]))
        h = [[v - mean for v in row] for row in h]
    return h


def _gradient_fields_from_normals_np(n, s):
    nz = n[:, :, 2]
    safe_nz = _np.where(_np.abs(nz) < 1e-9,
                        _np.where(nz < 0, -1e-9, 1e-9), nz)
    gx = -n[:, :, 0] / (s * safe_nz)
    gy = -n[:, :, 1] / (s * safe_nz)
    return gx, gy


def _poisson_np(n, s):
    gx, gy = _gradient_fields_from_normals_np(n, s)
    H, W = gx.shape
    # 像素中心梯度转换为前向边目标：边 (x,x+1) 取两中心的平均
    ex = 0.5 * (gx[:, :-1] + gx[:, 1:])
    ey = 0.5 * (gy[:-1, :] + gy[1:, :])
    rhs = _np.zeros((H, W), dtype=float)
    # div：D_x^T ex + D_y^T ey（前向差分边）
    rhs[:, :-1] -= ex
    rhs[:, 1:] += ex
    rhs[:-1, :] -= ey
    rhs[1:, :] += ey

    def laplacian(h):
        out = _np.zeros_like(h)
        dx = h[:, 1:] - h[:, :-1]
        dy = h[1:, :] - h[:-1, :]
        out[:, :-1] -= dx
        out[:, 1:] += dx
        out[:-1, :] -= dy
        out[1:, :] += dy
        return out

    # 加极小正则消除常数零空间，条件数好且不改变可分辨形状
    eps = 1e-10
    b = rhs
    x = _np.zeros((H, W), dtype=float)
    r = b - (laplacian(x) + eps * x)
    p = r.copy()
    rs_old = float(_np.sum(r * r))
    bnorm = math.sqrt(max(float(_np.sum(b * b)), 1e-30))
    max_iter = max(2000, 4 * max(H, W))
    tol = 1e-11
    for _ in range(max_iter):
        Ap = laplacian(p) + eps * p
        alpha = rs_old / max(float(_np.sum(p * Ap)), 1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(_np.sum(r * r))
        if math.sqrt(rs_new) < tol * bnorm:
            break
        beta = rs_new / max(rs_old, 1e-300)
        p = r + beta * p
        rs_old = rs_new
    x = x - float(_np.mean(x))
    return x


def _row_integrate_np(gx, gy):
    H, W = gx.shape
    # 利用中心差分的奇偶链反演每行（边界 clamp 时可精确还原到行常数）
    rows_h = _np.zeros((H, W), dtype=float)
    if W >= 2:
        rows_h[:, 1] = 2.0 * gx[:, 0]
        for x in range(1, W - 1):
            rows_h[:, x + 1] = rows_h[:, x - 1] + 2.0 * gx[:, x]
    # 行常数 c[y]：对纵向中心差分方程按 x 平均后递推
    # y=0 clamp: c1-c0 = 2*mean(gy0) - (r1bar-r0bar)
    rbar = rows_h.mean(axis=1)
    gbar = gy.mean(axis=1)
    c = _np.zeros(H, dtype=float)
    if H >= 2:
        c[1] = 2.0 * gbar[0] - (rbar[1] - rbar[0])
        for y in range(1, H - 1):
            d = 2.0 * gbar[y] - (rbar[y + 1] - rbar[y - 1])
            c[y + 1] = d + c[y - 1]
    rows_h = rows_h + c[:, None]
    rows_h = rows_h - float(_np.mean(rows_h))
    return rows_h


def _rows_py(n, s):
    H, W = len(n), len(n[0])
    gx = [[0.0] * W for _ in range(H)]
    gy = [[0.0] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            nx, ny, nz = n[y][x]
            if abs(nz) < 1e-9:
                nz = -1e-9 if nz < 0 else 1e-9
            gx[y][x] = -nx / (s * nz)
            gy[y][x] = -ny / (s * nz)
    # 利用中心差分的奇偶链反演，边界 clamp 时可精确还原每行（差一个常数）
    h = [[0.0] * W for _ in range(H)]
    for y in range(H):
        row = h[y]
        row[0] = 0.0
        if W >= 2:
            row[1] = 2.0 * gx[y][0]
            for x in range(1, W - 1):
                row[x + 1] = row[x - 1] + 2.0 * gx[y][x]
    # 行常数：纵向中心差分方程按 x 平均后递推
    if H >= 2:
        rbar = [sum(h[y]) / W for y in range(H)]
        gbar = [sum(gy[y]) / W for y in range(H)]
        c = [0.0] * H
        c[1] = 2.0 * gbar[0] - (rbar[1] - rbar[0])
        for y in range(1, H - 1):
            d = 2.0 * gbar[y] - (rbar[y + 1] - rbar[y - 1])
            c[y + 1] = d + c[y - 1]
        for y in range(H):
            for x in range(W):
                h[y][x] += c[y]
    m = sum(sum(r) for r in h) / (H * W)
    return [[v - m for v in r] for r in h]


def _align_reference(h, reference):
    ref, is_np = coerce_height(reference)
    if _np is not None and isinstance(h, _np.ndarray):
        href = ref if is_np else _np.asarray(ref, dtype=float)
        h = h - float(_np.mean(h))
        href = href - float(_np.mean(href))
        denom = float(_np.sum(h * h))
        if denom > 1e-30:
            a = float(_np.sum(h * href) / denom)
            h = a * h
        return h.tolist()
    # 纯 Python
    H = len(h)
    W = len(h[0])
    mh = sum(sum(r) for r in h) / (H * W)
    mr = sum(sum(r) for r in ref) / (H * W)
    hc = [[v - mh for v in r] for r in h]
    rc = [[ref[y][x] - mr for x in range(W)] for y in range(H)]
    num = sum(hc[y][x] * rc[y][x] for y in range(H) for x in range(W))
    den = sum(v * v for r in hc for v in r)
    a = num / den if den > 1e-30 else 1.0
    return [[a * v for v in r] for r in hc]


height_from_normals = reconstruct_height
from_normals = reconstruct_height
integrate_normals = reconstruct_height
integrate = reconstruct_height
normals_to_height = reconstruct_height
normal_to_height = reconstruct_height
normalmap_to_height = reconstruct_height


def integrate_rows(normals, strength=1.0, *, encoded="auto", reference=None):
    """逐行积分重建高度（行内奇偶链反演 + 纵向常数递推）。"""
    return reconstruct_height(normals, strength=strength, encoded=encoded,
                              method="rows", reference=reference)


row_integrate = integrate_rows
rowwise_integrate = integrate_rows


def poisson_reconstruct(normals, strength=1.0, *, encoded="auto",
                        reference=None):
    """泊松最小二乘重建高度。"""
    return reconstruct_height(normals, strength=strength, encoded=encoded,
                              method="poisson", reference=reference)


poisson_integrate = poisson_reconstruct
heightmap_from_normals = reconstruct_height
reconstruct_height_from_normals = reconstruct_height
height_from_normal = reconstruct_height
height_from_normals = reconstruct_height
normal_to_height_map = reconstruct_height


def height_to_normal(height, strength=1.0, **kwargs):
    """``from_height`` 的单数命名别名。"""
    return from_height(height, strength, **kwargs)


def height_to_normalmap(height, strength=1.0, **kwargs):
    """``from_height`` 的 normalmap 命名别名。"""
    return from_height(height, strength, **kwargs)


def height_to_normal_map(height, strength=1.0, **kwargs):
    """``from_height`` 的带下划线命名别名。"""
    return from_height(height, strength, **kwargs)


def normalmap_from_height(height, strength=1.0, **kwargs):
    """``from_height`` 的 normalmap 前缀别名。"""
    return from_height(height, strength, **kwargs)


# ---------------------------------------------------------------------------
# 切线空间正交归一基 / TBN
# ---------------------------------------------------------------------------
def orthonormal_basis(normal, hint=None):
    """由一个法线向量构造正交归一的 ``(T, B, N)``。

    返回的三个向量两两正交且长度为 1（误差在双精度机器精度量级）。
    """
    nx, ny, nz = _finite_vec3(normal, "normal")
    nrm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nrm < 1e-12:
        raise InvalidParameterError("无法为零长度法线构造切线基")
    N = (nx / nrm, ny / nrm, nz / nrm)
    if hint is None:
        hint = (0.0, 0.0, 1.0) if abs(N[2]) < 0.9 else (1.0, 0.0, 0.0)
    hx, hy, hz = _finite_vec3(hint, "hint")
    # T = normalize(cross(hint, N))
    cx = hy * N[2] - hz * N[1]
    cy = hz * N[0] - hx * N[2]
    cz = hx * N[1] - hy * N[0]
    tn = math.sqrt(cx * cx + cy * cy + cz * cz)
    if tn < 1e-12:
        hint = (1.0, 0.0, 0.0) if abs(N[0]) < 0.9 else (0.0, 1.0, 0.0)
        cx = hint[1] * N[2] - hint[2] * N[1]
        cy = hint[2] * N[0] - hint[0] * N[2]
        cz = hint[0] * N[1] - hint[1] * N[0]
        tn = math.sqrt(cx * cx + cy * cy + cz * cz)
    T = (cx / tn, cy / tn, cz / tn)
    # B = cross(N, T)，保证 (T, B, N) 右手系
    Bx = N[1] * T[2] - N[2] * T[1]
    By = N[2] * T[0] - N[0] * T[2]
    Bz = N[0] * T[1] - N[1] * T[0]
    bn = math.sqrt(Bx * Bx + By * By + Bz * Bz)
    B = (Bx / bn, By / bn, Bz / bn)
    return T, B, N


tangent_basis = orthonormal_basis
tangent_frame = orthonormal_basis
build_tbn = orthonormal_basis
tbn = orthonormal_basis


def tangent_matrix(normal, hint=None):
    """返回 3x3 的 TBN 正交矩阵 ``((Tx,Ty,Tz),(Bx,...),(Nx,...))``。"""
    T, B, N = orthonormal_basis(normal, hint=hint)
    return (tuple(T), tuple(B), tuple(N))


def tbn_matrix(normal, hint=None):
    """``tangent_matrix`` 的别名，返回 ``(T, B, N)`` 三元组。"""
    return orthonormal_basis(normal, hint=hint)


tangent_space = tangent_matrix
build_tangent_space = tangent_matrix
make_tangent_space = orthonormal_basis


def is_orthonormal(basis, tol=1e-6):
    """判断由三个三维向量组成的基 ``(v0,v1,v2)`` 是否正交归一。"""
    try:
        rows = [tuple(r) for r in basis]
        if len(rows) != 3 or any(len(r) != 3 for r in rows):
            return False
        for i in range(3):
            if abs(math.sqrt(sum(c * c for c in rows[i])) - 1.0) > tol:
                return False
            for j in range(i):
                if abs(sum(rows[i][k] * rows[j][k] for k in range(3))) > tol:
                    return False
        return True
    except Exception:
        return False


def orthonormalize(tangent, bitangent, normal):
    """对给定 T/B/N 做 Gram-Schmidt，返回严格正交归一的 ``(T, B, N)``。"""
    T0 = _finite_vec3(tangent, "tangent")
    B0 = _finite_vec3(bitangent, "bitangent")
    N0 = _finite_vec3(normal, "normal")

    def norm(v):
        m = math.sqrt(sum(c * c for c in v))
        if m < 1e-12:
            raise InvalidParameterError("正交归一化遇到零长度向量")
        return [c / m for c in v]

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    N = norm(N0)
    T = norm(T0)
    # T 去掉与 N 平行的分量
    d = dot(T, N)
    T = norm([T[i] - d * N[i] for i in range(3)])
    # B 依次去掉与 N、T 平行的分量
    B = B0[:]
    dn = dot(B, N)
    B = [B[i] - dn * N[i] for i in range(3)]
    dt = dot(B, T)
    B = norm([B[i] - dt * T[i] for i in range(3)])
    return tuple(T), tuple(B), tuple(N)


def frame_from_gradients(gx, gy):
    """由标量梯度场构造逐像素正交归一 TBN。

    高度场参数化 ``r(x,y)=(x,y,h)`` 的原始切向量为
    ``Tx=(1,0,gx)``、``Ty=(0,1,gy)``；归一化并 Gram-Schmidt 后输出
    ``(Tmap, Bmap, Nmap)``，每个为 HxWx3 嵌套列表。
    """
    if _np is not None:
        GX = _np.asarray(gx, dtype=float)
        GY = _np.asarray(gy, dtype=float)
        if GX.ndim != 2 or GX.shape != GY.shape:
            raise InvalidShapeError("gx/gy 必须是同形二维场")
        H, W = GX.shape
        Traw = _np.stack([_np.ones_like(GX), _np.zeros_like(GX), GX], axis=-1)
        Braw = _np.stack([_np.zeros_like(GX), _np.ones_like(GX), GY], axis=-1)
        T = Traw / _np.linalg.norm(Traw, axis=-1, keepdims=True)
        N = _np.stack([-GX, -GY, _np.ones_like(GX)], axis=-1)
        N = N / _np.linalg.norm(N, axis=-1, keepdims=True)
        # B = normalize(Braw - T*dot(Braw,T))，再对 N 正交化一次
        dotBT = _np.sum(Braw * T, axis=-1, keepdims=True)
        B = Braw - dotBT * T
        B = B / _np.linalg.norm(B, axis=-1, keepdims=True)
        dotBN = _np.sum(B * N, axis=-1, keepdims=True)
        B = B - dotBN * N
        B = B / _np.linalg.norm(B, axis=-1, keepdims=True)
        return T.tolist(), B.tolist(), N.tolist()

    H, W = len(gx), len(gx[0])
    Tmap = [[None] * W for _ in range(H)]
    Bmap = [[None] * W for _ in range(H)]
    Nmap = [[None] * W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            gxv, gyv = gx[y][x], gy[y][x]
            # 与 numpy 分支一致：T 取自参数切向量 Tx=(1,0,gx)，
            # 始终保有非零 x 分量，不会与 Ty 退化共线。
            tn = math.sqrt(1.0 + gxv * gxv)
            T = (1.0 / tn, 0.0, gxv / tn)
            nn = math.sqrt(gxv * gxv + gyv * gyv + 1.0)
            N = (-gxv / nn, -gyv / nn, 1.0 / nn)
            Braw = (0.0, 1.0, gyv)
            dt = sum(Braw[k] * T[k] for k in range(3))
            B = [Braw[k] - dt * T[k] for k in range(3)]
            dn = sum(B[k] * N[k] for k in range(3))
            B = [B[k] - dn * N[k] for k in range(3)]
            bn = math.sqrt(sum(c * c for c in B))
            if bn < 1e-12:  # 极端退化时用叉积兜底
                B = [N[1] * T[2] - N[2] * T[1],
                     N[2] * T[0] - N[0] * T[2],
                     N[0] * T[1] - N[1] * T[0]]
                bn = math.sqrt(sum(c * c for c in B))
            B = tuple(c / bn for c in B)
            Tmap[y][x] = list(T)
            Bmap[y][x] = list(B)
            Nmap[y][x] = list(N)
    return Tmap, Bmap, Nmap


tangent_frame_from_gradients = frame_from_gradients


def to_tangent_space(n, T, B, N):
    """世界/对象空间向量 → 切线空间分量 ``(dot(n,T), dot(n,B), dot(n,N))``。

    ``n`` 既可以是单个三维向量，也可以是 HxWx3 嵌套图。
    """
    Tv, Bv, Nv = _prepare_basis(T, B, N)
    if _looks_like_vector(n):
        nv = _finite_vec3(n, "n")
        return (_dot(nv, Tv), _dot(nv, Bv), _dot(nv, Nv))
    return [[[_dot(px, Tv), _dot(px, Bv), _dot(px, Nv)] for px in row]
            for row in n]


def from_tangent_space(nt, T, B, N):
    """切线空间分量 → 世界/对象空间向量 ``nx*T + ny*B + nz*N``。

    ``nt`` 既可以是单个三维向量，也可以是 HxWx3 嵌套图。
    """
    Tv, Bv, Nv = _prepare_basis(T, B, N)
    if _looks_like_vector(nt):
        v = _finite_vec3(nt, "nt")
        return tuple(v[0] * Tv[k] + v[1] * Bv[k] + v[2] * Nv[k]
                     for k in range(3))
    out = []
    for row in nt:
        orow = []
        for nt_yx in row:
            orow.append([
                nt_yx[0] * Tv[k] + nt_yx[1] * Bv[k] + nt_yx[2] * Nv[k]
                for k in range(3)])
        out.append(orow)
    return out


def _looks_like_vector(v):
    try:
        return len(v) == 3 and not hasattr(v[0], "__len__")
    except (TypeError, IndexError):
        return False


def _prepare_basis(T, B, N):
    if _looks_like_vector(T) and _looks_like_vector(B) and _looks_like_vector(N):
        return tuple(_finite_vec3(T, "T")), tuple(_finite_vec3(B, "B")), \
            tuple(_finite_vec3(N, "N"))
    # 已经是 (T,B,N) 三元组形式
    if B is None and N is None and isinstance(T, tuple) and len(T) == 3:
        return T[0], T[1], T[2]
    raise InvalidParameterError("需要三个三维基向量 T/B/N")


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


to_world_space = from_tangent_space
tangent_to_world = from_tangent_space
world_to_tangent = to_tangent_space


# ---------------------------------------------------------------------------
# 解析球面辅助与角度误差
# ---------------------------------------------------------------------------
def sphere_height(size=64, radius=8.0, half_width=1.0):
    """生成球冠高度场 ``h=sqrt(R^2-x^2-y^2)``，坐标跨度 [-half_width, half_width]。

    返回 ``(height, jacobian)``：网格按像素索引排列，每个像素对应的物理
    平面步长为 ``2*half_width/(size-1)``，故索引→物理坐标的雅可比为
    ``diag(step, step)``。需要旋转 UV 时对该雅可比左乘旋转即可。
    """
    if size < 2:
        raise InvalidParameterError("size 至少为 2")
    step = 2.0 * half_width / (size - 1)
    h = [[0.0] * size for _ in range(size)]
    for j in range(size):
        y = -half_width + j * step
        for i in range(size):
            x = -half_width + i * step
            inside = radius * radius - x * x - y * y
            h[j][i] = math.sqrt(max(inside, 0.0))
    return h, ((step, 0.0), (0.0, step))


def analytic_sphere_normals(size=64, radius=8.0, half_width=1.0,
                            jacobian=None):
    """返回与 :func:`sphere_height` 对应的解析单位球面法线 HxWx3。

    若提供 ``jacobian``（索引/UV→物理平面，可含旋转缩放），网格点坐标按
    ``p0 + J*(i,j)`` 计算；默认使用与 :func:`sphere_height` 相同的轴对齐网格。
    """
    step = 2.0 * half_width / (size - 1)
    if jacobian is None:
        j00, j01, j10, j11 = step, 0.0, 0.0, step
    else:
        j = _coerce_jacobian(jacobian)
        if _np is not None and isinstance(j, _np.ndarray):
            j00, j01, j10, j11 = float(j[0, 0]), float(j[0, 1]), \
                float(j[1, 0]), float(j[1, 1])
        else:
            j00, j01 = j[0][0], j[0][1]
            j10, j11 = j[1][0], j[1][1]
    out = [[None] * size for _ in range(size)]
    # 让坐标关于 0 对称：J 作用在 (i-c, j-c) 上
    c = (size - 1) / 2.0
    for row in range(size):
        for col in range(size):
            u = col - c
            v = row - c
            x = j00 * u + j01 * v
            y = j10 * u + j11 * v
            z2 = radius * radius - x * x - y * y
            z = math.sqrt(max(z2, 0.0)) if z2 > 0 else 0.0
            out[row][col] = list(_normalize3(x, y, z if z > 0 else 1e-9))
    return out


def angle_between(a, b):
    """两个三维向量的夹角（度）。"""
    av = _finite_vec3(a, "a")
    bv = _finite_vec3(b, "b")
    na = math.sqrt(sum(c * c for c in av))
    nb = math.sqrt(sum(c * c for c in bv))
    if na < 1e-12 or nb < 1e-12:
        raise InvalidParameterError("零向量无法计算夹角")
    cosv = _dot(av, bv) / (na * nb)
    cosv = min(1.0, max(-1.0, cosv))
    return math.degrees(math.acos(cosv))


def mean_angle_error(a_map, b_map):
    """两个 HxWx3 法线图之间的平均夹角误差（度），自动单位化。"""
    if _np is not None:
        A = _np.asarray(a_map, dtype=float)
        B = _np.asarray(b_map, dtype=float)
        if A.shape != B.shape or A.ndim != 3:
            raise InvalidShapeError("两个法线图形状必须一致且为 HxWx3")
        an = A / _np.maximum(_np.linalg.norm(A, axis=-1), 1e-12)[..., None]
        bn = B / _np.maximum(_np.linalg.norm(B, axis=-1), 1e-12)[..., None]
        cosv = _np.sum(an * bn, axis=-1)
        cosv = _np.clip(cosv, -1.0, 1.0)
        return float(_np.degrees(_np.arccos(cosv)).mean())
    H = len(a_map)
    W = len(a_map[0])
    total = 0.0
    cnt = 0
    for y in range(H):
        for x in range(W):
            total += angle_between(a_map[y][x], b_map[y][x])
            cnt += 1
    return total / max(cnt, 1)
