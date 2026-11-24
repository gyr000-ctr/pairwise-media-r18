"""纹理图集：矩形打包、UV 查询与出血（bleed）支持。

约定
----
- 图集为 ``size x size`` 的正方形像素纹理，原点在左下角，v 轴向上。
- 每个矩形在打包时四周预留 ``pad`` 个像素的出血区；出血区内由调用方
  填充边缘拉伸像素，因此对内层内容矩形的任意采样都不会越过该分配区。
- :meth:`TexAtlas.uv` 返回 ``(u0, v0, u1, v1)``，mip 0 时与打包得到的
  像素矩形严格一致（仅含一次整除浮点化，误差远小于 1e-12）；更高 mip
  级别按半纹素向内收缩，避免多级渐远过滤时采到相邻矩形而产生接缝。

状态模型
--------
打包注册表按 ``(size, pad)`` 配置保存在模块级全局表中，同配置的任意
:class:`TexAtlas` 句柄共享同一份状态（一次 ``add`` 对后续同配置的
``uv`` 立即可见）。注册表与句柄生命周期解耦，通过 :func:`reset` 清空；
测试侧由 ``conftest.py`` 的自动夹具在用例之间调用以获得干净状态。
"""

import math
import threading

__all__ = [
    "TexAtlas",
    "reset",
    "AtlasError",
    "AtlasFull",
    "DuplicateName",
    "UnknownName",
]


class AtlasError(ValueError):
    """图集相关错误的基类（同时是 :class:`ValueError`，便于统一捕获）。"""


class AtlasFull(AtlasError):
    """图集中没有足够空间容纳请求的矩形。"""


class DuplicateName(AtlasError):
    """向图集重复添加同名矩形。"""


class UnknownName(AtlasError, KeyError):
    """查询了图集中不存在的名字。

    同时继承 :class:`KeyError`，因此既能按 ``AtlasError`` 与其它图集错误
    区分，也兼容字典式的 ``KeyError`` 捕获。
    """


def _is_real_integer(v):
    """有限的、整数值的实数（显式拒绝 bool）。"""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float) and math.isfinite(v) and v.is_integer():
        return True
    return False


# 模块级共享注册表：(size, pad) -> _AtlasState。由 reset() 清空。
_STATES = {}
_STATES_GUARD = threading.Lock()


def reset():
    """清空全部配置的共享打包状态。

    主要供测试夹具在用例之间重置；生产代码一般无需调用。
    """
    with _STATES_GUARD:
        _STATES.clear()


class _AtlasState:
    """某个 ``(size, pad)`` 配置下被同配置句柄共享的可变打包状态。"""

    __slots__ = ("size", "pad", "lock", "rects", "shelves", "uv_cache")

    def __init__(self, size, pad):
        self.size = size
        self.pad = pad
        # 可重入锁保护全部可变状态；查询与添加均可在多线程下并发调用。
        self.lock = threading.RLock()
        # name -> (x, y, w, h)，内层内容矩形（不含 padding），写入后不再变更。
        self.rects = {}
        # (货架底边 y, 货架高度, 已用宽度)
        self.shelves = []
        # name -> {mip: (u0, v0, u1, v1)}；条目不可变，故缓存无需失效。
        self.uv_cache = {}

    def place(self, w, h):
        """在锁内选取分配位置，返回内容矩形的左下角像素坐标。"""
        size = self.size
        pad = self.pad
        alloc_w = w + 2 * pad
        alloc_h = h + 2 * pad
        if alloc_w > size or alloc_h > size:
            raise AtlasFull(
                "矩形 %dx%d（含 %d 像素 padding）超出图集尺寸 %d"
                % (w, h, pad, size)
            )

        # 最佳适应：能放下且高度余量最小的货架，减少垂直浪费。
        best = -1
        best_gap = None
        for i, (_, shelf_h, used) in enumerate(self.shelves):
            if alloc_h <= shelf_h and used + alloc_w <= size:
                gap = shelf_h - alloc_h
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best = i

        if best >= 0:
            sy, shelf_h, used = self.shelves[best]
            self.shelves[best] = (sy, shelf_h, used + alloc_w)
            # 左侧留出 pad 像素出血区。
            return used + pad, sy + pad

        # 现有货架放不下，开新货架。
        bottom = sum(shelf_h for _, shelf_h, _ in self.shelves)
        if bottom + alloc_h > size:
            raise AtlasFull(
                "图集剩余空间不足，无法容纳 %dx%d 的矩形" % (w, h)
            )
        self.shelves.append((bottom, alloc_h, alloc_w))
        return pad, bottom + pad


class TexAtlas:
    """货架式（shelf）纹理图集打包器与 UV 查询表。

    同 ``(size, pad)`` 配置的所有实例共享一份模块级打包状态。

    Parameters
    ----------
    size:
        图集边长（像素），必须为正整数。
    pad:
        每个矩形四周预留的出血区宽度（像素），必须为正整数。
    """

    def __init__(self, size, pad):
        if not _is_real_integer(size) or int(size) <= 0:
            raise ValueError("size 必须为正的有限整数")
        if not _is_real_integer(pad) or int(pad) <= 0:
            raise ValueError("pad 必须为正的有限整数")
        key = (int(size), int(pad))
        with _STATES_GUARD:
            state = _STATES.get(key)
            if state is None:
                state = _AtlasState(key[0], key[1])
                _STATES[key] = state
        self._state = state

    # ------------------------------------------------------------------ #
    # 基础属性
    # ------------------------------------------------------------------ #
    @property
    def size(self):
        """图集边长（像素）。"""
        return self._state.size

    @property
    def pad(self):
        """出血区宽度（像素）。"""
        return self._state.pad

    def names(self):
        """返回当前已注册的全部名字（元组，快照）。"""
        with self._state.lock:
            return tuple(self._state.rects)

    def __contains__(self, name):
        with self._state.lock:
            return name in self._state.rects

    def __len__(self):
        with self._state.lock:
            return len(self._state.rects)

    # ------------------------------------------------------------------ #
    # 打包
    # ------------------------------------------------------------------ #
    def add(self, name, w, h):
        """加入一个 ``w x h`` 的内容矩形，返回 ``self`` 以支持链式调用。

        实际占用为 ``(w + 2*pad) x (h + 2*pad)`` 的分配区，内容矩形居中，
        四周的 ``pad`` 像素即出血区。空间不足时抛出 :class:`AtlasFull`，
        重名时抛出 :class:`DuplicateName`。同配置的任意句柄都能查到结果。
        """
        if type(name) is not str or not name:
            raise ValueError("name 必须为非空字符串")
        if not _is_real_integer(w) or int(w) <= 0:
            raise ValueError("w 必须为正的有限整数")
        if not _is_real_integer(h) or int(h) <= 0:
            raise ValueError("h 必须为正的有限整数")
        w, h = int(w), int(h)

        state = self._state
        with state.lock:
            if name in state.rects:
                raise DuplicateName("名字已存在: %r" % name)
            x, y = state.place(w, h)
            state.rects[name] = (x, y, w, h)
        return self

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def rect(self, name):
        """返回名字对应的内容像素矩形 ``(x, y, w, h)``（不含 padding）。"""
        if type(name) is not str:
            raise TypeError("name 必须为字符串")
        state = self._state
        with state.lock:
            entry = state.rects.get(name)
            if entry is None:
                if not name:
                    raise ValueError("name 不能为空字符串")
                raise UnknownName("未知纹理名: %r" % name)
            return entry

    def uv_padded(self, name):
        """返回含 padding 的出血区 UV ``(u0, v0, u1, v1)``。

        出血区完全位于图集内部（打包时已保证），结果再钳制到 ``[0, 1]``，
        因此对出血区的采样绝不会越界到图集之外。
        """
        x, y, w, h = self.rect(name)
        size = self._state.size
        pad = self._state.pad
        px0 = max(0, x - pad)
        py0 = max(0, y - pad)
        px1 = min(size, x + w + pad)
        py1 = min(size, y + h + pad)
        return (px0 / size, py0 / size, px1 / size, py1 / size)

    def uv(self, name, mip=0):
        """返回名字在指定 mip 级别下的 UV 矩形 ``(u0, v0, u1, v1)``。

        - ``mip=0``：``(x/size, y/size, (x+w)/size, (y+h)/size)``，与打包
          结果严格一致（误差 <= 1e-12）。
        - ``mip>0``：矩形按 ``2**-mip`` 缩小，并在四周各收缩半个当前级
          纹素，保证线性/三线性过滤不会跨越矩形边界（避免接缝）。
        - 非法名字抛出 :class:`UnknownName`（``KeyError`` 子类），非法
          mip 抛出 :class:`ValueError`，两类错误可区分。
        """
        if type(name) is not str:
            raise TypeError("name 必须为字符串")
        # 热路径：str + int 是常态，先用精确类型快速判定，避免昂贵的
        # 浮点/有限性检查拖慢上万次查询。
        if type(mip) is int:
            if mip < 0:
                raise ValueError("mip 必须为非负整数")
        elif isinstance(mip, float) and math.isfinite(mip) and mip.is_integer():
            mip = int(mip)
            if mip < 0:
                raise ValueError("mip 必须为非负整数")
        else:
            raise ValueError("mip 必须为有限的非负整数")

        state = self._state
        with state.lock:
            entry = state.rects.get(name)
            if entry is None:
                if not name:
                    raise ValueError("name 不能为空字符串")
                raise UnknownName("未知纹理名: %r" % name)

            bucket = state.uv_cache.get(name)
            if bucket is not None:
                cached = bucket.get(mip)
                if cached is not None:
                    return cached

            x, y, w, h = entry
            size = state.size

            if mip == 0:
                uv_rect = (
                    x / size,
                    y / size,
                    (x + w) / size,
                    (y + h) / size,
                )
            else:
                max_mip = min(w, h).bit_length() - 1
                if mip > max_mip:
                    raise ValueError(
                        "mip=%d 超出纹理 %dx%d 支持的最大级别 %d"
                        % (mip, w, h, max_mip)
                    )
                scale = 2.0 ** (-mip)
                # 半纹素收缩（换算到 mip 0 的像素单位），防止接缝。
                inset = 0.5 * scale
                uv_rect = (
                    (x + inset) / size,
                    (y + inset) / size,
                    (x + w * scale - inset) / size,
                    (y + h * scale - inset) / size,
                )

            if bucket is None:
                bucket = {}
                state.uv_cache[name] = bucket
            bucket[mip] = uv_rect
            return uv_rect
