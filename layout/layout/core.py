"""Unicode 断行与字距排版核心。

简化 UAX#14 断行规则：
- 行首禁则：闭合/收尾类标点不得出现在行首；
- 行尾禁则：开括号类字符不得出现在行尾；
- 空格两侧、连字符之后、CJK 字符相邻处允许断行；
- 软连字符 U+00AD 标记允许的非强制断词点：仅在该处断行时渲染为可见
  连字符，否则零宽且不出现在结果中（连字符断词只发生在允许点）。

字距（tracking）在相邻字形之间线性叠加；kerning 对优先于字距。
结果中每个字形的宽度 = 原宽度 + 其后随间距（行末字形不加后随间距），
因此每行所有字形宽度之和即为该行宽度，保证 <= max_width（容差 1e-9）。

错误分类（均可按 ValueError 捕获，类型可区分）：
- InvalidTypeError   输入类型非法（同时是 TypeError 与 ValueError）
- NonFiniteError     输入含非有限数值（NaN / ±Inf）
- InvalidWidthError  max_width 非正数
- InvalidTrackingError  tracking 为负
- InvalidGlyphError  字形结构或字宽非法
- GlyphTooWideError  单个字形超过行宽上限，宽度约束不可满足
"""

from __future__ import annotations

import math

__all__ = [
    "TextLayout",
    "LayoutError",
    "InvalidTypeError",
    "NonFiniteError",
    "InvalidWidthError",
    "InvalidTrackingError",
    "InvalidGlyphError",
    "GlyphTooWideError",
]

_INF = float("inf")
_SHY = "­"  # SOFT HYPHEN：允许的非强制断词点
_HYPHEN = "-"


class LayoutError(ValueError):
    """排版相关错误的基类；同时是 ValueError。"""


class InvalidTypeError(TypeError, ValueError):
    """输入类型非法（可同时按 TypeError / ValueError 捕获）。"""


class NonFiniteError(LayoutError):
    """输入包含非有限数值（NaN 或 ±Inf）。"""


class InvalidWidthError(LayoutError):
    """行宽上限非法（非正数）。"""


class InvalidTrackingError(LayoutError):
    """字距非法（负数）。"""


class InvalidGlyphError(LayoutError):
    """字形结构或字宽非法。"""


class GlyphTooWideError(LayoutError):
    """单个字形的宽度超过行宽上限，无法满足宽度约束。"""


# 行首禁则字符（简化 UAX#14：闭合标点、句读、叹问等不得位于行首）
_LINE_START_FORBIDDEN = frozenset(
    ".,;:!?%)]}"
    "，。、；：？！…·％"
    "）】》」』〉〕〙〛"
    "﹚﹜﹞］｝｠"
)

# 行尾禁则字符（开括号类不得位于行尾）
_LINE_END_FORBIDDEN = frozenset(
    "([{"
    "（【《「『〈〔〘〚"
    "﹙﹛﹝［｛｟"
)

# 连字符/破折号：允许在其后断行
_HYPHENS = frozenset("-‐‑‒–—―−﹣－")


def _is_cjk(o):
    """码点 o 是否属于 CJK 表意/假名/谚文/全角范围（这些字符相邻处可断行）。"""
    return (
        0x2E80 <= o <= 0x9FFF
        or 0xAC00 <= o <= 0xD7AF
        or 0xF900 <= o <= 0xFAFF
        or 0xFF00 <= o <= 0xFFEF
        or 0x20000 <= o <= 0x2FA1F
        or 0x30000 <= o <= 0x323AF
    )


def _to_float(value, what):
    """把输入转成 float；类型不合法时抛出 InvalidTypeError。"""
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise InvalidTypeError("%s必须是实数，而不是 %s" % (what, type(value).__name__))
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        raise InvalidTypeError(
            "%s必须是实数，而不是 %s" % (what, type(value).__name__)
        ) from None


def _check_kerning(kerning):
    """校验并规范化 kerning 表，返回新 dict（构造后不受外部可变对象影响）。"""
    if kerning is None:
        return {}
    items = kerning.items() if hasattr(kerning, "items") else kerning
    try:
        iterator = iter(items)
    except TypeError:
        raise InvalidTypeError(
            "kerning 必须是映射或 (对, 值) / (字符1, 字符2, 值) 序列"
        ) from None
    out = {}
    for item in iterator:
        try:
            if len(item) == 3:
                a, b, v = item
            else:
                (a, b), v = item
        except (TypeError, ValueError):
            raise InvalidTypeError(
                "kerning 项必须是 ((字符1, 字符2), 值) 或 (字符1, 字符2, 值)"
            ) from None
        if not isinstance(a, str) or not isinstance(b, str):
            raise InvalidTypeError("kerning 对的字符必须是 str")
        fv = _to_float(v, "kerning 值")
        if not math.isfinite(fv):
            raise NonFiniteError("kerning 值必须是有限数值")
        out[(a, b)] = fv
    return out


class TextLayout:
    """文本排版器：按简化 UAX#14 规则断行，字距线性叠加，kerning 优先。

    线程安全：构造后内部状态不再改变，layout() 只使用局部变量，
    可被多个线程并发调用；也不会修改传入的 glyphs。
    """

    __slots__ = ("_tracking", "_kerning", "_hyphenate")

    def __init__(self, tracking=0.0, kerning=None, hyphenate=True):
        t = _to_float(tracking, "tracking ")
        if not math.isfinite(t):
            raise NonFiniteError("tracking 必须是有限数值")
        if t < 0.0:
            raise InvalidTrackingError("tracking 不能为负数")
        self._tracking = t
        self._kerning = _check_kerning(kerning)
        self._hyphenate = bool(hyphenate)

    def layout(self, glyphs, max_width):
        """把 glyphs（(字符, 宽度) 序列）按 max_width 断行。

        返回 list[list[(char, width)]]；空输入返回 []。
        每行宽度（行内字形结果宽度之和）不超过 max_width（容差 1e-9），
        且除文本本身以禁则字符开头外，没有行以行首禁则字符开头。
        """
        limit_w = _to_float(max_width, "max_width ")
        if not math.isfinite(limit_w):
            raise NonFiniteError("max_width 必须是有限数值")
        if limit_w <= 0.0:
            raise InvalidWidthError("max_width 必须是正数")
        if glyphs is None:
            raise InvalidTypeError("glyphs 不能为 None")

        # 1) 校验并物化输入（不修改原序列）
        chars = []
        widths = []
        c_ap = chars.append
        w_ap = widths.append
        isfinite = math.isfinite
        for g in glyphs:
            if type(g) is tuple and len(g) == 2:
                ch = g[0]
                w = g[1]
            elif type(g) is list and len(g) == 2:
                ch = g[0]
                w = g[1]
            elif isinstance(g, (str, bytes, bytearray)):
                raise InvalidGlyphError("每个字形必须是 (字符, 宽度) 二元组")
            else:
                try:
                    ch, w = g
                except (TypeError, ValueError):
                    raise InvalidGlyphError(
                        "每个字形必须是 (字符, 宽度) 二元组"
                    ) from None
            if not isinstance(ch, str):
                raise InvalidTypeError("字形字符必须是 str")
            if not ch:
                raise InvalidGlyphError("字形字符不能为空串")
            if type(w) is float:
                fw = w
            elif type(w) is int:
                try:
                    fw = float(w)
                except OverflowError:
                    raise NonFiniteError("字形宽度超出可表示范围") from None
            else:
                fw = _to_float(w, "字形宽度")
            if not isfinite(fw):
                raise NonFiniteError("字形宽度必须是有限数值")
            if fw < 0.0:
                raise InvalidGlyphError("字形宽度不能为负")
            c_ap(ch)
            w_ap(fw)

        n = len(chars)
        if n == 0:
            return []

        tracking = self._tracking
        kerning = self._kerning
        hyphenate = self._hyphenate

        # 2) 逐字形分类 + 有效宽度（软连字符不可见时零宽）
        f_forbid = bytearray(n)   # 首字符为行首禁则
        l_forbid = bytearray(n)   # 末字符为行尾禁则
        is_space = bytearray(n)
        is_cjk = bytearray(n)
        is_hyph = bytearray(n)    # 末字符为连字符（其后可断）
        shy = bytearray(n)        # 软连字符（允许的非强制断词点）
        ew = [0.0] * n            # 有效宽度
        lsf = _LINE_START_FORBIDDEN
        lef = _LINE_END_FORBIDDEN
        hyphens = _HYPHENS
        hard = limit_w + 1e-9     # 题面容差
        for i in range(n):
            ch = chars[i]
            first = ch[0]
            last = ch[-1]
            if first in lsf:
                f_forbid[i] = 1
            if last in lef:
                l_forbid[i] = 1
            if last in hyphens:
                is_hyph[i] = 1
            if ch.isspace():
                is_space[i] = 1
            if _is_cjk(ord(first)) or (len(ch) > 1 and _is_cjk(ord(last))):
                is_cjk[i] = 1
            if hyphenate and ch == _SHY:
                shy[i] = 1
            else:
                w = widths[i]
                if w > hard:
                    raise GlyphTooWideError(
                        "字形 %r 的宽度 %r 超过行宽上限 %r，无法满足宽度约束"
                        % (ch, w, limit_w)
                    )
                ew[i] = w

        # 3) 断点可行性：can_break 含词法规则，proh_break 仅禁则规则
        can_break = bytearray(n)
        proh_break = bytearray(n)
        for i in range(n - 1):
            if f_forbid[i + 1] or l_forbid[i]:
                continue
            proh_break[i] = 1
            if (
                shy[i]
                or is_space[i]
                or is_space[i + 1]
                or is_hyph[i]
                or is_cjk[i]
                or is_cjk[i + 1]
            ):
                can_break[i] = 1

        # 4) 后随间距：kerning 对优先于字距；软连字符不可见，两侧间距为零
        gap = [0.0] * n
        if tracking or kerning:
            kget = kerning.get
            for i in range(n - 1):
                if shy[i] or shy[i + 1]:
                    continue
                gap[i] = kget((chars[i], chars[i + 1]), tracking)

        # 5) prev_ok[j]：<= j 且满足禁则的最近断点（应急断行用）
        prev_ok = [0] * n
        ok = 0
        for e in range(1, n):
            if proh_break[e - 1]:
                ok = e
            prev_ok[e] = ok

        # 6) 贪心断行。limit 略小于容差上限，抵消浮点累加误差，
        #    保证调用方对结果求和仍满足 <= max_width + 1e-9。
        limit = hard - 1e-12 * (limit_w if limit_w > 1.0 else 1.0)
        emit_ranges = []
        s = 0
        cur = ew[0]
        last_break = 0          # 最近合法断点位置 b（在 b 之前断），0 表示无
        last_break_shy = False  # 该断点是否断在软连字符之后
        j = 1
        while j < n:
            cur += gap[j - 1] + ew[j]
            if cur <= limit:
                if can_break[j]:
                    if shy[j]:
                        if cur + widths[j] <= limit:
                            last_break = j + 1
                            last_break_shy = True
                    else:
                        last_break = j + 1
                        last_break_shy = False
                j += 1
                continue
            # 溢出：优先最近合法断点，否则应急断行（仍尽量遵守禁则）
            if last_break > s:
                b = last_break
                emit_ranges.append((s, b - 1, last_break_shy))
            else:
                b = prev_ok[j]
                while b > s and shy[b - 1]:
                    b = prev_ok[b - 1]
                if b <= s:
                    b = j
                    if shy[j - 1] and j - 1 > s:
                        b = j - 1
                emit_ranges.append((s, b - 1, False))
            # 从 b 重新累计到 j，并重建该区间内的候选断点
            s = b
            cur = ew[s]
            last_break = 0
            last_break_shy = False
            if can_break[s]:
                if shy[s]:
                    if cur + widths[s] <= limit:
                        last_break = s + 1
                        last_break_shy = True
                else:
                    last_break = s + 1
            k = s + 1
            while k <= j:
                cur += gap[k - 1] + ew[k]
                if can_break[k]:
                    if shy[k]:
                        if cur + widths[k] <= limit:
                            last_break = k + 1
                            last_break_shy = True
                    else:
                        last_break = k + 1
                        last_break_shy = False
                k += 1
            j += 1
        if s < n:
            emit_ranges.append((s, n - 1, False))

        # 7) 生成结果：非行末字形宽度 = 原宽 + 后随间距；
        #    软连字符仅在断行处渲染为可见连字符，否则丢弃。
        out_lines = []
        for s, e, shy_vis in emit_ranges:
            line = []
            ap = line.append
            for k in range(s, e + 1):
                if shy[k]:
                    if k == e and shy_vis:
                        ap((_HYPHEN, widths[k]))
                elif k == e:
                    ap((chars[k], widths[k]))
                else:
                    ap((chars[k], widths[k] + gap[k]))
            if line:
                out_lines.append(line)
        return out_lines
