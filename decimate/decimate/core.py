"""二次误差网格简化（QEM edge collapse）。

纯标准库实现，线程安全：:meth:`Decimate.simplify` 的全部工作结构都是
函数内局部变量，实例上仅保存最近一次 Hausdorff 误差上界，且受锁保护。

主数据结构
----------
- 堆：保存当前有合法候选点的边，按 QEM 误差弹出，惰性失效。
- pending 集合：暂无候选点（受包围盒约束）或暂时非法（法向翻转/重复面）
  的边。每次成功坍缩后统一重算，使邻域变化后这些边能够"复活"。
"""

import heapq
import math
import threading

__all__ = [
    "Decimate",
    "UnreachableTargetError",
    "ErrorBoundExceededError",
]

# 包围盒容差（绝对误差）。
_BBOX_TOL = 1e-9
# 法向翻转判定阈值（cos 夹角）。
_FLIP_EPS = 1e-10


class UnreachableTargetError(RuntimeError):
    """拓扑/几何约束下无法把面数降到目标值。"""


class ErrorBoundExceededError(RuntimeError):
    """继续坍缩将使 Hausdorff 误差上界超过给定阈值。"""


def _is_number(x):
    return isinstance(x, (int, float))


class Decimate:
    """二次误差度量（Garland-Heckbert）网格简化器。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._haus = 0.0

    # ------------------------------------------------------------------ API
    def hausdorff(self):
        """返回最近一次 simplify 的 Hausdorff 距离保守上界（float, >= 0）。"""
        with self._lock:
            return self._haus

    def simplify(self, verts, faces, target, max_error=None, **opts):
        """简化三角网格。

        参数
        ----
        verts: 顶点序列，每项为 3 个有限数。
        faces: 三角形序列，每项为 3 个互不相同的顶点索引。
        target: 目标面数，正整数；结果面数 <= target。
        max_error: 可选，Hausdorff 距离阈值（非负有限数）。也接受
            ``threshold`` / ``hausdorff`` 关键字别名。

        返回
        ----
        (out_verts, out_faces)，均为 list，顶点为三元组，面为三元组。
        输入不会被修改；相同输入产生完全一致的输出。

        异常
        ----
        TypeError: 目标非整数等类型错误。
        ValueError: 非有限坐标、非法索引、退化输入面、空网格、非法目标等。
        UnreachableTargetError: 合法坍缩无法达到目标面数。
        ErrorBoundExceededError: 达到目标前误差阈值已被突破。
        """
        # ---- 阈值别名
        alias = None
        for key in ("threshold", "hausdorff"):
            if key in opts:
                alias = opts.pop(key)
        if opts:
            raise TypeError("unknown keyword argument: %r" % next(iter(opts)))
        if max_error is None:
            max_error = alias
        elif alias is not None:
            raise ValueError("conflicting error threshold arguments")

        # ---- 参数校验
        if isinstance(target, bool) or not isinstance(target, int):
            raise TypeError("target must be an int, got %r" % type(target).__name__)
        if target < 1:
            raise ValueError("target must be >= 1, got %r" % target)
        if max_error is not None:
            if not _is_number(max_error) or not math.isfinite(max_error):
                raise ValueError("max_error must be a finite number")
            if max_error < 0.0:
                raise ValueError("max_error must be >= 0")

        pv = self._validate_verts(verts)
        flist = self._validate_faces(faces, len(pv))

        # 复制一份顶点，确保不修改调用方数据。
        pv = [list(p) for p in pv]
        nverts = len(pv)

        # ---- 标记零面积面（视为已死，不参与计数与 Q 贡献，输出时剔除）
        fdead = bytearray(len(flist))
        nfaces = len(flist)
        for fid, (a, b, c) in enumerate(flist):
            if self._cross(pv[a], pv[b], pv[c]) == (0.0, 0.0, 0.0):
                fdead[fid] = 1
                nfaces -= 1
        if nfaces == 0:
            raise UnreachableTargetError("mesh contains no non-degenerate triangles")

        # ---- 包围盒极值与计数
        emin = [min(p[a] for p in pv) for a in range(3)]
        emax = [max(p[a] for p in pv) for a in range(3)]
        cmin = [0, 0, 0]
        cmax = [0, 0, 0]
        for p in pv:
            for a in range(3):
                if p[a] == emin[a]:
                    cmin[a] += 1
                if p[a] == emax[a]:
                    cmax[a] += 1

        # ---- 二次曲面 Q（对称 4x4，存 10 个分量）
        quad = [[0.0] * 10 for _ in range(nverts)]
        vf = [set() for _ in range(nverts)]
        adj = [set() for _ in range(nverts)]
        for fid, (a, b, c) in enumerate(flist):
            vf[a].add(fid)
            vf[b].add(fid)
            vf[c].add(fid)
            adj[a].update((b, c))
            adj[b].update((a, c))
            adj[c].update((a, b))
            if not fdead[fid]:
                self._add_plane(quad[a], quad[b], quad[c], pv[a], pv[b], pv[c])

        dev = [0.0] * nverts          # 代表点到其原始顶点的最大偏差
        alive = bytearray([1]) * nverts

        # ---- 边表（eid 与 ekey/evalid/edata 平行索引）
        edge_of = {}                  # (min,max) -> eid
        ekey = []                     # eid -> (u,v)
        evalid = bytearray()          # eid -> 1/0
        edata = []                    # eid -> (point, err, newdev) | None
        estamp = []                   # eid -> 上次重算时 (gu, gv, bver)
        heap = []
        pending = set()               # 暂不可入堆、需要后续重算的 eid

        # 顶点代际戳：顶点自身位置或其邻域发生坍缩时自增。
        gen = [0] * nverts
        # 全局包围盒计数版本：每次坍缩都自增（远处极值持有者坍缩也会
        # 改变全局计数，代际戳覆盖不到这种情况）。
        bver = 0

        # 当前所有存活面的（排序顶点）集合，用于精确的重复面检测。
        face_keys = set()
        for fid, (a, b, c) in enumerate(flist):
            if not fdead[fid]:
                face_keys.add(tuple(sorted((a, b, c))))

        def make_edge(a, b):
            key = (a, b) if a < b else (b, a)
            existing = edge_of.get(key)
            if existing is not None:
                return existing, key
            eid = len(ekey)
            ekey.append(key)
            evalid.append(1)
            edge_of[key] = eid
            edata.append(None)
            estamp.append((-1, -1, -1))
            return eid, key

        def push_edge(eid):
            """（重新）计算候选点：有则入堆，无则放入 pending。"""
            a, b = ekey[eid]
            cand = self._edge_candidate(
                a, b, pv, quad, dev, emin, emax, cmin, cmax, _BBOX_TOL
            )
            edata[eid] = cand
            estamp[eid] = (gen[a], gen[b], bver)
            evalid[eid] = 1
            if cand is not None:
                pending.discard(eid)
                heapq.heappush(heap, (cand[1], eid))
            else:
                pending.add(eid)

        for (a, b, c) in flist:
            for x, y in ((a, b), (b, c), (a, c)):
                eid, _ = make_edge(x, y)
                push_edge(eid)

        haus = 0.0
        blocked_any = False
        # 停滞保护：没有坍缩的空转轮次上界。pending 在每轮被整体重算，
        # 若邻域不再变化，重算结果也不会变化，故必然终止。
        max_idle = max(16, len(ekey))
        idle_rounds = 0

        # ---- 坍缩主循环
        while nfaces > target:
            if not heap:
                if not pending or idle_rounds >= max_idle:
                    break
                idle_rounds += 1
                progressed = False
                for eid in list(pending):
                    if not evalid[eid] or edge_of.get(ekey[eid]) != eid:
                        pending.discard(eid)
                        continue
                    u, v = ekey[eid]
                    if not alive[u] or not alive[v]:
                        pending.discard(eid)
                        continue
                    # 端点代际与计数版本都未变则结果不会变化，跳过。
                    if estamp[eid] == (gen[u], gen[v], bver):
                        continue
                    push_edge(eid)
                    progressed = True
                if not progressed:
                    break
                continue

            err, eid = heapq.heappop(heap)
            if not evalid[eid] or edge_of.get(ekey[eid]) != eid:
                continue
            u, v = ekey[eid]
            if not alive[u] or not alive[v]:
                continue
            cand = edata[eid]
            if cand is None or eid in pending:
                continue
            point, eerr, newdev = cand
            # 候选点可能在计数变化前缓存：出堆时用当前包围盒计数复核，
            # 不再合法则挂到 pending 等待按新计数重算。
            if not self._bbox_feasible(point, u, v, pv, emin, emax,
                                       cmin, cmax, _BBOX_TOL):
                edata[eid] = None
                pending.add(eid)
                continue
            if max_error is not None and newdev > max_error + _BBOX_TOL:
                # 偏差只增不减，此边永远不会合法：直接移出边表。
                blocked_any = True
                evalid[eid] = 0
                edge_of.pop(ekey[eid], None)
                pending.discard(eid)
                continue

            legal, shared, redirect = self._check_collapse(
                u, v, point, flist, fdead, vf, pv, face_keys
            )
            if not legal:
                # 邻域后续坍缩可能消除冲突，挂到 pending 等待重算。
                pending.add(eid)
                edata[eid] = None
                continue

            # ---- 失效所有触及 u/v 的边（其余边几何不变，保留）
            touched = set()
            for x in adj[u]:
                touched.add((u, x) if u < x else (x, u))
            for x in adj[v]:
                if x != u:
                    touched.add((v, x) if v < x else (x, v))
            for ek in touched:
                old = edge_of.pop(ek, None)
                if old is not None:
                    evalid[old] = 0
                    pending.discard(old)

            # ---- 面结构更新
            for fid in shared:
                fdead[fid] = 1
                nfaces -= 1
                a, b, c = flist[fid]
                face_keys.discard(tuple(sorted((a, b, c))))
                if a in (u, v) and b in (u, v):
                    k = c
                elif a in (u, v) and c in (u, v):
                    k = b
                else:
                    k = a
                vf[k].discard(fid)
            for fid, triple in redirect:
                face_keys.discard(tuple(sorted(flist[fid])))
                flist[fid] = triple
                face_keys.add(tuple(sorted(triple)))
            ring = vf[u] | vf[v]
            vf[u] = {fid for fid in ring if not fdead[fid]}
            vf[v].clear()

            # ---- 顶点合并到 u
            qu, qv = quad[u], quad[v]
            for i in range(10):
                qu[i] += qv[i]
            quad[v] = None
            pu_old, pv_old = pv[u], pv[v]
            pv[u] = list(point)
            pv[v] = None
            dev[u] = newdev
            if newdev > haus:
                haus = newdev

            # ---- 包围盒计数（仅当计数实际变化时提升 bver）
            counts_changed = False
            for a in range(3):
                before_min, before_max = cmin[a], cmax[a]
                cmin[a] -= (pu_old[a] == emin[a]) + (pv_old[a] == emin[a])
                cmax[a] -= (pu_old[a] == emax[a]) + (pv_old[a] == emax[a])
                if point[a] == emin[a]:
                    cmin[a] += 1
                if point[a] == emax[a]:
                    cmax[a] += 1
                if cmin[a] != before_min or cmax[a] != before_max:
                    counts_changed = True

            # ---- 邻接合并
            for x in adj[v]:
                if x != u:
                    adj[x].discard(v)
                    adj[x].add(u)
                    adj[u].add(x)
            adj[u].discard(v)
            adj[v].clear()
            alive[v] = 0

            # ---- 代际戳：u 及其邻居的局部几何已变
            gen[u] += 1
            for x in adj[u]:
                gen[x] += 1
            if counts_changed:
                bver += 1

            # ---- 重建 u 环上的边（排序保证确定性；强制重算）
            for x in sorted(adj[u]):
                eid2, _ = make_edge(u, x)
                push_edge(eid2)

            # 发生了真实坍缩，空转计数清零。
            idle_rounds = 0

        if nfaces > target:
            if blocked_any:
                raise ErrorBoundExceededError(
                    "Hausdorff threshold %.6g blocks further collapse "
                    "(faces=%d, target=%d)" % (max_error, nfaces, target)
                )
            raise UnreachableTargetError(
                "cannot reduce below %d faces (target=%d)" % (nfaces, target)
            )

        # ---- 组装输出（剔除已死面，压缩顶点编号）
        remap = [-1] * nverts
        out_verts = []
        for old in range(nverts):
            if pv[old] is not None:
                remap[old] = len(out_verts)
                out_verts.append((pv[old][0], pv[old][1], pv[old][2]))
        out_faces = []
        seen = set()
        for fid, tri in enumerate(flist):
            if fdead[fid]:
                continue
            t = (remap[tri[0]], remap[tri[1]], remap[tri[2]])
            if len(set(t)) != 3 or t in seen:
                # 理论上不会发生，兜底保证无退化/重复。
                continue
            seen.add(t)
            out_faces.append(t)
        if not out_faces:
            raise UnreachableTargetError("no valid triangle remains after simplification")

        self._verify_bbox(pv, emin, emax)

        with self._lock:
            self._haus = float(haus)
        return out_verts, out_faces

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _cross(pa, pb, pc):
        ux, uy, uz = pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]
        vx, vy, vz = pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]
        return (
            uy * vz - uz * vy,
            uz * vx - ux * vz,
            ux * vy - uy * vx,
        )

    @staticmethod
    def _validate_verts(verts):
        pv = []
        try:
            iterator = iter(verts)
        except TypeError as exc:
            raise TypeError("verts must be a sequence of 3-tuples") from exc
        for vi, p in enumerate(iterator):
            if not hasattr(p, "__len__") or len(p) != 3:
                raise ValueError("vertex %d must have exactly 3 components" % vi)
            row = []
            for comp in p:
                if not _is_number(comp):
                    raise TypeError(
                        "vertex %d contains non-numeric component %r" % (vi, comp)
                    )
                if not math.isfinite(comp):
                    raise ValueError("vertex %d contains non-finite component" % vi)
                row.append(float(comp))
            pv.append((row[0], row[1], row[2]))
        if not pv:
            raise ValueError("verts must not be empty")
        return pv

    @staticmethod
    def _validate_faces(faces, nverts):
        flist = []
        try:
            iterator = iter(faces)
        except TypeError as exc:
            raise TypeError("faces must be a sequence of triples") from exc
        for fi, face in enumerate(iterator):
            if not hasattr(face, "__len__") or len(face) != 3:
                raise ValueError("face %d must have exactly 3 indices" % fi)
            tri = []
            for idx in face:
                if isinstance(idx, bool) or not isinstance(idx, int):
                    raise TypeError("face %d contains non-integer index" % fi)
                if idx < 0 or idx >= nverts:
                    raise ValueError(
                        "face %d references out-of-range vertex %d" % (fi, idx)
                    )
                tri.append(idx)
            if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                raise ValueError("face %d is degenerate (repeated index)" % fi)
            flist.append((tri[0], tri[1], tri[2]))
        if not flist:
            raise ValueError("faces must not be empty")
        return flist

    @staticmethod
    def _add_plane(qa, qb, qc, pa, pb, pc):
        nx, ny, nz = Decimate._cross(pa, pb, pc)
        norm = math.sqrt(nx * nx + ny * ny + nz * nz)
        if norm == 0.0:
            return  # 零面积面，不贡献二次曲面
        nx /= norm
        ny /= norm
        nz /= norm
        d = -(nx * pa[0] + ny * pa[1] + nz * pa[2])
        m = (
            nx * nx, nx * ny, nx * nz, nx * d,
            ny * ny, ny * nz, ny * d,
            nz * nz, nz * d,
            d * d,
        )
        for q in (qa, qb, qc):
            for i in range(10):
                q[i] += m[i]

    @staticmethod
    def _quad_error(q, x, y, z):
        return (
            q[0] * x * x + q[4] * y * y + q[7] * z * z
            + 2.0 * (
                q[1] * x * y + q[2] * x * z + q[5] * y * z
                + q[3] * x + q[6] * y + q[8] * z
            )
            + q[9]
        )

    @staticmethod
    def _solve_optimal(q):
        """求 Q3 x = b 的最优顶点；奇异或非有限时返回 None。"""
        a00, a01, a02 = q[0], q[1], q[2]
        a11, a12 = q[4], q[5]
        a22 = q[7]
        b0, b1, b2 = -q[3], -q[6], -q[8]
        c11 = a11 * a22 - a12 * a12
        c12 = a01 * a22 - a12 * a02
        c13 = a01 * a12 - a11 * a02
        det = a00 * c11 - a01 * c12 + a02 * c13
        scale = (1.0 + abs(a00)) * (1.0 + abs(a11)) * (1.0 + abs(a22))
        if abs(det) <= 1e-12 * scale:
            return None
        dx = b0 * c11 - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a12 - a11 * b2)
        dy = a00 * (b1 * a22 - a12 * b2) - b0 * c12 + a02 * (a01 * b2 - b1 * a02)
        dz = a00 * (a11 * b2 - b1 * a12) - a01 * (a01 * b2 - b1 * a02) + b0 * c13
        x, y, z = dx / det, dy / det, dz / det
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return None
        return x, y, z

    @staticmethod
    def _bbox_feasible(p, u, v, pv, emin, emax, cmin, cmax, tol):
        """移除 u,v 并以 p 作为合并点后，六个包围盒极值是否仍有人持有。"""
        pu, pv_v = pv[u], pv[v]
        for a in range(3):
            nmin = (
                cmin[a]
                - (pu[a] == emin[a])
                - (pv_v[a] == emin[a])
                + (1 if p[a] <= emin[a] + tol else 0)
            )
            nmax = (
                cmax[a]
                - (pu[a] == emax[a])
                - (pv_v[a] == emax[a])
                + (1 if p[a] >= emax[a] - tol else 0)
            )
            if nmin < 1 or nmax < 1:
                return False
        return True

    def _edge_candidate(self, u, v, pv, quad, dev, emin, emax, cmin, cmax, tol):
        pu, pv_v = pv[u], pv[v]
        q = [quad[u][i] + quad[v][i] for i in range(10)]
        candidates = []
        opt = self._solve_optimal(q)
        if opt is not None:
            candidates.append(opt)
        candidates.append((
            (pu[0] + pv_v[0]) * 0.5,
            (pu[1] + pv_v[1]) * 0.5,
            (pu[2] + pv_v[2]) * 0.5,
        ))
        candidates.append((pu[0], pu[1], pu[2]))
        candidates.append((pv_v[0], pv_v[1], pv_v[2]))

        best = None
        for raw in candidates:
            # 钳位到包围盒：任何越界（含 1e-15 级舍入）都精确吸附到边界；
            # tol 邻域内的内部点同样吸附，保证极值计数稳定。
            p = []
            for a in range(3):
                t = raw[a]
                if t <= emin[a] + tol:
                    t = emin[a]
                elif t >= emax[a] - tol:
                    t = emax[a]
                p.append(t)
            p = (p[0], p[1], p[2])
            # 包围盒极值计数：移除 u,v 后加入 p，六个极值仍须有人持有。
            if not self._bbox_feasible(p, u, v, pv, emin, emax,
                                       cmin, cmax, tol):
                continue
            err = self._quad_error(q, p[0], p[1], p[2])
            if -1e-12 < err < 0.0:
                err = 0.0
            if best is None or err < best[1]:
                best = (p, err)
        if best is None:
            return None
        p, err = best
        # 距离只对最终胜出的候选点计算一次。
        du = math.dist(p, pu)
        dv = math.dist(p, pv_v)
        newdev = max(dev[u], dev[v], du, dv)
        return (p, err, newdev)

    def _check_collapse(self, u, v, point, flist, fdead, vf, pv, face_keys):
        """返回 (legal, shared_fids, redirect[(fid,new_triple)])。"""
        shared = []
        redirect = []
        triples = set()
        # 环内现存面自身的 key，全局查重时排除（只含一个端点的面坍缩后
        # 索引表示不变，会命中自己的旧 key）。
        ring_old = set()
        for fid in vf[u] | vf[v]:
            if not fdead[fid]:
                ring_old.add(tuple(sorted(flist[fid])))
        for fid in vf[u] | vf[v]:
            if fdead[fid]:
                continue
            a, b, c = flist[fid]
            has_u = a == u or b == u or c == u
            has_v = a == v or b == v or c == v
            if has_u and has_v:
                shared.append(fid)
                continue
            na = point if a in (u, v) else pv[a]
            nb = point if b in (u, v) else pv[b]
            nc = point if c in (u, v) else pv[c]
            nnx, nny, nnz = self._cross(na, nb, nc)
            if nnx == 0.0 and nny == 0.0 and nnz == 0.0:
                return False, (), ()
            # 法向翻转检查（原面零面积则跳过）。
            onx, ony, onz = self._cross(pv[a], pv[b], pv[c])
            if not (onx == 0.0 and ony == 0.0 and onz == 0.0):
                dot = onx * nnx + ony * nny + onz * nnz
                if dot < -_FLIP_EPS * math.sqrt(
                    (onx * onx + ony * ony + onz * onz)
                    * (nnx * nnx + nny * nny + nnz * nnz)
                ):
                    return False, (), ()
            rep = (
                u if a in (u, v) else a,
                u if b in (u, v) else b,
                u if c in (u, v) else c,
            )
            key = tuple(sorted(rep))
            # 环内重复 或 与环外现存面重复，均拒绝。
            if key in triples or (key in face_keys and key not in ring_old):
                return False, (), ()
            triples.add(key)
            redirect.append((fid, list(rep)))
        return True, shared, redirect

    @staticmethod
    def _verify_bbox(pv, emin, emax):
        alive_pts = [p for p in pv if p is not None]
        for a in range(3):
            lo = min(p[a] for p in alive_pts)
            hi = max(p[a] for p in alive_pts)
            if abs(lo - emin[a]) > _BBOX_TOL or abs(hi - emax[a]) > _BBOX_TOL:
                raise UnreachableTargetError(
                    "bounding box preservation violated on axis %d" % a
                )
