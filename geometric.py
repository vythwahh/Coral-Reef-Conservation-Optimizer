import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

class GeometricSolver:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def solve(self, c, A, b, constraint_types, is_max=False, plot=True):
        assert len(c) == 2,     "Chỉ dùng cho bài 2 biến!"
        assert A.shape[1] == 2, "Ma trận A phải có đúng 2 cột!"

        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        m = len(b)

        if m == 2:
            return self._solve_two_constraints(c, A, b, constraint_types, is_max, plot)
        else:
            return self._solve_general(c, A, b, constraint_types, is_max, plot, m)

    # ─────────────────────────────────────────────
    # CASE 1: 2 RÀNG BUỘC
    # ─────────────────────────────────────────────

    def _solve_two_constraints(self, c, A, b, constraint_types, is_max, plot):
        lines = []
        for i in range(2):
            lines.append((A[i], b[i]))
        lines.append((np.array([-1.0, 0.0]), 0.0))
        lines.append((np.array([0.0, -1.0]), 0.0))

        candidates = []
        for i, j in combinations(range(4), 2):
            a_i, b_i = lines[i]
            a_j, b_j = lines[j]
            A_pair = np.array([a_i, a_j])
            b_pair = np.array([b_i, b_j])
            try:
                if abs(np.linalg.det(A_pair)) < self.epsilon:
                    continue
                v = np.linalg.solve(A_pair, b_pair)
                if np.all(v >= -self.epsilon):
                    candidates.append(np.maximum(v, 0))
            except np.linalg.LinAlgError:
                continue

        if not candidates:
            return "Infeasible", None, None, []

        feasible = [v for v in candidates
                    if self._is_feasible(v, A, b, constraint_types)]

        if not feasible:
            return "Infeasible", None, None, []

        return self._evaluate_and_plot(
            c, A, b, constraint_types, feasible, is_max, plot,
            title_note="(2 ràng buộc)"
        )

     
    # CASE 2: TỔNG QUÁT
  

    def _solve_general(self, c, A, b, constraint_types, is_max, plot, m):
        vertices = self._find_vertices(A, b, m)
        if not vertices:
            return "Infeasible", None, None, []

        feasible = [v for v in vertices
                    if self._is_feasible(v, A, b, constraint_types)]
        if not feasible:
            return "Infeasible", None, None, []

        return self._evaluate_and_plot(
            c, A, b, constraint_types, feasible, is_max, plot,
            title_note=f"({m} ràng buộc)"
        )

    # ─────────────────────────────────────────────
    # ĐÁNH GIÁ + PHÁT HIỆN VÔ SỐ NGHIỆM + VẼ
    # ─────────────────────────────────────────────

    def _evaluate_and_plot(self, c, A, b, constraint_types,
                           feasible, is_max, plot, title_note):
        history = []
        for v in feasible:
            f_val = float(c @ v)
            history.append({
                "vertex" : v.tolist(),
                "f_value": f_val,
                "message": (
                    f"Đỉnh ({v[0]:.4f}, {v[1]:.4f}): "
                    f"f = {c[0]}×{v[0]:.4f} + {c[1]}×{v[1]:.4f} = {f_val:.4f}"
                )
            })

        best  = max(history, key=lambda h: h["f_value"]) if is_max \
                else min(history, key=lambda h: h["f_value"])
        f_opt = best["f_value"]

        # Tìm tất cả đỉnh đạt giá trị tối ưu  
        optimal_vertices = [
            np.array(h["vertex"]) for h in history
            if abs(h["f_value"] - f_opt) < self.epsilon
        ]

        # Phát hiện vô số nghiệm  
        # Xảy ra khi >= 2 đỉnh cùng đạt f_opt
        # → kiểm tra thêm: vector c có song song với cạnh nối 2 đỉnh đó không
        infinite_solutions = False
        optimal_edge       = None  # cặp đỉnh tạo thành cạnh tối ưu

        if len(optimal_vertices) >= 2:
            for i, j in combinations(range(len(optimal_vertices)), 2):
                vi, vj   = optimal_vertices[i], optimal_vertices[j]
                edge_dir = vj - vi                   # vector cạnh
                # c song song edge_dir ⟺ cross product = 0
                cross = c[0] * edge_dir[1] - c[1] * edge_dir[0]
                if abs(cross) < self.epsilon:
                    infinite_solutions = True
                    optimal_edge       = (vi, vj)
                    break

        # x_opt: trả về đỉnh đầu tiên tối ưu (hoặc midpoint nếu vô số)
        if infinite_solutions and optimal_edge is not None:
            x_opt  = optimal_edge[0]
            status = "Optimal (vô số nghiệm)"
        else:
            x_opt  = optimal_vertices[0]
            status = "Optimal"

        if plot:
            self._plot(
                c, A, b, constraint_types, feasible,
                optimal_vertices, x_opt, f_opt, is_max,
                infinite_solutions, optimal_edge, title_note
            )

        return status, x_opt, f_opt, history

     
    # PRIVATE: TÌM ĐỈNH
     

    def _find_vertices(self, A, b, m):
        A_ext = np.vstack([A, [-1, 0], [0, -1]])
        b_ext = np.append(b, [0, 0])
        n_ext = len(b_ext)

        vertices = []
        for i, j in combinations(range(n_ext), 2):
            A_pair = A_ext[[i, j], :]
            b_pair = b_ext[[i, j]]
            try:
                if abs(np.linalg.det(A_pair)) < self.epsilon:
                    continue
                v = np.linalg.solve(A_pair, b_pair)
                if np.all(v >= -self.epsilon):
                    vertices.append(np.maximum(v, 0))
            except np.linalg.LinAlgError:
                continue
        return vertices

    def _is_feasible(self, v, A, b, constraint_types):
        for i in range(len(b)):
            lhs = A[i] @ v
            if constraint_types[i] == '<=' and lhs > b[i] + self.epsilon:
                return False
            if constraint_types[i] == '>=' and lhs < b[i] - self.epsilon:
                return False
            if constraint_types[i] == '='  and abs(lhs - b[i]) > self.epsilon:
                return False
        return True

    # ─────────────────────────────────────────────
    # PRIVATE: VẼ ĐỒ THỊ
    # ─────────────────────────────────────────────

    def _plot(self, c, A, b, constraint_types, feasible,
              optimal_vertices, x_opt, f_opt, is_max,
              infinite_solutions, optimal_edge, title_note):

        fig, ax = plt.subplots(figsize=(8, 7))

        xs       = [v[0] for v in feasible]
        ys       = [v[1] for v in feasible]
        margin   = max(max(xs + [1]) * 0.4, 1)
        x1_max   = max(xs + [1]) + margin
        x2_max   = max(ys + [1]) + margin
        x1_range = np.linspace(0, x1_max, 400)

        # ── Vẽ đường ràng buộc ───────────────────────────────────
        colors = plt.cm.tab10.colors
        for i, (row, bi, ct) in enumerate(zip(A, b, constraint_types)):
            a1, a2 = row
            label  = f"{a1:.4g}x₁ + {a2:.4g}x₂ {ct} {bi:.4g}"
            if abs(a2) > self.epsilon:
                x2_line = (bi - a1 * x1_range) / a2
                ax.plot(x1_range, x2_line,
                        color=colors[i % 10], linewidth=1.8,
                        label=label, zorder=3)
            else:
                x1_val = bi / a1 if abs(a1) > self.epsilon else 0
                ax.axvline(x=x1_val, color=colors[i % 10],
                           linewidth=1.8, label=label, zorder=3)

        # ── Tô miền khả thi ──────────────────────────────────────
        if len(feasible) >= 3:
            try:
                from scipy.spatial import ConvexHull
                pts  = np.array(feasible)
                hull = ConvexHull(pts)
                poly = plt.Polygon(pts[hull.vertices],
                                   alpha=0.2, color='steelblue',
                                   label='Miền khả thi', zorder=1)
                ax.add_patch(poly)
            except Exception:
                pass

        # ── Vẽ tất cả đỉnh ───────────────────────────────────────
        for v in feasible:
            f_v    = float(c @ v)
            is_opt = any(np.allclose(v, ov, atol=self.epsilon)
                         for ov in optimal_vertices)
            ax.scatter(v[0], v[1],
                       color='red' if is_opt else 'royalblue',
                       s=140 if is_opt else 55, zorder=6)
            ax.annotate(
                f"({'★' if is_opt else ''}({v[0]:.2f},{v[1]:.2f})) f={f_v:.2f}",
                xy=(v[0], v[1]), xytext=(8, 6),
                textcoords='offset points', fontsize=8.5,
                color='red' if is_opt else 'navy',
                fontweight='bold' if is_opt else 'normal',
                bbox=dict(boxstyle='round,pad=0.2',
                          fc='#fff0f0' if is_opt else 'white', alpha=0.85)
            )

        # ── Highlight cạnh tối ưu nếu vô số nghiệm ───────────────
        if infinite_solutions and optimal_edge is not None:
            vi, vj = optimal_edge
            ax.plot([vi[0], vj[0]], [vi[1], vj[1]],
                    color='red', linewidth=3.5, alpha=0.7,
                    label='Cạnh tối ưu (vô số nghiệm)', zorder=7)

        # ── Đường đẳng trị ───────────────────────────────────────
        c1, c2 = c
        if abs(c2) > self.epsilon:
            x2_iso = (f_opt - c1 * x1_range) / c2
            ax.plot(x1_range, x2_iso, 'r--', linewidth=1.5, alpha=0.7,
                    label=f'Đẳng trị f={f_opt:.2f}')

        # ── Format ───────────────────────────────────────────────
        ax.set_xlim(0, x1_max)
        ax.set_ylim(0, x2_max)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)

        obj_str  = f"{'Max' if is_max else 'Min'} f = {c[0]:.4g}x₁ + {c[1]:.4g}x₂"
        if infinite_solutions:
            note = "⚠ Vô số nghiệm tối ưu (cạnh đỏ)"
        else:
            note = f"x₁={x_opt[0]:.4f}, x₂={x_opt[1]:.4f}"
        ax.set_title(f"{obj_str}  {title_note}\n{note}, f={f_opt:.4f}",
                     fontsize=12, fontweight='bold')

        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    solver = GeometricSolver()

    print("=" * 50)
    print("TEST 1: Nghiệm duy nhất")
    print("=" * 50)
    status, x_opt, f_opt, history = solver.solve(
        c=[3, 2],
        A=np.array([[1, 1], [1, 3]]),
        b=[4, 6],
        constraint_types=['<=', '<='],
        is_max=True
    )
    print(f"Trạng thái : {status}")
    print(f"Nghiệm     : x1={x_opt[0]:.4f}, x2={x_opt[1]:.4f}")
    print(f"Giá trị Max: {f_opt:.4f}")
    for h in history: print(f"  {h['message']}")

    print("\n" + "=" * 50)
    print("TEST 2: Vô số nghiệm (như bài trong ảnh)")
    print("=" * 50)
    # Max Z = x1 - x2
    # 3x1 + x2 >= 5  (1)
    # x1 + 2x2 >= 4  (2)
    # x1 - x2  <= 1  (3)
    # x1       <= 5  (4)
    # x2       <= 5  (5)
    status, x_opt, f_opt, history = solver.solve(
        c=[1, -1],
        A=np.array([[3, 1], [1, 2], [1, -1], [1, 0], [0, 1]]),
        b=[5, 4, 1, 5, 5],
        constraint_types=['>=', '>=', '<=', '<=', '<='],
        is_max=True
    )
    print(f"Trạng thái : {status}")
    if x_opt is not None:
        print(f"Nghiệm     : x1={x_opt[0]:.4f}, x2={x_opt[1]:.4f}")
    print(f"Giá trị Max: {f_opt:.4f}")
    for h in history: print(f"  {h['message']}")