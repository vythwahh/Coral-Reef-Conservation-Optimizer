import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


class GeometricSolver:
    """
    Geometric method for 2-variable LP problems.

    Finds all vertices of the feasible region by solving pairwise
    constraint intersections, evaluates the objective at each vertex,
    and detects infinite optimal solutions (when c is parallel to an
    optimal edge).
    """

    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def solve(self, c, A, b, constraint_types, is_max=False, plot=True):
        """
        Parameters
         
        c                : objective coefficients [c1, c2]
        A                : constraint matrix (m x 2)
        b                : RHS vector
        constraint_types : list of '<=', '>=', or '='
        is_max           : True for maximization
        plot             : True to render the feasible region plot

        Returns
         
        status   : 'Optimal', 'Optimal (infinite solutions)', or 'Infeasible'
        x_opt    : optimal vertex [x1, x2]
        f_opt    : optimal objective value
        history  : list of vertex evaluation dicts
        """
        assert len(c) == 2,     "GeometricSolver only supports 2-variable problems!"
        assert A.shape[1] == 2, "Matrix A must have exactly 2 columns!"

        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        m = len(b)

        if m == 2:
            return self._solve_two_constraints(c, A, b, constraint_types, is_max, plot)
        else:
            return self._solve_general(c, A, b, constraint_types, is_max, plot, m)

     
    # CASE 1: TWO CONSTRAINTS
     

    def _solve_two_constraints(self, c, A, b, constraint_types, is_max, plot):
        """Handle the special case of exactly 2 constraints."""
        lines = []
        for i in range(2):
            lines.append((A[i], b[i]))
        # Add non-negativity bounds
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
            title_note="(2 constraints)"
        )

     
    # CASE 2: GENERAL (m constraints)
     

    def _solve_general(self, c, A, b, constraint_types, is_max, plot, m):
        """Handle the general case with m constraints."""
        vertices = self._find_vertices(A, b, m)
        if not vertices:
            return "Infeasible", None, None, []

        feasible = [v for v in vertices
                    if self._is_feasible(v, A, b, constraint_types)]
        if not feasible:
            return "Infeasible", None, None, []

        return self._evaluate_and_plot(
            c, A, b, constraint_types, feasible, is_max, plot,
            title_note=f"({m} constraints)"
        )

     
    # EVALUATE + DETECT INFINITE SOLUTIONS + PLOT
    

    def _evaluate_and_plot(self, c, A, b, constraint_types,
                           feasible, is_max, plot, title_note):
        """
        Evaluate objective at all feasible vertices, detect infinite solutions,
        and optionally render the plot.
        """
        history = []
        for v in feasible:
            f_val = float(c @ v)
            history.append({
                "vertex" : v.tolist(),
                "f_value": f_val,
                "message": (
                    f"Vertex ({v[0]:.4f}, {v[1]:.4f}): "
                    f"f = {c[0]}×{v[0]:.4f} + {c[1]}×{v[1]:.4f} = {f_val:.4f}"
                )
            })

        best  = max(history, key=lambda h: h["f_value"]) if is_max \
                else min(history, key=lambda h: h["f_value"])
        f_opt = best["f_value"]

        # All vertices achieving the optimal value
        optimal_vertices = [
            np.array(h["vertex"]) for h in history
            if abs(h["f_value"] - f_opt) < self.epsilon
        ]

        # Detect infinite solutions:
        # Occurs when >= 2 vertices share f_opt AND c is parallel to the edge between them
        # (cross product of c and edge direction = 0)
        infinite_solutions = False
        optimal_edge       = None

        if len(optimal_vertices) >= 2:
            for i, j in combinations(range(len(optimal_vertices)), 2):
                vi, vj   = optimal_vertices[i], optimal_vertices[j]
                edge_dir = vj - vi
                cross    = c[0] * edge_dir[1] - c[1] * edge_dir[0]
                if abs(cross) < self.epsilon:
                    infinite_solutions = True
                    optimal_edge       = (vi, vj)
                    break

        if infinite_solutions and optimal_edge is not None:
            x_opt  = optimal_edge[0]
            status = "Optimal (infinite solutions)"
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

     
    # PRIVATE: FIND VERTICES
     

    def _find_vertices(self, A, b, m):
        """Find candidate vertices by solving all pairwise constraint intersections."""
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
        """Check whether a vertex satisfies all constraints."""
        for i in range(len(b)):
            lhs = A[i] @ v
            if constraint_types[i] == '<=' and lhs > b[i] + self.epsilon:
                return False
            if constraint_types[i] == '>=' and lhs < b[i] - self.epsilon:
                return False
            if constraint_types[i] == '='  and abs(lhs - b[i]) > self.epsilon:
                return False
        return True

     
    # PRIVATE: PLOT
     

    def _plot(self, c, A, b, constraint_types, feasible,
              optimal_vertices, x_opt, f_opt, is_max,
              infinite_solutions, optimal_edge, title_note):
        """Render the feasible region, vertices, and optimal solution."""
        fig, ax = plt.subplots(figsize=(8, 7))

        xs       = [v[0] for v in feasible]
        ys       = [v[1] for v in feasible]
        margin   = max(max(xs + [1]) * 0.4, 1)
        x1_max   = max(xs + [1]) + margin
        x2_max   = max(ys + [1]) + margin
        x1_range = np.linspace(0, x1_max, 400)

        #  Constraint lines  
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

        # Shade feasible region 
        if len(feasible) >= 3:
            try:
                from scipy.spatial import ConvexHull
                pts  = np.array(feasible)
                hull = ConvexHull(pts)
                poly = plt.Polygon(pts[hull.vertices],
                                   alpha=0.2, color='steelblue',
                                   label='Feasible region', zorder=1)
                ax.add_patch(poly)
            except Exception:
                pass

        # Plot all vertices  
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

        # Highlight optimal edge (infinite solutions)  
        if infinite_solutions and optimal_edge is not None:
            vi, vj = optimal_edge
            ax.plot([vi[0], vj[0]], [vi[1], vj[1]],
                    color='red', linewidth=3.5, alpha=0.7,
                    label='Optimal edge (infinite solutions)', zorder=7)

        # Iso-objective line  
        c1, c2 = c
        if abs(c2) > self.epsilon:
            x2_iso = (f_opt - c1 * x1_range) / c2
            ax.plot(x1_range, x2_iso, 'r--', linewidth=1.5, alpha=0.7,
                    label=f'Iso-objective f={f_opt:.2f}')

        #  Formatting  
        ax.set_xlim(0, x1_max)
        ax.set_ylim(0, x2_max)
        ax.set_xlabel('x₁', fontsize=12)
        ax.set_ylabel('x₂', fontsize=12)

        obj_str = f"{'Max' if is_max else 'Min'} f = {c[0]:.4g}x₁ + {c[1]:.4g}x₂"
        note    = "⚠ Infinite optimal solutions (red edge)" \
                  if infinite_solutions \
                  else f"x₁={x_opt[0]:.4f}, x₂={x_opt[1]:.4f}"
        ax.set_title(f"{obj_str}  {title_note}\n{note}, f={f_opt:.4f}",
                     fontsize=12, fontweight='bold')

        ax.axhline(0, color='black', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()


 
# MAIN
 

if __name__ == "__main__":
    solver = GeometricSolver()

    print("=" * 50)
    print("TEST 1: Unique optimal solution")
    print("=" * 50)
    status, x_opt, f_opt, history = solver.solve(
        c=[3, 2],
        A=np.array([[1, 1], [1, 3]]),
        b=[4, 6],
        constraint_types=['<=', '<='],
        is_max=True
    )
    print(f"Status   : {status}")
    print(f"Solution : x1={x_opt[0]:.4f}, x2={x_opt[1]:.4f}")
    print(f"Max value: {f_opt:.4f}")
    for h in history:
        print(f"  {h['message']}")

    print("\n" + "=" * 50)
    print("TEST 2: Infinite optimal solutions")
    print("=" * 50)
    # Max Z = x1 - x2
    # 3x1 + x2 >= 5
    # x1 + 2x2 >= 4
    # x1 - x2  <= 1
    # x1       <= 5
    # x2       <= 5
    status, x_opt, f_opt, history = solver.solve(
        c=[1, -1],
        A=np.array([[3, 1], [1, 2], [1, -1], [1, 0], [0, 1]]),
        b=[5, 4, 1, 5, 5],
        constraint_types=['>=', '>=', '<=', '<=', '<='],
        is_max=True
    )
    print(f"Status   : {status}")
    if x_opt is not None:
        print(f"Solution : x1={x_opt[0]:.4f}, x2={x_opt[1]:.4f}")
    print(f"Max value: {f_opt:.4f}")
    for h in history:
        print(f"  {h['message']}")
