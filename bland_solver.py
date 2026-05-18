import numpy as np


class BlandSolver:
    """
    Single-phase Simplex using Bland's Rule.

    Requirements: BFS already available (all constraints are <=).
    Difference from Dantzig: entering variable = smallest index
    with negative reduced cost (instead of most negative).
    """

    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def solve(self, c, A, b):
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        assert np.all(b >= -self.epsilon), "b must be >= 0 (run Standardizer first)"

        m, n    = A.shape
        basis   = self._find_initial_basis(A, m, n)
        tableau = self._build_tableau(c, A, b, n)

        return self._simplex(tableau, basis, m, n, history=[], phase=1)

    def _simplex(self, tableau, basis, m, n_vars, history, phase=1):
        for iteration in range(1000):
            current_step = {
                "iteration": iteration,
                "tableau"  : tableau.copy(),
                "basis"    : basis.copy(),
                "pivot"    : None,
                "message"  : ""
            }

            # Bland's Rule: smallest index with negative reduced cost
            entering_col = -1
            for j in range(n_vars):
                if tableau[-1, j] < -self.epsilon:
                    entering_col = j
                    break

            if entering_col == -1:
                current_step["message"] = f"[Phase {phase}] Optimal."
                history.append(current_step)
                return "Optimal", self._extract_solution(tableau, basis, n_vars), float(tableau[-1, -1]), history

            # Ratio test
            ratios = [
                tableau[i, -1] / tableau[i, entering_col]
                if tableau[i, entering_col] > self.epsilon else float('inf')
                for i in range(m)
            ]

            min_ratio = min(ratios)
            if min_ratio == float('inf'):
                current_step["message"] = f"[Phase {phase}] Unbounded."
                history.append(current_step)
                return "Unbounded", None, None, history

            # Bland tiebreak: smallest basis index
            candidates  = [i for i, r in enumerate(ratios) if abs(r - min_ratio) < self.epsilon]
            leaving_row = min(candidates, key=lambda i: basis[i])

            current_step["pivot"]   = (leaving_row, entering_col)
            current_step["message"] = (
                f"[Phase {phase}] "
                f"x{entering_col+1} enters (smallest negative index), "
                f"x{basis[leaving_row]+1} leaves "
                f"(pivot [{leaving_row}, {entering_col}])."
            )
            history.append(current_step)

            self._pivot(tableau, leaving_row, entering_col, m)
            basis[leaving_row] = entering_col

        return "Max iterations", None, None, history

    def _find_initial_basis(self, A, m, n):
        """Detect identity columns in A to form initial basis."""
        basis = [-1] * m
        for j in range(n - 1, -1, -1):
            col   = A[:, j]
            ones  = np.where(np.abs(col - 1.0) < self.epsilon)[0]
            zeros = np.where(np.abs(col)        < self.epsilon)[0]
            if len(ones) == 1 and len(zeros) == m - 1:
                row = ones[0]
                if basis[row] == -1:
                    basis[row] = j
        return basis

    def _build_tableau(self, c, A, b, n):
        """Build initial simplex tableau."""
        m       = A.shape[0]
        tableau = np.zeros((m + 1, n + 1))
        tableau[:m, :n] = A
        tableau[:m, -1] = b
        tableau[-1, :n] = c
        return tableau

    def _pivot(self, tableau, pivot_row, pivot_col, m):
        """Gauss-Jordan elimination pivot step."""
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    def _extract_solution(self, tableau, basis, n_original):
        """Read solution values from tableau."""
        solution = np.zeros(n_original)
        for i, b_idx in enumerate(basis):
            if b_idx < n_original:
                solution[b_idx] = tableau[i, -1]
        return solution


if __name__ == "__main__":
    # Example: Max 3x + 2y
    # s.t. x + y  <= 4
    #      x + 3y <= 6
    c = [-3, -2, 0, 0]
    A = np.array([[1, 1, 1, 0],
                  [1, 3, 0, 1]])
    b = [4, 6]

    solver = BlandSolver()
    status, x_opt, f_opt, history = solver.solve(c, A, b)
    print(f"Status : {status}")
    print(f"Solution: x={x_opt[0]:.4f}, y={x_opt[1]:.4f}")
    print(f"Max value: {-f_opt:.4f}")
    print(f"Steps: {len(history)}")
