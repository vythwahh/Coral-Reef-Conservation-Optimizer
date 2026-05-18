import numpy as np

class DantzigSolver:
    """
    Simplex Dantzig thuần túy 1 pha.
    Yêu cầu: BFS đã có sẵn (toàn slack <=).
    Nếu cần 2 pha → dùng TwoPhaseSolver.
    """

    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon

    def solve(self, c, A, b):
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        assert np.all(b >= -self.epsilon), "b phải >= 0"

        m, n    = A.shape
        basis   = self._find_initial_basis(A, m, n)
        tableau = self._build_tableau(c, A, b, n)

        return self._simplex(tableau, basis, m, n, history=[])

    def _simplex(self, tableau, basis, m, n_vars, history):
        for iteration in range(1000):
            current_step = {
                "iteration": iteration,
                "tableau"  : tableau.copy(),
                "basis"    : basis.copy(),
                "pivot"    : None,
                "message"  : ""
            }

            # Dantzig: chọn cột âm nhất
            obj_row      = tableau[-1, :n_vars]
            entering_col = int(np.argmin(obj_row))

            if obj_row[entering_col] >= -self.epsilon:
                current_step["message"] = "Tối ưu."
                history.append(current_step)
                return "Optimal", self._extract_solution(tableau, basis, n_vars), float(tableau[-1, -1]), history

            ratios = [
                tableau[i, -1] / tableau[i, entering_col]
                if tableau[i, entering_col] > self.epsilon else float('inf')
                for i in range(m)
            ]

            min_ratio = min(ratios)
            if min_ratio == float('inf'):
                current_step["message"] = "Vô hạn."
                history.append(current_step)
                return "Unbounded", None, None, history

            candidates  = [i for i, r in enumerate(ratios) if abs(r - min_ratio) < self.epsilon]
            leaving_row = min(candidates, key=lambda i: basis[i])

            current_step["pivot"]   = (leaving_row, entering_col)
            current_step["message"] = (
                f"x{entering_col+1} vào (hệ số {tableau[-1, entering_col]:.4f}), "
                f"x{basis[leaving_row]+1} ra (pivot [{leaving_row}, {entering_col}])."
            )
            history.append(current_step)

            self._pivot(tableau, leaving_row, entering_col, m)
            basis[leaving_row] = entering_col

        return "Max iterations", None, None, history

    def _find_initial_basis(self, A, m, n):
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
        m       = A.shape[0]
        tableau = np.zeros((m + 1, n + 1))
        tableau[:m, :n] = A
        tableau[:m, -1] = b
        tableau[-1, :n] = c
        return tableau

    def _pivot(self, tableau, pivot_row, pivot_col, m):
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    def _extract_solution(self, tableau, basis, n_original):
        solution = np.zeros(n_original)
        for i, b_idx in enumerate(basis):
            if b_idx < n_original:
                solution[b_idx] = tableau[i, -1]
        return solution