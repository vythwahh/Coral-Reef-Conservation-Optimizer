import numpy as np

class BlandSolver:
    """
    Simplex Bland thuần túy 1 pha.
    Yêu cầu: BFS đã có sẵn (toàn slack <=).
    Khác DantzigSolver duy nhất 1 chỗ:
        entering_col = index NHỎ NHẤT có hệ số âm (thay vì âm nhất)
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

        return self._simplex(tableau, basis, m, n, history=[], phase=1)

    def _simplex(self, tableau, basis, m, n_vars, history, phase=1):  # ← thêm phase
        for iteration in range(1000):
            current_step = {
                "iteration": iteration,
                "tableau"  : tableau.copy(),
                "basis"    : basis.copy(),
                "pivot"    : None,
                "message"  : ""
            }

            # Bland: index NHỎ NHẤT có hệ số âm
            entering_col = -1
            for j in range(n_vars):
                if tableau[-1, j] < -self.epsilon:
                    entering_col = j
                    break

            if entering_col == -1:
                current_step["message"] = f"[Phase {phase}] Tối ưu."  # ← thêm phase
                history.append(current_step)
                return "Optimal", self._extract_solution(tableau, basis, n_vars), float(tableau[-1, -1]), history

            ratios = [
                tableau[i, -1] / tableau[i, entering_col]
                if tableau[i, entering_col] > self.epsilon else float('inf')
                for i in range(m)
            ]

            min_ratio = min(ratios)
            if min_ratio == float('inf'):
                current_step["message"] = f"[Phase {phase}] Vô hạn."  # ← thêm phase
                history.append(current_step)
                return "Unbounded", None, None, history

            # Bland tiebreak: basis index nhỏ nhất
            candidates  = [i for i, r in enumerate(ratios) if abs(r - min_ratio) < self.epsilon]
            leaving_row = min(candidates, key=lambda i: basis[i])

            current_step["pivot"]   = (leaving_row, entering_col)
            current_step["message"] = (
                f"[Phase {phase}] "                                    # ← thêm phase
                f"x{entering_col+1} vào (index nhỏ nhất âm), "
                f"x{basis[leaving_row]+1} ra "
                f"(pivot [{leaving_row}, {entering_col}])."
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


if __name__ == "__main__":
    c = [-3, -2, 0, 0]
    A = np.array([[1, 1, 1, 0],
                  [1, 3, 0, 1]])
    b = [4, 6]

    solver = BlandSolver()
    status, x_opt, f_opt, history = solver.solve(c, A, b)
    print(f"Trạng thái : {status}")
    print(f"Nghiệm     : x={x_opt[0]:.4f}, y={x_opt[1]:.4f}")
    print(f"Giá trị Max: {-f_opt:.4f}")
    print(f"Số bước    : {len(history)}")