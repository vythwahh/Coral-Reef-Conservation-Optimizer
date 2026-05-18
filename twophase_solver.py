import numpy as np
from bland_solver import BlandSolver
from danzig_solver import DantzigSolver

class TwoPhaseSolver:
    """
    Giải LP bằng phương pháp 2 pha khi cần (có >= hoặc =).

    Phase 1: tìm BFS hợp lệ.
    Phase 2: tối ưu hàm mục tiêu thật.

    Tự động chọn rule xoay:
    - Chạy thử cả Dantzig và Bland trên Phase 1.
    - Rule nào ít bước hơn → dùng cho cả Phase 1 và Phase 2.
    - Nếu bằng nhau → dùng Dantzig.
    """

    def __init__(self, epsilon=1e-9):
        self.epsilon   = epsilon
        self.rule_used = None
        self._bland    = BlandSolver(epsilon)
        self._dantzig  = DantzigSolver(epsilon)

    # ─────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────

    def solve(self, c, A, b):
        """
        Đầu vào: c, A, b dạng chuẩn (Min, b >= 0, Ax = b, x >= 0)
        Trả về:  status, x_opt, f_opt, history

        status:
            "NeedSinglePhase" → BFS có sẵn, báo main dùng Dantzig/Bland
            "Optimal"         → tìm được nghiệm
            "Infeasible"      → vô nghiệm
            "Unbounded"       → vô hạn
        """
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        assert np.all(b >= -self.epsilon), "b phải >= 0 (chạy Standardizer trước)"

        m, n  = A.shape
        basis = self._find_initial_basis(A, m, n)

        # BFS có sẵn → không cần 2 pha, báo ra ngoài
        if -1 not in basis:
            return "NeedSinglePhase", None, None, []

        return self._run_two_phase(c, A, b, m, n, basis)

    # ─────────────────────────────────────────────
    # PRIVATE: 2 PHA
    # ─────────────────────────────────────────────

    def _run_two_phase(self, c, A, b, m, n, basis):
        history = []

        # ── Build tableau Phase 1 ─────────────────────────────────
        missing_rows = [i for i, v in enumerate(basis) if v == -1]
        n_art        = len(missing_rows)
        n_p1         = n + n_art

        tab_p1 = np.zeros((m + 1, n_p1 + 1))
        tab_p1[:m, :n] = A
        tab_p1[:m, -1] = b

        basis_p1 = basis.copy()
        for k, row_i in enumerate(missing_rows):
            col_idx                = n + k
            tab_p1[row_i, col_idx] = 1.0
            basis_p1[row_i]        = col_idx

        # Dòng obj Phase 1: đặt hệ số 1 cho biến nhân tạo rồi khử
        for k, row_i in enumerate(missing_rows):
            col_idx             = n + k
            tab_p1[-1, col_idx] = 1.0
            tab_p1[-1, :]      -= tab_p1[row_i, :]

        # ── Chạy thử để chọn rule tối ưu ─────────────────────────
        steps_dantzig = self._count_steps(tab_p1.copy(), basis_p1.copy(), m, n_p1, rule='dantzig')
        steps_bland   = self._count_steps(tab_p1.copy(), basis_p1.copy(), m, n_p1, rule='bland')

        self.rule_used = 'bland' if steps_bland < steps_dantzig else 'dantzig'

        print(f"[TwoPhaseSolver] Phase 1: Dantzig={steps_dantzig} bước, "
              f"Bland={steps_bland} bước → chọn '{self.rule_used}'")

        # ── Phase 1 thật sự với rule đã chọn ─────────────────────
        basis_run = basis.copy()
        for k, row_i in enumerate(missing_rows):
            basis_run[row_i] = n + k

        if self.rule_used == 'bland':
            status_p1, _, _, history = self._bland._simplex(
                tab_p1, basis_run, m, n_p1, history=history, phase=1
            )
        else:
            status_p1, _, _, history = self._dantzig._simplex(
                tab_p1, basis_run, m, n_p1, history=history, phase=1
            )

        if status_p1 != "Optimal":
            return "Infeasible", None, None, history

        # Biến nhân tạo còn > 0 trong basis → vô nghiệm
        for i, b_idx in enumerate(basis_run):
            if b_idx >= n and tab_p1[i, -1] > self.epsilon:
                return "Infeasible", None, None, history

        # ── Build tableau Phase 2 ─────────────────────────────────
        tab_p2 = np.zeros((m + 1, n + 1))
        tab_p2[:m, :n] = tab_p1[:m, :n]
        tab_p2[:m, -1] = tab_p1[:m, -1]

        # Thay biến nhân tạo còn trong basis bằng biến gốc
        for i, b_idx in enumerate(basis_run):
            if b_idx >= n:
                basis_run[i] = self._find_replacement(tab_p2, i, n, basis_run)

        # Đặt lại hàm mục tiêu thật rồi chuẩn hóa theo basis hiện tại
        tab_p2[-1, :n] = c
        for i, b_idx in enumerate(basis_run):
            if b_idx < n and abs(tab_p2[-1, b_idx]) > self.epsilon:
                tab_p2[-1, :] -= tab_p2[-1, b_idx] * tab_p2[i, :]

        # ── Phase 2 với cùng rule đã chọn ────────────────────────
        if self.rule_used == 'bland':
            return self._bland._simplex(
                tab_p2, basis_run, m, n, history=history, phase=2
            )
        else:
            return self._dantzig._simplex(
                tab_p2, basis_run, m, n, history=history, phase=2
            )

    # ─────────────────────────────────────────────
    # PRIVATE: ĐẾM BƯỚC THỬ
    # ─────────────────────────────────────────────

    def _count_steps(self, tableau, basis, m, n_vars, rule):
        """Chạy thử trên bản sao để đếm số bước, không lưu history."""
        solver = self._bland if rule == 'bland' else self._dantzig
        for iteration in range(1000):
            # Dùng _entering() của solver tương ứng
            entering_col = solver._entering(tableau, n_vars) \
                           if hasattr(solver, '_entering') \
                           else self._entering_by_rule(tableau, n_vars, rule)

            if entering_col == -1:
                return iteration

            ratios = [
                tableau[i, -1] / tableau[i, entering_col]
                if tableau[i, entering_col] > self.epsilon else float('inf')
                for i in range(m)
            ]

            min_ratio = min(ratios)
            if min_ratio == float('inf'):
                return iteration

            candidates  = [i for i, r in enumerate(ratios) if abs(r - min_ratio) < self.epsilon]
            leaving_row = min(candidates, key=lambda i: basis[i])

            self._pivot(tableau, leaving_row, entering_col, m)
            basis[leaving_row] = entering_col

        return 1000

    def _entering_by_rule(self, tableau, n_vars, rule):
        """Fallback nếu solver không có _entering()."""
        obj_row = tableau[-1, :n_vars]
        if rule == 'bland':
            for j in range(n_vars):
                if obj_row[j] < -self.epsilon:
                    return j
            return -1
        else:
            col = int(np.argmin(obj_row))
            return col if obj_row[col] < -self.epsilon else -1

    # ─────────────────────────────────────────────
    # PRIVATE: HELPER
    # ─────────────────────────────────────────────

    def _find_initial_basis(self, A, m, n):
        """Tìm cột đơn vị trong A để làm basis ban đầu."""
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

    def _find_replacement(self, tableau, row, n, basis):
        """Thay biến nhân tạo = 0 trong basis bằng biến gốc (degenerate)."""
        for j in range(n):
            if j not in basis and abs(tableau[row, j]) > self.epsilon:
                return j
        return basis[row]

    def _pivot(self, tableau, pivot_row, pivot_col, m):
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]


if __name__ == "__main__":
    # Bài có >= → bắt buộc 2 pha
    # Min -3x - 2y
    # x + y  >= 2  → x + y - s1 = 2
    # x + 3y <= 6  → x + 3y + s2 = 6
    c = [-3, -2,  0,  0]
    A = [[ 1,  1, -1,  0],
         [ 1,  3,  0,  1]]
    b = [2, 6]

    solver = TwoPhaseSolver()
    status, x_opt, f_opt, history = solver.solve(c, A, b)

    print(f"\nTrạng thái : {status}")
    if x_opt is not None:
        print(f"Nghiệm     : x={x_opt[0]:.2f}, y={x_opt[1]:.2f}")
        print(f"Giá trị Max: {-f_opt:.2f}")
    print(f"Số bước    : {len(history)}")
    print(f"Rule dùng  : {solver.rule_used}")