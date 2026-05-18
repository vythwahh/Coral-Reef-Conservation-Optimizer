import numpy as np
from bland_solver import BlandSolver
from danzig_solver import DantzigSolver


class TwoPhaseSolver:
    """
    Two-Phase Simplex for LP problems with >= or = constraints.

    Phase 1: Find a valid Basic Feasible Solution (BFS).
    Phase 2: Optimize the true objective function.

    Automatically selects the pivot rule:
    - Runs both Dantzig and Bland on Phase 1 (dry run).
    - Uses whichever requires fewer steps for both phases.
    - Defaults to Dantzig on tie.
    """

    def __init__(self, epsilon=1e-9):
        self.epsilon   = epsilon
        self.rule_used = None
        self._bland    = BlandSolver(epsilon)
        self._dantzig  = DantzigSolver(epsilon)

     

    def solve(self, c, A, b):
        """
        Input : c, A, b in standard form (Min, b >= 0, Ax = b, x >= 0)
        Output: status, x_opt, f_opt, history

        Status values:
            "NeedSinglePhase" → BFS already available, use Dantzig/Bland directly
            "Optimal"         → solution found
            "Infeasible"      → no feasible solution exists
            "Unbounded"       → objective is unbounded
        """
        c = np.array(c, dtype=float)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)

        assert np.all(b >= -self.epsilon), "b must be >= 0 (run Standardizer first)"

        m, n  = A.shape
        basis = self._find_initial_basis(A, m, n)

        # BFS already available → no two-phase needed
        if -1 not in basis:
            return "NeedSinglePhase", None, None, []

        return self._run_two_phase(c, A, b, m, n, basis)
 

    def _run_two_phase(self, c, A, b, m, n, basis):
        history = []

        # Build Phase 1 tableau  
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

        # Phase 1 objective: Min sum of artificial variables
        # Set coefficient = 1, then eliminate (since artificials are in basis)
        for k, row_i in enumerate(missing_rows):
            col_idx             = n + k
            tab_p1[-1, col_idx] = 1.0
            tab_p1[-1, :]      -= tab_p1[row_i, :]

        # Dry run to select optimal pivot rule  
        steps_dantzig = self._count_steps(tab_p1.copy(), basis_p1.copy(), m, n_p1, rule='dantzig')
        steps_bland   = self._count_steps(tab_p1.copy(), basis_p1.copy(), m, n_p1, rule='bland')

        self.rule_used = 'bland' if steps_bland < steps_dantzig else 'dantzig'

        print(f"[TwoPhaseSolver] Phase 1: Dantzig={steps_dantzig} steps, "
              f"Bland={steps_bland} steps → using '{self.rule_used}'")

        # Run Phase 1 with selected rule  
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

        # Artificial variables still > 0 in basis → infeasible
        for i, b_idx in enumerate(basis_run):
            if b_idx >= n and tab_p1[i, -1] > self.epsilon:
                return "Infeasible", None, None, history

        # Build Phase 2 tableau  
        tab_p2 = np.zeros((m + 1, n + 1))
        tab_p2[:m, :n] = tab_p1[:m, :n]   # A transformed by Phase 1
        tab_p2[:m, -1] = tab_p1[:m, -1]   # RHS

        # Replace any remaining artificial variables in basis
        for i, b_idx in enumerate(basis_run):
            if b_idx >= n:
                basis_run[i] = self._find_replacement(tab_p2, i, n, basis_run)

        # Restore true objective and normalize w.r.t. current basis
        tab_p2[-1, :n] = c
        for i, b_idx in enumerate(basis_run):
            if b_idx < n and abs(tab_p2[-1, b_idx]) > self.epsilon:
                tab_p2[-1, :] -= tab_p2[-1, b_idx] * tab_p2[i, :]

        #  Run Phase 2 with same rule 
        if self.rule_used == 'bland':
            return self._bland._simplex(
                tab_p2, basis_run, m, n, history=history, phase=2
            )
        else:
            return self._dantzig._simplex(
                tab_p2, basis_run, m, n, history=history, phase=2
            )

     

    def _count_steps(self, tableau, basis, m, n_vars, rule):
        """Simulate simplex on a tableau copy to count steps (no history saved)."""
        for iteration in range(1000):
            entering_col = self._entering_by_rule(tableau, n_vars, rule)
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
        """Select entering variable by pivot rule. Returns -1 if optimal."""
        obj_row = tableau[-1, :n_vars]
        if rule == 'bland':
            for j in range(n_vars):
                if obj_row[j] < -self.epsilon:
                    return j
            return -1
        else:  # dantzig
            col = int(np.argmin(obj_row))
            return col if obj_row[col] < -self.epsilon else -1

    

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

    def _find_replacement(self, tableau, row, n, basis):
        """Replace zero-valued artificial in basis with an original variable (degenerate case)."""
        for j in range(n):
            if j not in basis and abs(tableau[row, j]) > self.epsilon:
                return j
        return basis[row]

    def _pivot(self, tableau, pivot_row, pivot_col, m):
        """Gauss-Jordan elimination pivot step."""
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]


if __name__ == "__main__":
    # Example with >= constraint → requires two-phase
    # Min -3x - 2y
    # x + y  >= 2  →  x + y  - s1 = 2
    # x + 3y <= 6  →  x + 3y + s2 = 6
    c = [-3, -2,  0,  0]
    A = [[ 1,  1, -1,  0],
         [ 1,  3,  0,  1]]
    b = [2, 6]

    solver = TwoPhaseSolver()
    status, x_opt, f_opt, history = solver.solve(c, A, b)

    print(f"\nStatus   : {status}")
    if x_opt is not None:
        print(f"Solution : x={x_opt[0]:.2f}, y={x_opt[1]:.2f}")
        print(f"Max value: {-f_opt:.2f}")
    print(f"Steps    : {len(history)}")
    print(f"Rule used: {solver.rule_used}")
