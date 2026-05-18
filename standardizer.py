import numpy as np


class Standardizer:
    """
    Converts a general LP problem into standard form for the Simplex solvers.

    Standard form: Min c^T x,  s.t. Ax = b,  x >= 0,  b >= 0

    Handles:
    - Max → Min conversion (negate objective)
    - Variable bounds: >= 0, <= 0, free
    - Negative RHS rows (multiply by -1)
    - Slack variables for <=
    - Surplus variables for >=
    - No extra variable for =
    """

    def __init__(self):
        self.is_max = False

    def transform(self, c, A, b, constraint_types, variable_bounds, is_max=False):
        """
        Parameters
        ----------
        c                : objective function coefficients
        A                : constraint coefficient matrix
        b                : right-hand side vector
        constraint_types : list of '<=', '>=', or '='
        variable_bounds  : list of '>=0', '<=0', or 'free'
        is_max           : True if maximization problem

        Returns
        -------
        new_c    : standardized objective (always Min)
        final_A  : standardized constraint matrix
        new_b    : standardized RHS (all >= 0)
        """
        self.is_max = is_max

        # Max → Min: negate objective
        new_c = -np.array(c, dtype=float) if is_max else np.array(c, dtype=float)
        new_A = np.array(A, dtype=float)
        new_b = np.array(b, dtype=float)

        # Avoid mutating original list
        con_types = list(constraint_types)

        # Step 1: Handle variable bounds  
        n_vars = new_A.shape[1]

        for j in range(n_vars):
            bound = variable_bounds[j]
            if bound == '<=0':
                # x_j <= 0 → substitute x_j = -x_j' (x_j' >= 0)
                new_A[:, j] *= -1
                new_c[j]    *= -1
            elif bound == 'free':
                # x_j free → substitute x_j = x_j' - x_j'' (both >= 0)
                new_col = -new_A[:, j].reshape(-1, 1)
                new_A   = np.hstack((new_A, new_col))
                new_c   = np.append(new_c, -new_c[j])

        # Step 2: Ensure b >= 0  
        for i in range(len(new_b)):
            if new_b[i] < 0:
                new_b[i]    *= -1
                new_A[i]    *= -1
                if con_types[i] == '<=':
                    con_types[i] = '>='
                elif con_types[i] == '>=':
                    con_types[i] = '<='

        # Step 3: Add slack / surplus variables  
        final_A = new_A.copy()

        for i, t in enumerate(con_types):
            if t == '=':
                continue  # no extra variable needed

            slack_col = np.zeros((len(new_b), 1))
            if t == '<=':
                slack_col[i] =  1.0   # slack variable
            elif t == '>=':
                slack_col[i] = -1.0   # surplus variable

            final_A = np.hstack((final_A, slack_col))
            new_c   = np.append(new_c, 0.0)

        return new_c, final_A, new_b

    def get_final_result(self, min_value):
        """Convert Min result back to Max if original problem was maximization."""
        return -min_value if self.is_max else min_value


if __name__ == "__main__":
    # Example: Max 3x + 2y
    # s.t. x + y  <= 4
    #      x - y  >= 1
    #      x + 2y  = 6
    c  = [3, 2]
    A  = [[1,  1],
          [1, -1],
          [1,  2]]
    b  = [4, 1, 6]
    ct = ['<=', '>=', '=']
    vb = ['>=0', '>=0']

    std = Standardizer()
    new_c, final_A, new_b = std.transform(c, A, b, ct, vb, is_max=True)

    print("Standardized objective (Min):")
    print(new_c)
    print("\nStandardized A:")
    print(final_A)
    print("\nStandardized b:")
    print(new_b)
