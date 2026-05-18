import numpy as np

class Standardizer:
    def __init__(self):
        self.is_max = False

    def transform(self, c, A, b, constraint_types, variable_bounds, is_max=False):
        """
        c: hệ số hàm mục tiêu
        A: ma trận hệ số ràng buộc
        b: vế phải
        constraint_types: danh sách ['<=', '>=', '=']
        variable_bounds: danh sách ['>=0', '<=0', 'free']
        is_max: True nếu bài toán là Max, False nếu là Min
        """

        self.is_max = is_max
        new_c = -np.array(c, dtype=float) if is_max else np.array(c, dtype=float)
        new_A = np.array(A, dtype=float)
        new_b = np.array(b, dtype=float)

        # Tránh mutate list gốc
        con_types = list(constraint_types)

        #  1. Xử lý điều kiện biến  
        n_vars = new_A.shape[1]
        cols_to_delete = []

        for j in range(n_vars):
            bound = variable_bounds[j]
            if bound == '<=0':
                # x_j <= 0 → thay x_j = -x_j' (x_j' >= 0)
                new_A[:, j] *= -1
                new_c[j] *= -1
            elif bound == 'free':
                # x_j free → thay x_j = x_j' - x_j'' (cả hai >= 0)
                new_col = -new_A[:, j].reshape(-1, 1)  # cột của x_j''
                new_A = np.hstack((new_A, new_col))
                new_c = np.append(new_c, -new_c[j])

        #   2. Đưa b về không âm 
        for i in range(len(new_b)):
            if new_b[i] < 0:
                new_b[i] *= -1
                new_A[i] *= -1
                if con_types[i] == '<=':
                    con_types[i] = '>='
                elif con_types[i] == '>=':
                    con_types[i] = '<='

        #  3. Thêm slack / surplus variables 
        final_A = new_A.copy()  # copy tường minh
        for i, t in enumerate(con_types):
            if t == '=':
                continue  # không cần thêm biến phụ
            slack_col = np.zeros((len(new_b), 1))
            if t == '<=':
                slack_col[i] = 1.0   # biến bù (slack)
            elif t == '>=':
                slack_col[i] = -1.0  # biến dư (surplus)
            final_A = np.hstack((final_A, slack_col))
            new_c = np.append(new_c, 0.0)

        return new_c, final_A, new_b

    def get_final_result(self, min_value):
        return -min_value if self.is_max else min_value