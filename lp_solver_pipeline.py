import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PARQUET_PATH  = "./output/risk_scores"
OUTPUT_PATH   = "./output/lp_results"

# Chi phí phục hồi mock (USD/km²) theo từng vùng
RESTORATION_COST = {
    "GBR_01": 15000,
    "GBR_02": 12000,
    "CRL_01": 18000,
    "RED_01": 20000,
    "CAR_01": 16000,
}

# Diện tích (km²) mỗi vùng
REEF_AREA = {
    "GBR_01": 350,
    "GBR_02": 280,
    "CRL_01": 120,
    "RED_01": 90,
    "CAR_01": 150,
}

# ─────────────────────────────────────────────
# ĐỌC KẾT QUẢ TỪ SPARK
# ─────────────────────────────────────────────

def load_risk_scores():
    """Đọc Parquet output từ Spark Processor."""
    try:
        df = pd.read_parquet(PARQUET_PATH)
        print(f"✅ Đọc được {len(df)} vùng san hô từ Parquet")
        return df
    except Exception as e:
        print(f"⚠️ Không đọc được Parquet: {e} → dùng mock data")
        return pd.DataFrame({
            "reef_id"        : ["GBR_01", "GBR_02", "CRL_01", "RED_01", "CAR_01"],
            "name"           : ["Great Barrier Reef North", "Great Barrier Reef South",
                               "Coral Sea", "Red Sea North", "Caribbean Reef"],
            "lat"            : [-16.0, -23.0, -18.0, 27.0, 15.0],
            "lon"            : [145.5, 152.0, 152.5, 34.0, -63.0],
            "avg_sst"        : [26.96, 25.18, 26.56, 25.69, 26.89],
            "avg_risk_score" : [0.02,  0.0,   0.0,   0.0,   0.026],
            "avg_bleaching_risk": [0.0, 0.0,  0.0,   0.0,   0.0],
        })

# ─────────────────────────────────────────────
# FORMULATE BÀI TOÁN LP
# ─────────────────────────────────────────────

def build_lp_inputs(df, budget):
    """
    Bài toán LP:
        Max  Σ risk_score[i] × x[i]   ← tối đa tổng risk được cứu
        s.t. Σ cost[i] × x[i] <= B    ← trong ngân sách
             0 <= x[i] <= area[i]     ← giới hạn diện tích mỗi vùng

    Biến x[i]: diện tích (km²) được phục hồi tại vùng i
    """
    n = len(df)
    reef_ids = df["reef_id"].tolist()

    # Hệ số hàm mục tiêu (âm vì solver giải Min)
    c = []
    for _, row in df.iterrows():
        rid = row["reef_id"]
        # Risk score × diện tích → tổng "impact" được cứu
        c.append(-float(row["avg_risk_score"]) if float(row["avg_risk_score"]) > 0 else -0.001)

    # Ma trận ràng buộc ngân sách
    costs = [RESTORATION_COST.get(rid, 15000) for rid in reef_ids]
    areas = [REEF_AREA.get(rid, 100) for rid in reef_ids]

    # A: [budget_constraint; upper_bound per zone]
    # b: [budget; area per zone]
    A = []
    b_vec = []

    # Ràng buộc 1: tổng chi phí <= budget
    A.append(costs)
    b_vec.append(float(budget))

    # Ràng buộc 2: mỗi vùng không vượt diện tích tối đa
    for i in range(n):
        row_constraint = [0.0] * n
        row_constraint[i] = 1.0
        A.append(row_constraint)
        b_vec.append(float(areas[i]))

    constraint_types = ["<="] * len(b_vec)
    variable_bounds  = [">=0"] * n

    return np.array(c), np.array(A), np.array(b_vec), constraint_types, variable_bounds, reef_ids, costs, areas

# ─────────────────────────────────────────────
# CHẠY LP SOLVER
# ─────────────────────────────────────────────

def run_lp_solver(budget):
    """
    Đọc risk scores từ Spark → formulate LP → giải → trả về kết quả.
    """
    from standardizer import Standardizer
    from danzig_solver import DantzigSolver

    df = load_risk_scores()
    c, A, b_vec, ct, vb, reef_ids, costs, areas = build_lp_inputs(df, budget)

    # Chuẩn hóa
    std = Standardizer()
    c_std, A_std, b_std = std.transform(
        c.tolist(), A.tolist(), b_vec.tolist(),
        ct, vb, is_max=False
    )

    # Giải
    solver = DantzigSolver()
    status, x_opt, f_opt, history = solver.solve(c_std, A_std, b_std)

    if status != "Optimal" or x_opt is None:
        return None, status, df

    # Lấy nghiệm (chỉ n biến gốc)
    n = len(reef_ids)
    allocation = x_opt[:n]

    # Build kết quả
    results = []
    total_cost = 0
    total_impact = 0

    for i, rid in enumerate(reef_ids):
        area_restored = round(float(allocation[i]), 2)
        cost_used     = round(area_restored * costs[i], 2)
        risk_score    = float(df[df["reef_id"] == rid]["avg_risk_score"].values[0])
        impact        = round(area_restored * risk_score, 4)
        name          = df[df["reef_id"] == rid]["name"].values[0]

        total_cost   += cost_used
        total_impact += impact

        results.append({
            "reef_id"       : rid,
            "name"          : name,
            "area_restored" : area_restored,
            "cost_used"     : cost_used,
            "risk_score"    : risk_score,
            "impact"        : impact,
            "budget_pct"    : 0,  # fill sau
        })

    # Tính % ngân sách
    for r in results:
        r["budget_pct"] = round(r["cost_used"] / budget * 100, 1) if budget > 0 else 0

    result_df = pd.DataFrame(results).sort_values("impact", ascending=False)

    # Lưu kết quả
    summary = {
        "timestamp"    : datetime.now().isoformat(),
        "budget"       : budget,
        "total_cost"   : round(total_cost, 2),
        "budget_used_pct": round(total_cost / budget * 100, 1),
        "total_impact" : round(total_impact, 4),
        "status"       : status,
        "n_zones"      : len([r for r in results if r["area_restored"] > 0]),
    }

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(f"{OUTPUT_PATH}/allocation.parquet", index=False)
    with open(f"{OUTPUT_PATH}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ LP Solver hoàn thành!")
    print(f"   Budget: ${budget:,.0f}")
    print(f"   Đã dùng: ${total_cost:,.0f} ({summary['budget_used_pct']}%)")
    print(f"   Tổng impact: {total_impact:.4f}")
    print(f"\n📊 Kết quả phân bổ:")
    print(result_df[["reef_id", "name", "area_restored", "cost_used", "impact"]].to_string(index=False))

    return result_df, status, summary


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    result_df, status, summary = run_lp_solver(budget=500_000)

    if result_df is not None:
        print(f"\n📋 Summary:")
        print(f"   Status       : {status}")
        print(f"   Budget       : ${summary['budget']:,.0f}")
        print(f"   Used         : ${summary['total_cost']:,.0f} ({summary['budget_used_pct']}%)")
        print(f"   Total impact : {summary['total_impact']:.4f}")