import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

 
# CONFIG

PARQUET_PATH = "./output/risk_scores"
OUTPUT_PATH  = "./output/lp_results"

# Restoration cost (USD/km^2) per reef zone — based on literature estimates
RESTORATION_COST = {
    "GBR_01": 15000,
    "GBR_02": 12000,
    "CRL_01": 18000,
    "RED_01": 20000,
    "CAR_01": 16000,
}

# Available restoration area (km^2) per reef zone
REEF_AREA = {
    "GBR_01": 350,
    "GBR_02": 280,
    "CRL_01": 120,
    "RED_01": 90,
    "CAR_01": 150,
}

 
# LOAD RISK SCORES FROM SPARK OUTPUT
 

def load_risk_scores():
    """Load Parquet output from Spark Processor. Falls back to mock data if unavailable."""
    try:
        df = pd.read_parquet(PARQUET_PATH)
        print(f" Loaded {len(df)} reef zones from Parquet")
        return df
    except Exception as e:
        print(f"  Could not read Parquet: {e} → using mock data")
        return pd.DataFrame({
            "reef_id"           : ["GBR_01", "GBR_02", "CRL_01", "RED_01", "CAR_01"],
            "name"              : ["Great Barrier Reef North", "Great Barrier Reef South",
                                   "Coral Sea", "Red Sea North", "Caribbean Reef"],
            "lat"               : [-16.0, -23.0, -18.0, 27.0,  15.0],
            "lon"               : [145.5, 152.0, 152.5, 34.0, -63.0],
            "avg_sst"           : [26.96, 25.18, 26.56, 25.69, 26.89],
            "avg_risk_score"    : [0.02,  0.0,   0.0,   0.0,   0.026],
            "avg_bleaching_risk": [0.0,   0.0,   0.0,   0.0,   0.0],
        })

 
# FORMULATE LP PROBLEM
 

def build_lp_inputs(df, budget):
    """
    LP formulation:
        Max  Σ risk_score[i] × x[i]    ← maximize total conservation impact
        s.t. Σ cost[i] × x[i] <= B     ← within budget
             0 <= x[i] <= area[i]      ← area constraints per zone

    Decision variable x[i]: area restored (km²) in reef zone i.
    """
    n        = len(df)
    reef_ids = df["reef_id"].tolist()

    # Objective coefficients (negated for Min formulation)
    # Use small fallback (-0.001) to avoid zero coefficient for zones with no risk
    c = [
        -float(row["avg_risk_score"]) if float(row["avg_risk_score"]) > 0 else -0.001
        for _, row in df.iterrows()
    ]

    costs = [RESTORATION_COST.get(rid, 15000) for rid in reef_ids]
    areas = [REEF_AREA.get(rid, 100)          for rid in reef_ids]

    A     = []
    b_vec = []

    # Constraint 1: total cost <= budget
    A.append(costs)
    b_vec.append(float(budget))

    # Constraint 2: area restored per zone <= max available area
    for i in range(n):
        row_constraint    = [0.0] * n
        row_constraint[i] = 1.0
        A.append(row_constraint)
        b_vec.append(float(areas[i]))

    constraint_types = ["<="] * len(b_vec)
    variable_bounds  = [">=0"] * n

    return np.array(c), np.array(A), np.array(b_vec), constraint_types, variable_bounds, reef_ids, costs, areas

 
# RUN LP SOLVER
 

def run_lp_solver(budget):
    """
    Full pipeline: load risk scores → formulate LP → solve → return results.

    Parameters
    ----------
    budget : float — total conservation budget in USD

    Returns
    -------
    result_df : DataFrame with allocation per reef zone
    status    : solver status string
    summary   : dict with aggregated metrics
    """
    from standardizer import Standardizer
    from danzig_solver import DantzigSolver

    df = load_risk_scores()
    c, A, b_vec, ct, vb, reef_ids, costs, areas = build_lp_inputs(df, budget)

    # Standardize to Min form
    std = Standardizer()
    c_std, A_std, b_std = std.transform(
        c.tolist(), A.tolist(), b_vec.tolist(),
        ct, vb, is_max=False
    )

    # Solve
    solver = DantzigSolver()
    status, x_opt, f_opt, history = solver.solve(c_std, A_std, b_std)

    if status != "Optimal" or x_opt is None:
        return None, status, df

    # Extract original variable values (first n variables)
    n          = len(reef_ids)
    allocation = x_opt[:n]

    # Build result records
    results      = []
    total_cost   = 0.0
    total_impact = 0.0

    for i, rid in enumerate(reef_ids):
        area_restored = round(float(allocation[i]), 2)
        cost_used     = round(area_restored * costs[i], 2)
        risk_score    = float(df[df["reef_id"] == rid]["avg_risk_score"].values[0])
        impact        = round(area_restored * risk_score, 4)
        name          = df[df["reef_id"] == rid]["name"].values[0]

        total_cost   += cost_used
        total_impact += impact

        results.append({
            "reef_id"      : rid,
            "name"         : name,
            "area_restored": area_restored,
            "cost_used"    : cost_used,
            "risk_score"   : risk_score,
            "impact"       : impact,
            "budget_pct"   : 0,  # filled below
        })

    # Fill budget percentage
    for r in results:
        r["budget_pct"] = round(r["cost_used"] / budget * 100, 1) if budget > 0 else 0

    result_df = pd.DataFrame(results).sort_values("impact", ascending=False)

    # Summary metrics
    summary = {
        "timestamp"      : datetime.now().isoformat(),
        "budget"         : budget,
        "total_cost"     : round(total_cost, 2),
        "budget_used_pct": round(total_cost / budget * 100, 1),
        "total_impact"   : round(total_impact, 4),
        "status"         : status,
        "n_zones"        : len([r for r in results if r["area_restored"] > 0]),
    }

    # Save outputs
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(f"{OUTPUT_PATH}/allocation.parquet", index=False)
    with open(f"{OUTPUT_PATH}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n LP Solver completed!")
    print(f"   Budget      : ${budget:,.0f}")
    print(f"   Used        : ${total_cost:,.0f} ({summary['budget_used_pct']}%)")
    print(f"   Total impact: {total_impact:.4f}")
    print(f"\n Allocation results:")
    print(result_df[["reef_id", "name", "area_restored", "cost_used", "impact"]].to_string(index=False))

    return result_df, status, summary

 
# MAIN
 

if __name__ == "__main__":
    result_df, status, summary = run_lp_solver(budget=500_000)

    if result_df is not None:
        print(f"\n Summary:")
        print(f"   Status      : {status}")
        print(f"   Budget      : ${summary['budget']:,.0f}")
        print(f"   Used        : ${summary['total_cost']:,.0f} ({summary['budget_used_pct']}%)")
        print(f"   Total impact: {summary['total_impact']:.4f}")
