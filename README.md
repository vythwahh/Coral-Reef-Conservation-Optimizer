 # 🪸 CoralOpt — Coral Reef Conservation Optimizer

An enterprise-grade, end-to-end data engineering and operations research pipeline. CoralOpt ingests real-time sea surface temperature (SST) data from the NOAA ERDDAP API, processes streaming data via PySpark to calculate marine ecosystem risk scores, and deploys a custom Linear Programming (LP) Simplex Solver to optimally allocate conservation budgets across critical global reef zones.

---

## 🏗️ Pipeline Architecture

```text
+------------------+      +-------------------+      +--------------------+
|  NOAA ERDDAP API | ---> |   Apache Kafka    | ---> |  PySpark Streaming |
| (Real-time SST)  |      | (Confluent Cloud) |      | (Risk Computation) |
+------------------+      +-------------------+      +--------------------+
                                                               |
+------------------+      +-------------------+                v
|Streamlit Frontend| <--- | Custom LP Solver  | <--- +--------------------+
|   (Dashboard)    |      | (Simplex Engine)  |      | Apache Airflow DAG |
+------------------+      +-------------------+      | (Daily Orchestration)|
                                                     +--------------------+
