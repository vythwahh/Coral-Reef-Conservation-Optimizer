# CoralOpt — Coral Reef Conservation Optimizer

An enterprise-grade, end-to-end data engineering and operations research pipeline. CoralOpt ingests real-time sea surface temperature (SST) data from the NOAA ERDDAP API, processes streaming data via PySpark to calculate marine ecosystem risk scores, and deploys a custom Linear Programming (LP) Simplex Solver to optimally allocate conservation budgets across critical global reef zones.

---

## Pipeline Architecture

<pre>
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
</pre>

---

## Tech Stack & Infrastructure

| Layer | Technology | Operational Function |
|---|---|---|
| **Data Ingestion** | Apache Kafka (Confluent Cloud) | Real-time message queuing and event-driven data streaming. |
| **Stream Processing**| PySpark 4.1.1 | Distributed windowed aggregations and composite reef risk scoring. |
| **Orchestration** | Apache Airflow | Daily automated pipeline triggering (6 AM UTC) and workflow tracking. |
| **Operations Research**| Custom Simplex LP Engine | Mathematical optimization written from scratch (Bland, Dantzig, 2-Phase). |
| **Visualization** | Streamlit | Interactive data platform for environmental budget allocation planning. |

---

## Mathematical Formulation (LP Model)

The core optimization engine allocates financial budgets to maximize ecological resilience across $N$ monitored reef zones, subject to realistic capital and resource constraints.

### 1. Objective Function
Maximize the total mitigated risk score across all reef zones:

$$\max \sum_{i=1}^{N} R_i \cdot x_i$$

Where:
- $R_i$: The calculated composite ecological risk score for reef zone $i$ (derived via PySpark pipeline).
- $x_i$: The financial budget allocated to reef zone $i$ (Decision Variable).

### 2. Operational Constraints
- **Total Budget Limitation:** Total spent cannot exceed the maximum available conservation fund ($B$):

$$\sum_{i=1}^{N} x_i \leq B$$

- **Zone Cap Constraints:** To prevent monopolization of funds, each reef zone has a maximum absorption capacity ($M_i$):

$$0 \leq x_i \leq M_i \quad \forall i \in \{1, \dots, N\}$$

---

## Key Engineering Features

- **Resilient Network Ingestion:** Real-time SST data streaming from NOAA API with automated schema validation and a randomized mock data generator fallback for network resilience.
- **Distributed Risk Aggregation:** Utilizes PySpark User Defined Functions (UDFs) to process windowed sliding metrics, capturing thermal anomalies indicative of coral bleaching events.
- **Custom Optimization Solvers:** Implemented foundational Operations Research solvers from scratch, including **Dantzig's rule** for standard pivoting, **Bland's rule** to eliminate cycling loops, and **Two-Phase Simplex** for artificial initial bases.
- **Production-Ready Guardrails:** Built-in PyTest suites covering extreme constraint edge cases (e.g., zero budget, unfeasible targets) and structured system ---

 

## Execution Guide

### 1. Environment Setup

<pre>
# Clone the repository
git clone https://github.com/vythwahh/Coral-Reef-Conservation-Optimizer.git
cd Coral-Reef-Conservation-Optimizer

# Install environment dependencies
pip install -r requirements.txt
</pre>

### 2. Running the Ecosystem Pipeline

Execute the following processes in separate terminal instances or trigger them natively via the orchestrated Airflow DAG:

<pre>
# Step 1: Boot up real-time ingestion from NOAA to Kafka Cloud
python src/kafka_producer.py
</pre>

<pre>
# Step 2: Spin up the Spark cluster to execute distributed stream processing
python src/spark_processor.py
</pre>

<pre>
# Step 3: Launch the Optimization User Dashboard
streamlit run src/app.py
</pre>
---
