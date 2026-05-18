# CoralOpt — Coral Reef Conservation Optimizer

An end-to-end data engineering pipeline that collects real-time ocean temperature data from NOAA, processes it with PySpark, and applies Linear Programming to optimally allocate coral reef conservation budgets.

## Pipeline Architecture
## Tech Stack

| Layer | Technology |
|---|---|
| Data Ingestion | Kafka (Confluent Cloud) |
| Stream Processing | PySpark 4.1.1 |
| Orchestration | Apache Airflow |
| Optimization | Simplex LP (from scratch) |
| Dashboard | Streamlit |

## Features

- Real-time SST data from NOAA with automatic mock fallback
- Composite risk scoring per reef zone via PySpark UDF
- Daily automated pipeline via Airflow DAG (6AM)
- Custom Simplex solvers: Dantzig, Bland, Two-Phase
- Interactive budget planning dashboard

## Reef Zones Monitored

| ID | Zone | Location |
|---|---|---|
| GBR_01 | Great Barrier Reef North | -16.0, 145.5 |
| GBR_02 | Great Barrier Reef South | -23.0, 152.0 |
| CRL_01 | Coral Sea | -18.0, 152.5 |
| RED_01 | Red Sea North | 27.0, 34.0 |
| CAR_01 | Caribbean Reef | 15.0, -63.0 |

## Setup

```bash
pip install pyspark kafka-python streamlit pandas numpy matplotlib apache-airflow
```

Run Kafka Producer:
```bash
python Kafka_producer.py
```

Run Spark Processor:
```bash
python spark_processor.py
```

Run Streamlit Dashboard:
```bash
streamlit run app.py
```
