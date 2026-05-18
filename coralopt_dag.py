from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import subprocess
import sys

CORALOPT_DIR = "/Users/vythu/Documents/CoralOpt"
PYTHON_BIN   = sys.executable

DEFAULT_ARGS = {
    "owner"           : "coralopt",
    "retries"         : 2,
    "retry_delay"     : timedelta(minutes=5),
    "email_on_failure": False,
}

def run_kafka_producer(**context):
    import json, numpy as np
    from datetime import datetime, UTC
    from kafka import KafkaProducer

    KAFKA_BROKER     = "pkc-921jm.us-east-2.aws.confluent.cloud:9092"
    KAFKA_API_KEY    = "UQBDEWAQHTU6FFO5"
    KAFKA_API_SECRET = "cfltgcxnX6d/CoQMdwv3f2DpmxUN0Dq0RUfLzQ1gXXt+wD1HelWtCAMUwJZ+u2QQ"
    KAFKA_TOPIC      = "coral-raw-data"

    REEF_ZONES = [
        {"reef_id": "GBR_01", "name": "Great Barrier Reef North", "lat": -16.0, "lon": 145.5},
        {"reef_id": "GBR_02", "name": "Great Barrier Reef South", "lat": -23.0, "lon": 152.0},
        {"reef_id": "CRL_01", "name": "Coral Sea",                "lat": -18.0, "lon": 152.5},
        {"reef_id": "RED_01", "name": "Red Sea North",             "lat": 27.0,  "lon": 34.0},
        {"reef_id": "CAR_01", "name": "Caribbean Reef",            "lat": 15.0,  "lon": -63.0},
    ]

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        security_protocol="SASL_SSL",
        sasl_mechanism="PLAIN",
        sasl_plain_username=KAFKA_API_KEY,
        sasl_plain_password=KAFKA_API_SECRET,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    timestamp = datetime.now(UTC).isoformat()
    for zone in REEF_ZONES:
        sst  = round(28.0 - abs(zone["lat"]) * 0.1 + np.random.normal(0, 0.8), 2)
        risk = max(0.0, min(1.0, (sst - 29.0) / 2.0)) if sst > 29 else 0.0
        msg  = {
            "reef_id": zone["reef_id"], "name": zone["name"],
            "lat": zone["lat"], "lon": zone["lon"],
            "sst_celsius": sst, "bleaching_risk": round(risk, 4),
            "data_source": "mock", "timestamp": timestamp
        }
        producer.send(KAFKA_TOPIC, value=msg)
        print(f"Sent: {zone['reef_id']} | SST={sst} | Risk={risk:.4f}")

    producer.flush()
    producer.close()
    print("Kafka Producer hoàn thành!")


def run_spark_processor(**context):
    result = subprocess.run(
        [PYTHON_BIN, f"{CORALOPT_DIR}/spark_processor.py"],
        capture_output=True, text=True, cwd=CORALOPT_DIR
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception(f"Spark Processor thất bại! Code: {result.returncode}")
    print("Spark Processor hoàn thành!")


def check_results(**context):
    import os
    output_path = f"{CORALOPT_DIR}/output/risk_scores"
    if not os.path.exists(output_path):
        raise Exception(f"Không tìm thấy output: {output_path}")
    files = os.listdir(output_path)
    print(f"Pipeline hoàn thành! {len(files)} files trong {output_path}")


def run_lp_solver_task(**context):
    import sys
    sys.path.insert(0, CORALOPT_DIR)
    from lp_solver_pipeline import run_lp_solver

    budget = 500_000
    result_df, status, summary = run_lp_solver(budget=budget)

    if status != "Optimal":
        raise Exception(f"LP Solver thất bại: {status}")
    print(f"LP Solver hoàn thành! Impact: {summary['total_impact']:.4f}")


with DAG(
    dag_id="coralopt_pipeline",
    description="CoralOpt: Kafka -> Spark -> Check -> LP Solver pipeline",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * *",
    start_date=datetime(2026, 5, 17),
    catchup=False,
    tags=["coralopt", "coral", "pipeline"],
) as dag:

    task_kafka = PythonOperator(
        task_id="kafka_producer",
        python_callable=run_kafka_producer,
    )

    task_spark = PythonOperator(
        task_id="spark_processor",
        python_callable=run_spark_processor,
    )

    task_check = PythonOperator(
        task_id="check_results",
        python_callable=check_results,
    )

    task_lp = PythonOperator(
        task_id="lp_solver",
        python_callable=run_lp_solver_task,
    )

    # Kafka → Spark → Check → LP Solver
    task_kafka >> task_spark >> task_check >> task_lp