import json
from datetime import datetime, UTC
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, current_timestamp, round as spark_round
)
from pyspark.sql.types import (
    StructType, StructField,
    StringType, FloatType, DoubleType
)

 
# CONFIG
 
KAFKA_BROKER     = "pkc-921jm.us-east-2.aws.confluent.cloud:9092"
KAFKA_API_KEY    = "UQBDEWAQHTU6FFO5"
KAFKA_API_SECRET = "cfltgcxnX6d/CoQMdwv3f2DpmxUN0Dq0RUfLzQ1gXXt+wD1HelWtCAMUwJZ+u2QQ"
KAFKA_TOPIC      = "coral-raw-data"
OUTPUT_PATH      = "./output/risk_scores"

 
# SCHEMA — matches Kafka Producer message format
 
MESSAGE_SCHEMA = StructType([
    StructField("reef_id",        StringType(), True),
    StructField("name",           StringType(), True),
    StructField("lat",            DoubleType(), True),
    StructField("lon",            DoubleType(), True),
    StructField("sst_celsius",    FloatType(),  True),
    StructField("bleaching_risk", FloatType(),  True),
    StructField("data_source",    StringType(), True),
    StructField("timestamp",      StringType(), True),
])

 
# RISK SCORE UDF
 

def compute_risk_score(sst, bleaching_risk):
    """
    Composite risk score from SST and bleaching risk.

    Formula:
        risk_score = 0.4 * sst_factor + 0.6 * bleaching_risk

    Where:
        sst_factor = clip((SST - 27) / 4, 0, 1)
        → SST <= 27°C: factor = 0  (normal)
        → SST >= 31°C: factor = 1  (critical)
    """
    if sst is None or bleaching_risk is None:
        return 0.0
    sst_factor = max(0.0, min(1.0, (sst - 27.0) / 4.0))
    risk_score = 0.4 * sst_factor + 0.6 * bleaching_risk
    return round(float(risk_score), 4)

risk_score_udf = udf(compute_risk_score, FloatType())

 
# SPARK SESSION
 

def create_spark_session():
    """Initialize SparkSession with Kafka connector."""
    return (
        SparkSession.builder
        .appName("CoralOpt_SparkProcessor")
        .config("spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.0")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

 
# READ FROM KAFKA
 

def read_from_kafka(spark):
    """
    Read data from Confluent Cloud Kafka topic.
    Authenticated via SASL_SSL / PLAIN mechanism.
    """
    jaas_config = (
        f"org.apache.kafka.common.security.plain.PlainLoginModule required "
        f"username='{KAFKA_API_KEY}' "
        f"password='{KAFKA_API_SECRET}';"
    )

    return (
        spark.read
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("kafka.security.protocol", "SASL_SSL")
        .option("kafka.sasl.mechanism", "PLAIN")
        .option("kafka.sasl.jaas.config", jaas_config)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("endingOffsets", "latest")
        .load()
    )

 
# PROCESS DATA
 

def process_data(raw_df):
    """
    Parse JSON messages → compute risk score → aggregate by reef zone.
    Returns (scored_df, aggregated_df).
    """
    from pyspark.sql.functions import from_json

    # Parse JSON from Kafka value field
    parsed_df = (
        raw_df
        .select(
            from_json(col("value").cast("string"), MESSAGE_SCHEMA).alias("data")
        )
        .select("data.*")
        .filter(col("reef_id").isNotNull())
    )

    # Compute composite risk score via UDF
    scored_df = (
        parsed_df
        .withColumn("risk_score", risk_score_udf(col("sst_celsius"), col("bleaching_risk")))
        .withColumn("processed_at", current_timestamp())
    )

    # Aggregate by reef zone — average SST, bleaching risk, risk score
    aggregated_df = (
        scored_df
        .groupBy("reef_id", "name", "lat", "lon")
        .agg({
            "sst_celsius"    : "avg",
            "bleaching_risk" : "avg",
            "risk_score"     : "avg"
        })
        .withColumnRenamed("avg(sst_celsius)",    "avg_sst")
        .withColumnRenamed("avg(bleaching_risk)", "avg_bleaching_risk")
        .withColumnRenamed("avg(risk_score)",     "avg_risk_score")
        .withColumn("avg_sst",            spark_round(col("avg_sst"), 4))
        .withColumn("avg_bleaching_risk", spark_round(col("avg_bleaching_risk"), 4))
        .withColumn("avg_risk_score",     spark_round(col("avg_risk_score"), 4))
        .orderBy(col("avg_risk_score").desc())
    )

    return scored_df, aggregated_df

 
# SAVE RESULTS
 

def save_results(aggregated_df):
    """Save aggregated results to Parquet format."""
    import os
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    aggregated_df.write \
        .mode("overwrite") \
        .parquet(OUTPUT_PATH)

    print(f"\n Results saved to: {OUTPUT_PATH}")

 
# MAIN
 

def main():
    print(" Starting Spark Processor")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    print(" Reading data from Kafka topic: coral-raw-data ")
    raw_df = read_from_kafka(spark)

    print("  Processing data and computing risk scores ")
    scored_df, aggregated_df = process_data(raw_df)

    print("\n Risk Score by reef zone:")
    print("=" * 60)
    aggregated_df.show(truncate=False)

    print("\n Message-level detail (top 20):")
    print("=" * 60)
    scored_df.select(
        "reef_id", "name", "sst_celsius",
        "bleaching_risk", "risk_score", "data_source"
    ).show(20, truncate=False)

    save_results(aggregated_df)

    spark.stop()
    print("\n Spark Processor completed successfully!")


if __name__ == "__main__":
    main()
