import json
import time
import requests
from datetime import datetime
from kafka import KafkaProducer

# ─────────────────────────────────────────────
# CONFIG — Confluent Cloud
# ─────────────────────────────────────────────
KAFKA_TOPIC      = "coral-raw-data"
KAFKA_BROKER     = "pkc-921jm.us-east-2.aws.confluent.cloud:9092"
KAFKA_API_KEY    = "UQBDEWAQHTU6FFO5"
KAFKA_API_SECRET = "cfltgcxnX6d/CoQMdwv3f2DpmxUN0Dq0RUfLzQ1gXXt+wD1HelWtCAMUwJZ+u2QQ"

# Tọa độ các vùng san hô
REEF_ZONES = [
    {"reef_id": "GBR_01", "name": "Great Barrier Reef North", "lat": -16.0, "lon": 145.5},
    {"reef_id": "GBR_02", "name": "Great Barrier Reef South", "lat": -23.0, "lon": 152.0},
    {"reef_id": "CRL_01", "name": "Coral Sea",                "lat": -18.0, "lon": 152.5},
    {"reef_id": "RED_01", "name": "Red Sea North",             "lat": 27.0,  "lon": 34.0},
    {"reef_id": "CAR_01", "name": "Caribbean Reef",            "lat": 15.0,  "lon": -63.0},
]

# ─────────────────────────────────────────────
# PULL SST TỪ NOAA ERDDAP
# ─────────────────────────────────────────────

def fetch_sst(lat, lon):
    today = datetime.utcnow().strftime("%Y-%m-%dT00:00:00Z")
    url = (
        f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.json"
        f"?analysed_sst[({today})]"
        f"[({lat:.4f}):1:({lat:.4f})]"
        f"[({lon:.4f}):1:({lon:.4f})]"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            sst_kelvin = data["table"]["rows"][0][3]
            return round(sst_kelvin - 273.15, 2)
        else:
            print(f"  NOAA API lỗi {resp.status_code} → dùng mock data")
            return None
    except Exception as e:
        print(f"  Lỗi kết nối NOAA: {e} → dùng mock data")
        return None

def mock_sst(lat):
    import numpy as np
    base = 28.0 - abs(lat) * 0.1
    return round(base + np.random.normal(0, 0.8), 2)

# ─────────────────────────────────────────────
# TÍNH BLEACHING RISK
# ─────────────────────────────────────────────

def calc_bleaching_risk(sst):
    if sst <= 29.0:
        return 0.0
    elif sst >= 31.0:
        return 1.0
    else:
        return round((sst - 29.0) / 2.0, 4)

# ─────────────────────────────────────────────
# KAFKA PRODUCER — Confluent Cloud
# ─────────────────────────────────────────────

def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        security_protocol="SASL_SSL",
        sasl_mechanism="PLAIN",
        sasl_plain_username=KAFKA_API_KEY,
        sasl_plain_password=KAFKA_API_SECRET,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

def produce_data():
    producer = create_producer()
    print(f"✅ Kafka Producer kết nối Confluent Cloud thành công!")
    print(f"📡 Bắt đầu pull data NOAA → topic '{KAFKA_TOPIC}'\n")

    while True:
        timestamp = datetime.utcnow().isoformat()

        for zone in REEF_ZONES:
            print(f"🌊 Fetching {zone['name']} ({zone['lat']}, {zone['lon']})...")

            sst = fetch_sst(zone["lat"], zone["lon"])
            if sst is None:
                sst = mock_sst(zone["lat"])
                source = "mock"
            else:
                source = "NOAA_ERDDAP"

            bleaching_risk = calc_bleaching_risk(sst)

            message = {
                "reef_id"        : zone["reef_id"],
                "name"           : zone["name"],
                "lat"            : zone["lat"],
                "lon"            : zone["lon"],
                "sst_celsius"    : sst,
                "bleaching_risk" : bleaching_risk,
                "data_source"    : source,
                "timestamp"      : timestamp
            }

            producer.send(KAFKA_TOPIC, value=message)
            print(f"  ✅ Sent → SST: {sst}°C | Risk: {bleaching_risk} | Source: {source}")

        producer.flush()
        print(f"\n⏳ Chờ 60 giây...\n")
        time.sleep(60)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    produce_data()