"""
01_seed_data.py
Наполнение тестовыми данными: NYC Taxi поездки + зоны
Генерирует ~500K поездок и 50 зон NYC с реалистичными значениями.

Запуск: python scripts/01_seed_data.py
"""

import clickhouse_connect
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# ── Подключение ────────────────────────────────────────────────
client = clickhouse_connect.get_client(
    host='localhost',
    port=8123,
    username='default',
    password=''
)
print("✓ Подключение к ClickHouse")

# ── NYC Taxi Zones (реальные зоны, упрощённый список) ──────────
ZONES = [
    (1,  "Newark Airport",        "EWR",       "Airport",      -74.175, 40.690),
    (4,  "Alphabet City",         "Manhattan", "Residential",  -73.978, 40.726),
    (12, "Battery Park",          "Manhattan", "Business",     -74.017, 40.703),
    (13, "Battery Park City",     "Manhattan", "Business",     -74.016, 40.710),
    (24, "Bloomingdale",          "Manhattan", "Residential",  -73.983, 40.790),
    (25, "Boerum Hill",           "Brooklyn",  "Residential",  -73.985, 40.686),
    (33, "Brooklyn Heights",      "Brooklyn",  "Residential",  -73.997, 40.697),
    (36, "Bushwick North",        "Brooklyn",  "Residential",  -73.921, 40.702),
    (41, "Carroll Gardens",       "Brooklyn",  "Residential",  -73.999, 40.677),
    (42, "Central Harlem",        "Manhattan", "Residential",  -73.944, 40.810),
    (43, "Central Park",          "Manhattan", "Green",        -73.965, 40.785),
    (45, "Chelsea",               "Manhattan", "Mixed",        -74.003, 40.748),
    (48, "Clinton East",          "Manhattan", "Mixed",        -73.990, 40.762),
    (50, "Clinton Hill",          "Brooklyn",  "Residential",  -73.966, 40.695),
    (68, "East Chelsea",          "Manhattan", "Mixed",        -74.000, 40.745),
    (74, "East Harlem North",     "Manhattan", "Residential",  -73.939, 40.798),
    (75, "East Harlem South",     "Manhattan", "Residential",  -73.942, 40.793),
    (79, "East Village",          "Manhattan", "Residential",  -73.982, 40.726),
    (82, "Elmhurst",              "Queens",    "Residential",  -73.882, 40.737),
    (87, "Financial District",    "Manhattan", "Business",     -74.010, 40.708),
    (88, "Flatbush West",         "Brooklyn",  "Residential",  -73.960, 40.641),
    (90, "Flushing",              "Queens",    "Mixed",        -73.833, 40.767),
    (107,"Greenwich Village",     "Manhattan", "Mixed",        -74.002, 40.733),
    (113,"Hamilton Heights",      "Manhattan", "Residential",  -73.949, 40.822),
    (114","Harlem",               "Manhattan", "Residential",  -73.943, 40.812),
    (116,"Hell's Kitchen North",  "Manhattan", "Mixed",        -73.993, 40.768),
    (120,"Highbridge",            "Bronx",     "Residential",  -73.925, 40.837),
    (125,"Howard Beach",          "Queens",    "Residential",  -73.843, 40.659),
    (132,"JFK Airport",           "Queens",    "Airport",      -73.778, 40.641),
    (138","LaGuardia Airport",    "Queens",    "Airport",      -73.873, 40.776),
    (140,"Lenox Hill East",       "Manhattan", "Residential",  -73.959, 40.768),
    (141,"Lenox Hill West",       "Manhattan", "Residential",  -73.966, 40.770),
    (142,"Lincoln Square East",   "Manhattan", "Mixed",        -73.986, 40.773),
    (143,"Lincoln Square West",   "Manhattan", "Mixed",        -73.990, 40.775),
    (144,"Little Italy/NoLiTa",   "Manhattan", "Mixed",        -73.997, 40.722),
    (148,"Lower East Side",       "Manhattan", "Residential",  -73.985, 40.715),
    (151,"Manhattan Valley",      "Manhattan", "Residential",  -73.966, 40.797),
    (152,"Manhattanville",        "Manhattan", "Residential",  -73.955, 40.819),
    (158","Meatpacking/West",     "Manhattan", "Mixed",        -74.008, 40.740),
    (161,"Midtown Center",        "Manhattan", "Business",     -73.984, 40.757),
    (162,"Midtown East",          "Manhattan", "Business",     -73.974, 40.755),
    (163,"Midtown North",         "Manhattan", "Business",     -73.983, 40.763),
    (164,"Midtown South",         "Manhattan", "Business",     -73.988, 40.750),
    (170,"Murray Hill",           "Manhattan", "Mixed",        -73.978, 40.748),
    (186","Penn Station/Madison", "Manhattan", "Transit",      -73.994, 40.751),
    (194,"Prospect-Lefferts",     "Brooklyn",  "Residential",  -73.957, 40.652),
    (202,"Ridgewood",             "Queens",    "Residential",  -73.904, 40.704),
    (209","SoHo",                 "Manhattan", "Mixed",        -74.003, 40.723),
    (261,"World Trade Center",    "Manhattan", "Business",     -74.011, 40.712),
    (263","Yorkville East",       "Manhattan", "Residential",  -73.950, 40.774),
]

zone_ids   = [z[0] for z in ZONES]
zone_lons  = {z[0]: z[4] for z in ZONES}
zone_lats  = {z[0]: z[5] for z in ZONES}

# ── Генерация поездок ─────────────────────────────────────────
print("Генерация данных поездок NYC Taxi...")

np.random.seed(42)
random.seed(42)

N = 500_000
start_date = datetime(2015, 1, 1)
end_date   = datetime(2015, 6, 30)
date_range_seconds = int((end_date - start_date).total_seconds())

# Временны́е метки с суточным профилем (больше поездок вечером)
raw_times = np.random.randint(0, date_range_seconds, N)
hour_bias = np.random.choice(range(24), N, p=[
    0.01,0.01,0.01,0.01,0.01,0.02,  # 0–5
    0.03,0.05,0.06,0.05,0.04,0.05,  # 6–11
    0.05,0.05,0.04,0.05,0.06,0.07,  # 12–17
    0.08,0.07,0.06,0.05,0.04,0.03   # 18–23
])
pickup_times = [
    start_date + timedelta(seconds=int(t))
    for t in raw_times
]

# Дистанция: логнормальное распределение (реалистично для такси)
trip_distance = np.random.lognormal(mean=0.8, sigma=0.8, size=N).clip(0.1, 50.0)

# Длительность поездки: зависит от дистанции + шум
duration_minutes = (trip_distance * 4 + np.random.normal(5, 3, N)).clip(1, 120)

dropoff_times = [
    pt + timedelta(minutes=float(dm))
    for pt, dm in zip(pickup_times, duration_minutes)
]

# Тарифы
fare_amount  = (trip_distance * 2.5 + 3.0 + np.random.normal(0, 1.5, N)).clip(2.5, 200)
passenger_count = np.random.choice([1,2,3,4], N, p=[0.6,0.2,0.12,0.08]).astype(np.uint8)

# Чаевые: зависят от часа дня и дистанции (ночью и за аэропорт дают больше)
hour_arr = np.array([pt.hour for pt in pickup_times])
airport_pickup = np.isin(
    np.random.choice(zone_ids, N),
    [1, 132, 138]  # Newark, JFK, LaGuardia
).astype(float)

base_tip_ratio = (
    0.15
    + 0.05 * ((hour_arr >= 22) | (hour_arr <= 5)).astype(float)  # ночные надбавки
    + 0.08 * airport_pickup                                        # аэропортные
    + np.random.normal(0, 0.08, N)                                 # случайность
).clip(0, 0.5)

tip_amount   = (fare_amount * base_tip_ratio).clip(0, 50)
total_amount = fare_amount + tip_amount + np.random.choice([0, 0.5, 1.0], N)

# Зоны (pickup/dropoff)
pickup_zone_ids  = np.random.choice(zone_ids, N).astype(np.uint16)
dropoff_zone_ids = np.random.choice(zone_ids, N).astype(np.uint16)

pickup_lons = np.array([zone_lons[z] + np.random.normal(0, 0.005) for z in pickup_zone_ids], dtype=np.float32)
pickup_lats = np.array([zone_lats[z] + np.random.normal(0, 0.005) for z in pickup_zone_ids], dtype=np.float32)
dropoff_lons = np.array([zone_lons[z] + np.random.normal(0, 0.005) for z in dropoff_zone_ids], dtype=np.float32)
dropoff_lats = np.array([zone_lats[z] + np.random.normal(0, 0.005) for z in dropoff_zone_ids], dtype=np.float32)

# ── Вставка батчами ───────────────────────────────────────────
BATCH = 50_000
inserted = 0

print(f"Вставка {N:,} строк батчами по {BATCH:,}...")
for i in range(0, N, BATCH):
    sl = slice(i, i + BATCH)
    rows = list(zip(
        pickup_times[sl],
        dropoff_times[sl],
        passenger_count[sl].tolist(),
        trip_distance[sl].astype(np.float32).tolist(),
        pickup_lons[sl].tolist(),
        pickup_lats[sl].tolist(),
        dropoff_lons[sl].tolist(),
        dropoff_lats[sl].tolist(),
        fare_amount[sl].astype(np.float32).tolist(),
        tip_amount[sl].astype(np.float32).tolist(),
        total_amount[sl].astype(np.float32).tolist(),
        pickup_zone_ids[sl].tolist(),
        dropoff_zone_ids[sl].tolist(),
    ))
    client.insert('nyc_taxi', rows, column_names=[
        'pickup_datetime','dropoff_datetime','passenger_count',
        'trip_distance','pickup_lon','pickup_lat',
        'dropoff_lon','dropoff_lat',
        'fare_amount','tip_amount','total_amount',
        'pickup_zone_id','dropoff_zone_id'
    ])
    inserted += len(rows)
    print(f"  {inserted:>7,} / {N:,} строк вставлено", end='\r')

print(f"\n✓ NYC Taxi: {inserted:,} поездок вставлено")

# ── Вставка зон ───────────────────────────────────────────────
print("\nВставка зон NYC (без эмбеддингов — эмбеддинги генерирует 03_vector_search.py)...")

zone_rows = [
    (z[0], z[1], z[2], z[3], [])   # embedding пустой — заполнится в следующем скрипте
    for z in ZONES
]
client.insert('taxi_zones', zone_rows,
              column_names=['zone_id','zone_name','borough','zone_type','embedding'])

print(f"✓ Зоны NYC: {len(ZONES)} записей вставлено")

# ── Проверка ──────────────────────────────────────────────────
count = client.query("SELECT count() FROM nyc_taxi").result_rows[0][0]
print(f"\n── Итог ──")
print(f"nyc_taxi:    {count:>10,} строк")
print(f"taxi_zones:  {len(ZONES):>10,} зон")
print(f"\nДалее: python scripts/03_vector_search.py")
