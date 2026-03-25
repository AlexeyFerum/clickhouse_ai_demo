-- =============================================================
-- 00_setup.sql
-- Создание всех таблиц для демо вебинара
-- Запуск: clickhouse-client < scripts/00_setup.sql
-- =============================================================

-- Блок 1: Сырые данные поездок NYC Taxi
CREATE TABLE IF NOT EXISTS nyc_taxi (
    pickup_datetime   DateTime,
    dropoff_datetime  DateTime,
    passenger_count   UInt8,
    trip_distance     Float32,
    pickup_lon        Float32,
    pickup_lat        Float32,
    dropoff_lon       Float32,
    dropoff_lat       Float32,
    fare_amount       Float32,
    tip_amount        Float32,
    total_amount      Float32,
    pickup_zone_id    UInt16,
    dropoff_zone_id   UInt16
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(pickup_datetime)
ORDER BY (pickup_datetime)
SETTINGS index_granularity = 8192;

-- Блок 1: Таблица агрегированных фич (результат MV)
CREATE TABLE IF NOT EXISTS taxi_hourly_features (
    hour_of_day       UInt8,
    day_of_week       UInt8,
    avg_tip_state     AggregateFunction(avg, Float32),
    trip_count_state  AggregateFunction(count)
)
ENGINE = AggregatingMergeTree()
ORDER BY (hour_of_day, day_of_week);

-- Materialized View: автообновление фич при каждой вставке
CREATE MATERIALIZED VIEW IF NOT EXISTS taxi_hourly_features_mv
TO taxi_hourly_features
AS
SELECT
    toHour(pickup_datetime)      AS hour_of_day,
    toDayOfWeek(pickup_datetime) AS day_of_week,
    avgState(tip_amount)         AS avg_tip_state,
    countState()                 AS trip_count_state
FROM nyc_taxi
GROUP BY hour_of_day, day_of_week;

-- Блок 2: Зоны NYC с векторными эмбеддингами
CREATE TABLE IF NOT EXISTS taxi_zones (
    zone_id    UInt16,
    zone_name  String,
    borough    String,
    zone_type  String,
    embedding  Array(Float32),
    INDEX ann_idx embedding
        TYPE vector_similarity('hnsw', 'cosineDistance')
        GRANULARITY 1
)
ENGINE = MergeTree()
ORDER BY zone_id;

-- Блок 3: Таблица предсказаний модели
CREATE TABLE IF NOT EXISTS taxi_predictions (
    trip_id          UInt64,
    pickup_datetime  DateTime,
    tip_probability  Float32,
    predicted_at     DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(pickup_datetime)
ORDER BY (pickup_datetime, trip_id);

SELECT 'Tables created successfully' AS status;
