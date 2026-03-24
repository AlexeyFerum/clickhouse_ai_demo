-- =============================================================
-- 02_feature_engineering.sql
-- Блок 1: Feature Engineering в ClickHouse
-- Запуск: clickhouse-client < scripts/02_feature_engineering.sql
-- =============================================================

-- ── Шаг 1: Базовые временны́е фичи ────────────────────────────
SELECT
    pickup_datetime,
    toHour(pickup_datetime)          AS hour_of_day,
    toDayOfWeek(pickup_datetime)     AS day_of_week,
    if(toDayOfWeek(pickup_datetime) IN (6,7), 1, 0) AS is_weekend,
    trip_distance,
    fare_amount,
    tip_amount,
    round(tip_amount / nullIf(fare_amount, 0), 3) AS tip_ratio,
    if(tip_amount / nullIf(fare_amount, 0) > 0.2, 1, 0) AS target
FROM nyc_taxi
WHERE pickup_datetime >= '2015-01-01'
  AND fare_amount > 0
LIMIT 10;

-- ── Шаг 2: Скользящее среднее (window function) ───────────────
-- Аналог df.rolling('3H')['tip_amount'].mean() в pandas,
-- но работает на всей таблице без выгрузки в память.
SELECT
    pickup_datetime,
    fare_amount,
    round(tip_amount, 2) AS tip_amount,

    round(
        avg(tip_amount) OVER (
            ORDER BY toUInt32(pickup_datetime)
            RANGE BETWEEN 10800 PRECEDING AND CURRENT ROW
        ),
        3
    ) AS rolling_avg_tip_3h
FROM nyc_taxi
WHERE pickup_datetime BETWEEN '2015-01-15' AND '2015-01-15 06:00:00'
ORDER BY pickup_datetime
LIMIT 20;

-- ── Шаг 3: LAG-фича (дистанция предыдущей поездки) ───────────
-- lagInFrame — аналог LAG() в стандартном SQL.
-- Lag-фичи критичны для time-series ML.
SELECT
    pickup_datetime,
    trip_distance,
    round(lagInFrame(trip_distance, 1, 0) OVER (
        ORDER BY pickup_datetime
    ), 2)                            AS prev_trip_distance
FROM nyc_taxi
WHERE pickup_datetime >= '2015-01-01'
  AND pickup_datetime <  '2015-01-02'
ORDER BY pickup_datetime
LIMIT 20;

-- ── Шаг 4: Materialized View как feature store ────────────────
-- Смотрим финализированные агрегаты по часам и дням
-- (MV уже создан в 00_setup.sql и обновляется автоматически)
SELECT
    hour_of_day,
    day_of_week,
    round(avgMerge(avg_tip_state), 3)     AS avg_tip,
    countMerge(trip_count_state)          AS trip_count
FROM taxi_hourly_features
GROUP BY hour_of_day, day_of_week
ORDER BY hour_of_day, day_of_week
LIMIT 24;

-- ── Шаг 5: Итоговый feature-запрос для обучения модели ────────
SELECT
    toHour(pickup_datetime)                                      AS hour_of_day,
    toDayOfWeek(pickup_datetime)                                 AS day_of_week,
    if(toDayOfWeek(pickup_datetime) IN (6,7), 1, 0)              AS is_weekend,
    round(trip_distance, 2)                                      AS trip_distance,
    passenger_count,
    round(fare_amount, 2)                                        AS fare_amount,
    round(
        avg(tip_amount) OVER (
            ORDER BY toUInt32(pickup_datetime)
            RANGE BETWEEN 10800 PRECEDING AND CURRENT ROW
        ),
        3
    )                                                            AS rolling_avg_tip_3h,
    round(
        lagInFrame(trip_distance, 1, 0) OVER (
            ORDER BY pickup_datetime
        ),
        2
    )                                                            AS prev_trip_distance,
    if(tip_amount / nullIf(fare_amount, 0) > 0.2, 1, 0)          AS target
FROM nyc_taxi
WHERE pickup_datetime BETWEEN '2015-01-01' AND '2015-03-01'
  AND fare_amount > 0
ORDER BY pickup_datetime
LIMIT 100000;
