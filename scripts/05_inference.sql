-- =============================================================
-- 05_inference.sql
-- Блок 3: Инференс CatBoost-модели в ClickHouse
--
-- Используется catboostEvaluate() — встроенная функция ClickHouse.
-- Модель должна быть задеплоена скриптом 04_train_export_onnx.py.
--
-- catboostEvaluate() возвращает Float64 — логит (сырой скор).
-- Для получения вероятности применяем sigmoid: 1 / (1 + exp(-x))
--
-- Запуск: clickhouse-client < scripts/05_inference.sql
-- =============================================================

-- ── Шаг 1: Предсказание для одной поездки ─────────────────────
SELECT
    pickup_datetime,
    round(trip_distance, 1)  AS distance,
    round(fare_amount, 2)    AS fare,
    round(tip_amount, 2)     AS tip,
    round(1 / (1 + exp(-catboostEvaluate(
        '/var/lib/clickhouse/models/tip_classifier.cbm',
        toFloat32(toHour(pickup_datetime)),
        toFloat32(toDayOfWeek(pickup_datetime)),
        toFloat32(if(toDayOfWeek(pickup_datetime) IN (6,7), 1, 0)),
        trip_distance,
        toFloat32(passenger_count),
        fare_amount
    ))), 3)                                              AS tip_probability,
    if(tip_amount / nullIf(fare_amount, 0) > 0.2, 1, 0) AS actual
FROM nyc_taxi
WHERE pickup_datetime >= '2015-01-15'
  AND fare_amount > 0
LIMIT 1;

-- ── Шаг 2: Топ-10 поездок за день по вероятности чаевых ───────
SELECT
    pickup_datetime,
    round(trip_distance, 1)                              AS distance,
    round(fare_amount, 2)                                AS fare,
    round(tip_amount, 2)                                 AS tip,
    round(1 / (1 + exp(-catboostEvaluate(
        '/var/lib/clickhouse/models/tip_classifier.cbm',
        toFloat32(toHour(pickup_datetime)),
        toFloat32(toDayOfWeek(pickup_datetime)),
        toFloat32(if(toDayOfWeek(pickup_datetime) IN (6,7), 1, 0)),
        trip_distance,
        toFloat32(passenger_count),
        fare_amount
    ))), 3)                                              AS tip_probability,
    if(tip_amount / nullIf(fare_amount, 0) > 0.2, 1, 0) AS actual
FROM nyc_taxi
WHERE pickup_datetime BETWEEN '2015-01-15' AND '2015-01-16'
  AND fare_amount > 0
ORDER BY tip_probability DESC
LIMIT 10;

-- ── Шаг 3: Batch-инференс и запись предсказаний ────────────────
INSERT INTO taxi_predictions (trip_id, pickup_datetime, tip_probability)
SELECT
    cityHash64(pickup_datetime, trip_distance, fare_amount) AS trip_id,
    pickup_datetime,
    round(1 / (1 + exp(-catboostEvaluate(
        '/var/lib/clickhouse/models/tip_classifier.cbm',
        toFloat32(toHour(pickup_datetime)),
        toFloat32(toDayOfWeek(pickup_datetime)),
        toFloat32(if(toDayOfWeek(pickup_datetime) IN (6,7), 1, 0)),
        trip_distance,
        toFloat32(passenger_count),
        fare_amount
    ))), 4)                                                AS tip_probability
FROM nyc_taxi
WHERE pickup_datetime BETWEEN '2015-01-01' AND '2015-06-30'
  AND fare_amount > 0;

-- ── Шаг 4: Статистика предсказаний ────────────────────────────
SELECT
    count()                                                AS total,
    round(avg(tip_probability), 3)                         AS avg_prob,
    countIf(tip_probability > 0.5)                         AS predicted_positive,
    round(countIf(tip_probability > 0.5) / count(), 3)     AS positive_rate
FROM taxi_predictions;

-- ── Шаг 5: Калибровка — факт vs предсказание по бакетам ───────
-- trip_id воссоздаётся через тот же cityHash64, что и при вставке
SELECT
    round(tip_probability, 1)                              AS prob_bucket,
    count()                                                AS trips,
    round(avg(actual), 3)                                  AS actual_positive_rate
FROM (
    SELECT
        p.tip_probability,
        if(t.tip_amount / nullIf(t.fare_amount, 0) > 0.2, 1, 0) AS actual
    FROM taxi_predictions p
    JOIN nyc_taxi t
      ON p.trip_id = cityHash64(t.pickup_datetime, t.trip_distance, t.fare_amount)
    WHERE t.fare_amount > 0
    LIMIT 100000
)
GROUP BY prob_bucket
ORDER BY prob_bucket;
