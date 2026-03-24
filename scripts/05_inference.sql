-- =============================================================
-- 05_inference.sql
-- Блок 3: Инференс ONNX-модели прямо в ClickHouse
-- Запуск: clickhouse-client < scripts/05_inference.sql
-- =============================================================

-- Предварительно: убедитесь, что модель зарегистрирована
-- sudo mkdir -p /var/lib/clickhouse/models
-- sudo mkdir -p /etc/clickhouse-server/models

-- sudo cp models/tip_classifier.onnx /var/lib/clickhouse/models/
-- sudo cp models/tip_classifier.xml  /etc/clickhouse-server/models/

-- sudo systemctl restart clickhouse-server

-- ── Шаг 1: Предсказание для одной поездки ─────────────────────
SELECT
    pickup_datetime,
    trip_distance,
    fare_amount,
    tip_amount,
    modelEvaluate(
        'tip_classifier',
        toFloat32(toHour(pickup_datetime)),
        toFloat32(toDayOfWeek(pickup_datetime)),
        toFloat32(if(toDayOfWeek(pickup_datetime) IN (6,7), 1, 0)),
        trip_distance,
        toFloat32(passenger_count),
        fare_amount
    )                                           AS raw_output,
    round(toFloat32(raw_output[2]), 3)          AS tip_probability,
    if(tip_amount / nullIf(fare_amount,0) > 0.2, 1, 0) AS actual_target
FROM nyc_taxi
WHERE pickup_datetime >= '2015-01-15'
  AND fare_amount > 0
LIMIT 1;

-- ── Шаг 2: Batch-инференс за один день (топ-10 по вероятности) ─
SELECT
    pickup_datetime,
    round(trip_distance, 1)                         AS distance_mi,
    round(fare_amount, 2)                           AS fare,
    round(tip_amount, 2)                            AS tip,
    round(toFloat32(modelEvaluate(
        'tip_classifier',
        toFloat32(toHour(pickup_datetime)),
        toFloat32(toDayOfWeek(pickup_datetime)),
        toFloat32(if(toDayOfWeek(pickup_datetime) IN (6,7), 1, 0)),
        trip_distance,
        toFloat32(passenger_count),
        fare_amount
    )[2]), 3)                                       AS tip_probability,
    if(tip_amount / nullIf(fare_amount,0) > 0.2, 1, 0) AS actual
FROM nyc_taxi
WHERE pickup_datetime BETWEEN '2015-01-15' AND '2015-01-16'
  AND fare_amount > 0
ORDER BY tip_probability DESC
LIMIT 10;

-- ── Шаг 3: Сохранение предсказаний в таблицу ──────────────────
INSERT INTO taxi_predictions (trip_id, pickup_datetime, tip_probability)
SELECT
    cityHash64(pickup_datetime, trip_distance, fare_amount) AS trip_id,
    pickup_datetime,
    round(toFloat32(modelEvaluate(
        'tip_classifier',
        toFloat32(toHour(pickup_datetime)),
        toFloat32(toDayOfWeek(pickup_datetime)),
        toFloat32(if(toDayOfWeek(pickup_datetime) IN (6,7), 1, 0)),
        trip_distance,
        toFloat32(passenger_count),
        fare_amount
    )[2]), 4)                                               AS tip_probability
FROM nyc_taxi
WHERE pickup_datetime BETWEEN '2015-01-01' AND '2015-06-30'
  AND fare_amount > 0;

-- ── Шаг 4: Проверка результатов ───────────────────────────────
SELECT
    count()                                         AS total_predictions,
    round(avg(tip_probability), 3)                  AS avg_tip_prob,
    countIf(tip_probability > 0.5)                  AS predicted_positive,
    round(countIf(tip_probability > 0.5) / count(), 3) AS positive_rate
FROM taxi_predictions;

-- ── Шаг 5: Анализ качества — сравнение с фактом ───────────────
SELECT
    round(tip_probability, 1)                       AS prob_bucket,
    count()                                         AS trips,
    round(avg(actual), 3)                           AS actual_positive_rate
FROM (
    SELECT
        p.tip_probability,
        if(t.tip_amount / nullIf(t.fare_amount,0) > 0.2, 1, 0) AS actual
    FROM taxi_predictions p
    JOIN nyc_taxi t USING (trip_id)
    WHERE t.fare_amount > 0
    LIMIT 100000
)
GROUP BY prob_bucket
ORDER BY prob_bucket;
