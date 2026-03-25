"""
04_train_export_onnx.py
Блок 3: Обучение CatBoost и деплой модели в ClickHouse
- Читает фичи из ClickHouse
- Обучает бинарный классификатор (tip > 20%)
- Сохраняет модель в формате .cbm
- Копирует в /var/lib/clickhouse/models/ и регистрирует XML-конфиг

Запуск: python scripts/04_train_export_onnx.py
"""

import clickhouse_connect
import numpy as np
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Подключение ────────────────────────────────────────────────
client = clickhouse_connect.get_client(
    host='localhost', port=8123,
    username='default', password=''
)
print("✓ Подключение к ClickHouse")

# ── Выгрузка фич из ClickHouse ─────────────────────────────────
print("\nЗагружаем обучающие данные из ClickHouse...")

df = client.query_df("""
    SELECT
        toHour(pickup_datetime)                              AS hour_of_day,
        toDayOfWeek(pickup_datetime)                         AS day_of_week,
        if(toDayOfWeek(pickup_datetime) IN (6,7), 1, 0)     AS is_weekend,
        round(trip_distance, 2)                              AS trip_distance,
        toFloat32(passenger_count)                           AS passenger_count,
        round(fare_amount, 2)                                AS fare_amount,
        if(tip_amount / nullIf(fare_amount, 0) > 0.2, 1, 0) AS target
    FROM nyc_taxi
    WHERE pickup_datetime BETWEEN '2015-01-01' AND '2015-05-01'
      AND fare_amount > 0
      AND fare_amount < 200
""")

print(f"✓ Загружено {len(df):,} строк")
print(f"  Доля положительных (tip > 20%): {df['target'].mean():.1%}")

FEATURES = ['hour_of_day', 'day_of_week', 'is_weekend',
            'trip_distance', 'passenger_count', 'fare_amount']

X = df[FEATURES].values.astype(np.float32)
y = df['target'].values.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Обучение CatBoost ──────────────────────────────────────────
print("\nОбучение CatBoost...")
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=50
)
model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=20
)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print(f"✓ Обучено")
print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")

# ── Сохранение модели (.cbm) ───────────────────────────────────
os.makedirs('models', exist_ok=True)
model_path = 'models/tip_classifier.cbm'
model.save_model(model_path)
size_kb = os.path.getsize(model_path) / 1024
print(f"\n✓ Модель сохранена: {model_path} ({size_kb:.1f} KB)")

# ── Деплой в ClickHouse ────────────────────────────────────────
print("\nДеплой модели в ClickHouse...")

CH_MODELS_DIR = '/var/lib/clickhouse/models'
CH_CONFIG_DIR = '/etc/clickhouse-server/config.d'

# Создаём директорию если нет
if not os.path.isdir(CH_MODELS_DIR):
    subprocess.run(['sudo', 'mkdir', '-p', CH_MODELS_DIR], check=True)

# Копируем .cbm
r = subprocess.run(
    ['sudo', 'cp', model_path, f'{CH_MODELS_DIR}/tip_classifier.cbm'],
    capture_output=True, text=True
)
if r.returncode == 0:
    print(f"✓ Модель скопирована → {CH_MODELS_DIR}/tip_classifier.cbm")
else:
    print(f"⚠ Ошибка: {r.stderr.strip()}")
    print(f"  Выполните вручную: sudo cp {model_path} {CH_MODELS_DIR}/")

subprocess.run(
    ['sudo', 'chown', 'clickhouse:clickhouse',
     f'{CH_MODELS_DIR}/tip_classifier.cbm'],
    capture_output=True
)

# ── XML-конфиг для регистрации модели ─────────────────────────
# catboostEvaluate() требует регистрации модели через конфиг
config_xml = """<clickhouse>
    <models_config>/etc/clickhouse-server/models/*.xml</models_config>
</clickhouse>
"""
models_xml = """<models>
    <model>
        <type>catboost</type>
        <name>tip_classifier</name>
        <path>/var/lib/clickhouse/models/tip_classifier.cbm</path>
        <lifetime>0</lifetime>
    </model>
</models>
"""

# Пишем конфиги во временные файлы, потом sudo cp
with open('/tmp/catboost_config.xml', 'w') as f:
    f.write(config_xml)
with open('/tmp/tip_classifier.xml', 'w') as f:
    f.write(models_xml)

subprocess.run(['sudo', 'mkdir', '-p', '/etc/clickhouse-server/models'], check=True)
subprocess.run(['sudo', 'cp', '/tmp/catboost_config.xml',
                f'{CH_CONFIG_DIR}/catboost_config.xml'], check=True)
subprocess.run(['sudo', 'cp', '/tmp/tip_classifier.xml',
                '/etc/clickhouse-server/models/tip_classifier.xml'], check=True)
print("✓ XML-конфиги записаны")

# Перезапуск сервера
print("Перезапуск clickhouse-server...")
r = subprocess.run(
    ['sudo', 'systemctl', 'restart', 'clickhouse-server'],
    capture_output=True, text=True
)
if r.returncode == 0:
    print("✓ Сервер перезапущен")
else:
    print(f"⚠ Ошибка перезапуска: {r.stderr.strip()}")
    print("  Выполните вручную: sudo systemctl restart clickhouse-server")

print("\n── Итог ──────────────────────────────────────────────────")
print(f"Модель    : {CH_MODELS_DIR}/tip_classifier.cbm")
print("SQL-функция: catboostEvaluate('/var/lib/clickhouse/models/tip_classifier.cbm', f1, f2, ...)")
print("\n✓ Готово! Далее: clickhouse-client < scripts/05_inference.sql")
