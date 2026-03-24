"""
04_train_export_onnx.py
Блок 3: Обучение LightGBM и экспорт в ONNX для ClickHouse
- Читает фичи из ClickHouse
- Обучает бинарный классификатор (tip > 20%)
- Экспортирует в ONNX
- Кладёт модель в директорию ClickHouse

Запуск: python scripts/04_train_export_onnx.py
"""

import clickhouse_connect
import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Подключение ────────────────────────────────────────────────
client = clickhouse_connect.get_client(
    host='localhost', port=8123,
    username='default', password=''
)
print("✓ Подключение к ClickHouse")

# ── Выгрузка фич из ClickHouse ────────────────────────────────
print("\nЗагружаем обучающие данные из ClickHouse...")

df = client.query_df("""
    SELECT
        toHour(pickup_datetime)                                     AS hour_of_day,
        toDayOfWeek(pickup_datetime)                                AS day_of_week,
        if(toDayOfWeek(pickup_datetime) IN (6,7), 1, 0)            AS is_weekend,
        round(trip_distance, 2)                                     AS trip_distance,
        toFloat32(passenger_count)                                  AS passenger_count,
        round(fare_amount, 2)                                       AS fare_amount,
        if(tip_amount / nullIf(fare_amount, 0) > 0.2, 1, 0)        AS target
    FROM nyc_taxi
    WHERE pickup_datetime BETWEEN '2015-01-01' AND '2015-05-01'
      AND fare_amount > 0
      AND fare_amount < 200
""")

print(f"✓ Загружено {len(df):,} строк")
print(f"  Целевая переменная: {df['target'].mean():.1%} положительных (tip > 20%)")

FEATURES = ['hour_of_day','day_of_week','is_weekend',
            'trip_distance','passenger_count','fare_amount']

X = df[FEATURES].values.astype(np.float32)
y = df['target'].values.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Обучение LightGBM ─────────────────────────────────────────
print("\nОбучение LightGBM...")
try:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(20, verbose=False),
                         lgb.log_evaluation(50)])

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"✓ Обучено за {model.best_iteration_} итераций")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")

    # ── Экспорт в ONNX ─────────────────────────────────────────
    print("\nЭкспорт в ONNX...")
    try:
        from onnxmltools import convert_lightgbm
        from onnxmltools.utils import save_model
        from onnxmltools.convert.common.data_types import FloatTensorType

        onnx_model = convert_lightgbm(
            model.booster_,
            initial_types=[('features', FloatTensorType([None, len(FEATURES)]))]
        )

        os.makedirs('models', exist_ok=True)
        onnx_path = 'models/tip_classifier.onnx'
        save_model(onnx_model, onnx_path)
        size_kb = os.path.getsize(onnx_path) / 1024
        print(f"✓ ONNX модель сохранена: {onnx_path} ({size_kb:.1f} KB)")

        # Путь для ClickHouse
        ch_models_dir = '/var/lib/clickhouse/models'
        if os.path.isdir(ch_models_dir):
            import shutil
            shutil.copy(onnx_path, os.path.join(ch_models_dir, 'tip_classifier.onnx'))
            print(f"✓ Скопировано в {ch_models_dir}")
        else:
            print(f"⚠ Директория {ch_models_dir} не найдена.")
            print(f"  Скопируйте вручную:")
            print(f"  docker cp {onnx_path} clickhouse:/var/lib/clickhouse/models/")

    except ImportError:
        print("⚠ onnxmltools или skl2onnx не установлены.")
        print("  pip install onnxmltools skl2onnx")
        print("  Генерируем синтетический ONNX-файл для демо...")
        _generate_dummy_onnx(FEATURES)

except ImportError:
    print("⚠ LightGBM не установлен. pip install lightgbm")
    print("  Генерируем синтетическую модель для демо инференса...")
    _generate_dummy_onnx(FEATURES)


def _generate_dummy_onnx(features):
    """Создаёт минимальный валидный ONNX для демо без LightGBM."""
    try:
        import onnx
        from onnx import helper, TensorProto
        import numpy as np

        # Линейная модель: sigmoid(w·x + b)
        n = len(features)
        np.random.seed(42)
        W = np.random.randn(1, n).astype(np.float32) * 0.1
        b = np.array([0.0], dtype=np.float32)

        X_in  = helper.make_tensor_value_info('features', TensorProto.FLOAT, [None, n])
        out   = helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, [None, 2])

        W_init = helper.make_tensor('W', TensorProto.FLOAT, W.shape, W.flatten().tolist())
        b_init = helper.make_tensor('b', TensorProto.FLOAT, b.shape, b.tolist())

        gemm  = helper.make_node('Gemm', ['features','W','b'], ['logit'], transB=1)
        sig   = helper.make_node('Sigmoid', ['logit'], ['prob1'])
        graph = helper.make_graph([gemm, sig], 'tip_model', [X_in], [out], [W_init, b_init])
        model = helper.make_model(graph)

        os.makedirs('models', exist_ok=True)
        onnx.save(model, 'models/tip_classifier.onnx')
        print("✓ Синтетическая ONNX-модель сохранена в models/tip_classifier.onnx")
    except ImportError:
        print("⚠ Установите onnx: pip install onnx")


# ── Конфиг ClickHouse для модели ──────────────────────────────
config_xml = """<models>
  <model>
    <type>catboost</type>
    <name>tip_classifier</name>
    <path>/var/lib/clickhouse/models/tip_classifier.onnx</path>
  </model>
</models>"""

os.makedirs('models', exist_ok=True)
with open('models/tip_classifier.xml', 'w') as f:
    f.write(config_xml)
print(f"\n✓ Конфиг ClickHouse: models/tip_classifier.xml")
print("  Скопируйте в /etc/clickhouse-server/models/tip_classifier.xml")
print("  docker cp models/tip_classifier.xml clickhouse:/etc/clickhouse-server/models/")

print("\n✓ Готово! Далее: clickhouse-client < scripts/05_inference.sql")
