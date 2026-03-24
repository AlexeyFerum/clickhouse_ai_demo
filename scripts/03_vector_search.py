"""
03_vector_search.py
Блок 2: Векторный поиск по районам NYC
- Генерирует эмбеддинги через sentence-transformers
- Обновляет таблицу taxi_zones
- Демонстрирует brute-force vs ANN-поиск

Запуск: python scripts/03_vector_search.py
"""

import clickhouse_connect
import numpy as np
import time

# ── Подключение ────────────────────────────────────────────────
client = clickhouse_connect.get_client(
    host='localhost', port=8123,
    username='default', password=''
)

# ── Загрузка зон из БД ─────────────────────────────────────────
print("Загружаем зоны из ClickHouse...")
zones_result = client.query(
    "SELECT zone_id, zone_name, borough, zone_type FROM taxi_zones ORDER BY zone_id"
)
zones = zones_result.result_rows
print(f"✓ Загружено {len(zones)} зон")

# ── Генерация эмбеддингов ──────────────────────────────────────
print("\nГенерируем эмбеддинги (sentence-transformers all-MiniLM-L6-v2)...")

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')  # dim=384

    texts = [
        f"{z[1]}, {z[2]}, {z[3]} zone, New York City"
        for z in zones
    ]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    print(f"✓ Эмбеддинги сгенерированы: {len(embeddings)} векторов размерности {len(embeddings[0])}")

except ImportError:
    # Fallback: синтетические эмбеддинги для демо без GPU
    print("⚠ sentence-transformers не установлен — используем синтетические эмбеддинги (dim=64)")
    print("  Установите: pip install sentence-transformers")

    np.random.seed(42)
    DIM = 64

    # Синтетические эмбеддинги с реалистичной кластеризацией по боро
    borough_centers = {
        'Manhattan': np.random.randn(DIM) * 0.5,
        'Brooklyn':  np.random.randn(DIM) * 0.5,
        'Queens':    np.random.randn(DIM) * 0.5,
        'Bronx':     np.random.randn(DIM) * 0.5,
        'EWR':       np.random.randn(DIM) * 0.5,
    }
    # Аэропорты — отдельный кластер
    airport_center = np.random.randn(DIM) * 0.5

    embeddings = []
    for z in zones:
        borough = z[2]
        zone_type = z[3]
        center = borough_centers.get(borough, np.zeros(DIM))
        if zone_type == 'Airport':
            vec = airport_center + np.random.randn(DIM) * 0.2
        else:
            vec = center + np.random.randn(DIM) * 0.3
        # Нормализация
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        embeddings.append(vec.astype(np.float32).tolist())

    print(f"✓ Синтетические эмбеддинги: {len(embeddings)} векторов размерности {DIM}")

# ── Запись эмбеддингов в ClickHouse ───────────────────────────
print("\nОбновляем таблицу taxi_zones...")

# Пересоздаём с эмбеддингами (ALTER UPDATE для массивов не поддерживается)
client.command("TRUNCATE TABLE taxi_zones")

rows = [
    (z[0], z[1], z[2], z[3], emb)
    for z, emb in zip(zones, embeddings)
]
client.insert('taxi_zones', rows,
              column_names=['zone_id','zone_name','borough','zone_type','embedding'])

print(f"✓ {len(rows)} зон с эмбеддингами записано")

# ── Демо: поиск похожих районов ────────────────────────────────
print("\n" + "="*60)
print("ДЕМО: Семантический поиск по районам NYC")
print("="*60)

# Вектор запроса — эмбеддинг JFK Airport
jfk_idx = next(i for i, z in enumerate(zones) if 'JFK' in z[1])
query_vector = embeddings[jfk_idx]
query_str = str(query_vector).replace(' ', '')

# 1. Brute-force
print("\n[1] Brute-force (без индекса): районы, похожие на JFK Airport")
t0 = time.time()
result = client.query(f"""
    SELECT
        zone_name,
        borough,
        zone_type,
        round(cosineDistance(embedding, {query_str}::Array(Float32)), 4) AS distance
    FROM taxi_zones
    ORDER BY distance ASC
    LIMIT 5
""")
elapsed_bf = (time.time() - t0) * 1000

print(f"  Время: {elapsed_bf:.1f}ms")
print(f"  {'Район':<30} {'Боро':<12} {'Тип':<15} {'Дистанция':>10}")
print(f"  {'-'*30} {'-'*12} {'-'*15} {'-'*10}")
for row in result.result_rows:
    print(f"  {row[0]:<30} {row[1]:<12} {row[2]:<15} {row[3]:>10.4f}")

# 2. ANN с индексом
print("\n[2] ANN-поиск (с usearch индексом):")
client.command("SET allow_experimental_usearch_index = 1")
t0 = time.time()
result_ann = client.query(f"""
    SELECT
        zone_name,
        borough,
        zone_type,
        round(cosineDistance(embedding, {query_str}::Array(Float32)), 4) AS distance
    FROM taxi_zones
    WHERE cosineDistance(embedding, {query_str}::Array(Float32)) < 0.5
    ORDER BY distance ASC
    LIMIT 5
""")
elapsed_ann = (time.time() - t0) * 1000

print(f"  Время: {elapsed_ann:.1f}ms")
for row in result_ann.result_rows:
    print(f"  {row[0]:<30} {row[1]:<12} {row[2]:<15} {row[3]:>10.4f}")

print(f"\n  Ускорение ANN vs brute-force: {elapsed_bf/max(elapsed_ann,0.1):.1f}×")
print("  (На реальных объёмах 1M+ векторов разница 40–100×)")

# 3. Гибридный поиск: вектор + SQL-фильтр
print("\n[3] Гибридный поиск: похожие на JFK, но только Manhattan")
result_hybrid = client.query(f"""
    SELECT
        zone_name,
        borough,
        round(cosineDistance(embedding, {query_str}::Array(Float32)), 4) AS distance
    FROM taxi_zones
    WHERE borough = 'Manhattan'
      AND cosineDistance(embedding, {query_str}::Array(Float32)) < 0.7
    ORDER BY distance ASC
    LIMIT 5
""")

print(f"  {'Район':<30} {'Боро':<12} {'Дистанция':>10}")
print(f"  {'-'*30} {'-'*12} {'-'*10}")
for row in result_hybrid.result_rows:
    print(f"  {row[0]:<30} {row[1]:<12} {row[3] if len(row)>3 else row[2]:>10.4f}")

print("\n✓ Демо векторного поиска завершено")
print("Далее: python scripts/04_train_export_onnx.py")
