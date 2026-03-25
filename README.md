# ClickHouse AI — Demo Project
## Вебинар «РазвИИтие на основе ClickHouse»

Проект содержит все скрипты для воспроизведения демо из трёх блоков вебинара.

### Структура
```
demo_project/
├── scripts/
│   ├── 00_setup.sql                # Создание таблиц
│   ├── 01_seed_data.py             # Наполнение тестовыми данными
│   ├── 02_feature_engineering.sql  # Блок 1: фичи
│   ├── 03_vector_search.py         # Блок 2: эмбеддинги и поиск
│   ├── 04_train_export_onnx.py     # Блок 3: обучение CatBoost и деплой
│   └── 05_inference.sql            # Блок 3: инференс в ClickHouse
└── README.md
```

### Требования
```
clickhouse-connect>=0.7
sentence-transformers>=2.2
catboost>=1.2
lightgbm>=4.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
```

---

## Установка ClickHouse (Ubuntu/Debian)

```bash
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
ARCH=$(dpkg --print-architecture)
echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] \
  https://packages.clickhouse.com/deb stable main" \
  | sudo tee /etc/apt/sources.list.d/clickhouse.list
sudo apt-get update
sudo apt-get install -y clickhouse-server clickhouse-client
sudo service clickhouse-server start
```

### ⚠ Дополнительный пакет для инференса моделей

`catboostEvaluate()` требует отдельный bridge-процесс — он не входит
в базовый пакет и должен быть установлен явно:

```bash
sudo apt-get install -y clickhouse-library-bridge
```

Без этого пакета при вызове `catboostEvaluate()` будет ошибка:
`CHILD_WAS_NOT_EXITED_NORMALLY` (код 302).

### ⚠ Библиотека libcatboostmodel.so

Скачайте и положите в директорию ClickHouse:

```bash
wget https://github.com/catboost/catboost/releases/download/v1.2.7/libcatboostmodel.so \
  -O /tmp/libcatboostmodel.so
sudo cp /tmp/libcatboostmodel.so /var/lib/clickhouse/
sudo chown clickhouse:clickhouse /var/lib/clickhouse/libcatboostmodel.so
```

### ⚠ Конфигурация ClickHouse для CatBoost

Создайте файл `/etc/clickhouse-server/config.d/catboost_config.xml`:

```xml
<clickhouse>
    <catboost_lib_path>/var/lib/clickhouse/libcatboostmodel.so</catboost_lib_path>
    <models_config>/etc/clickhouse-server/models/*.xml</models_config>
</clickhouse>
```

Оба параметра обязательны в одном файле:
- `catboost_lib_path` — путь к `.so` библиотеке
- `models_config` — маска пути к XML-конфигам моделей

После создания файла:
```bash
sudo systemctl restart clickhouse-server
```

### ⚠ ANN-индекс для векторного поиска

В ClickHouse 26.x тип индекса изменился. Используйте `vector_similarity`,
а не `usearch`:

```sql
-- Правильно (26.x+):
INDEX ann_idx embedding
    TYPE vector_similarity('hnsw', 'cosineDistance')
    GRANULARITY 1

-- Включить экспериментальный индекс:
SET allow_experimental_vector_similarity_index = 1;
```

---

## Быстрый старт

```bash
# 1. Установить зависимости Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Создать таблицы
clickhouse-client < scripts/00_setup.sql

# 3. Наполнить данными (~5 мин)
python scripts/01_seed_data.py

# 4. Блок 1: feature engineering
clickhouse-client < scripts/02_feature_engineering.sql

# 5. Блок 2: векторный поиск
python scripts/03_vector_search.py

# 6. Блок 3: обучение и деплой модели
python scripts/04_train_export_onnx.py
# Скрипт сам скопирует модель и конфиги, перезапустит сервер

# 7. Блок 3: инференс
clickhouse-client < scripts/05_inference.sql
```

---

## Проверка установки перед запуском

```bash
# CatBoost bridge установлен?
which clickhouse-library-bridge

# Библиотека на месте?
sudo ls -la /var/lib/clickhouse/libcatboostmodel.so

# Конфиг применился?
grep -r 'catboost_lib_path' /etc/clickhouse-server/

# Быстрый тест функции (ожидать ошибку про файл, не про функцию):
clickhouse-client --query \
  "SELECT catboostEvaluate('/tmp/test', 1.0, 1.0, 0.0, 2.5, 1.0, 10.0)"
# Правильный ответ: DB::Exception: Can't load model /tmp/test: file doesn't exist
# Неправильный:    DB::Exception: Child process was exited with return code 88
```
