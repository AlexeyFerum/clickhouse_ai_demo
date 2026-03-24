# ClickHouse AI — Demo Project
## Вебинар «РазвИИтие на основе ClickHouse»

Проект содержит все скрипты для воспроизведения демо из трёх блоков вебинара.

### Структура
```
demo_project/
├── scripts/
│   ├── 00_setup.sql            # Создание таблиц
│   ├── 01_seed_data.py         # Наполнение тестовыми данными
│   ├── 02_feature_engineering.sql  # Блок 1: фичи
│   ├── 03_vector_search.py     # Блок 2: эмбеддинги и поиск
│   ├── 04_train_export_onnx.py # Блок 3: обучение и экспорт
│   └── 05_inference.sql        # Блок 3: инференс в ClickHouse
├── sql/
│   └── queries_reference.sql   # Все SQL-запросы вебинара
└── README.md
```

### Требования
```
clickhouse-connect>=0.7
sentence-transformers>=2.2
lightgbm>=4.0
onnxmltools>=1.12
skl2onnx>=1.16
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
```

### Быстрый старт
```bash
# 1. Запустить ClickHouse
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
curl -fsSL 'https://packages.clickhouse.com/rpm/lts/repodata/repomd.xml.key' | sudo gpg --dearmor -o /usr/share/keyrings/clickhouse-keyring.gpg
ARCH=$(dpkg --print-architecture)
echo "deb [signed-by=/usr/share/keyrings/clickhouse-keyring.gpg arch=${ARCH}] https://packages.clickhouse.com/deb stable main" | sudo tee /etc/apt/sources.list.d/clickhouse.list
sudo apt-get update

sudo apt-get install -y clickhouse-server clickhouse-client
sudo service clickhouse-server start


# 2. Установить зависимости
sudo apt install python3-pip
sudo apt install python3-venv

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt


# 3. Создать таблицы
clickhouse-client < scripts/00_setup.sql


# 4. Наполнить данными (~5 мин)
python scripts/01_seed_data.py


# 5. Запустить демо по блокам
clickhouse-client < scripts/02_feature_engineering.sql
python scripts/03_vector_search.py
python scripts/04_train_export_onnx.py


# Предварительно: убедитесь, что модель зарегистрирована
sudo mkdir -p /var/lib/clickhouse/models
sudo mkdir -p /etc/clickhouse-server/models

sudo cp models/tip_classifier.onnx /var/lib/clickhouse/models/
sudo cp models/tip_classifier.xml  /etc/clickhouse-server/models/

sudo systemctl restart clickhouse-server

clickhouse-client < scripts/05_inference.sql
```
