from pathlib import Path

BASE_INCREMENTAL_PATH = Path("/Users/macbook/Desktop/Corretagem_2026/Coletas/base_incremental.csv")
BASE_ANALITICA_PATH = Path("/Users/macbook/Desktop/Corretagem_2026/Coletas/data/base_analitica.csv")
PIPELINE_LOG_PATH = Path("/Users/macbook/Desktop/Corretagem_2026/Coletas/logs/pipeline_history.jsonl")

MIN_SAMPLE_CLUSTER = 30
MIN_SAMPLE_SERIE = 12
MOVING_AVG_WINDOW = 3

METRAGEM_BINS = [0, 75, 90, 130, 160, 200, 9999]
METRAGEM_LABELS = ["Até 75", "75-90", "90-130", "130-160", "160-200", ">200"]

OUTLIER_STRATEGY_SMALL = "iqr"
OUTLIER_STRATEGY_MEDIUM = "iqr"
OUTLIER_STRATEGY_LARGE = "quantile_clip"

SMALL_BASE_MAX = 500
MEDIUM_BASE_MAX = 5000
