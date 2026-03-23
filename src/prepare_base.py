import re
import numpy as np
import pandas as pd

from config import (
    BASE_INCREMENTAL_PATH,
    BASE_ANALITICA_PATH,
    PIPELINE_LOG_PATH,
    METRAGEM_BINS,
    METRAGEM_LABELS,
    SMALL_BASE_MAX,
    MEDIUM_BASE_MAX,
    OUTLIER_STRATEGY_SMALL,
    OUTLIER_STRATEGY_MEDIUM,
    OUTLIER_STRATEGY_LARGE,
)
from logger_store import append_pipeline_log


def to_float_br(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    s = re.sub(r"[^0-9,.\-]", "", s)
    if s.count(",") > 0 and s.count(".") > 0:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(",") > 0:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def normalize_text(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    return re.sub(r"\s+", " ", s) if s else None


def extract_group_quadra(value):
    if pd.isna(value):
        return None
    s = str(value).upper().strip()
    m = re.search(r"(\d{3})", s)
    if not m:
        return None
    centena = m.group(1)[0] + "00"
    prefix = re.match(r"([A-Z]+)", s.replace(" ", ""))
    pref = prefix.group(1) if prefix else ""
    if pref.startswith("SQN"):
        return f"SQN {centena}"
    if pref.startswith("CLN"):
        return f"CLN {centena}"
    return f"{pref} {centena}".strip()


def extract_bloco(value):
    if pd.isna(value):
        return None
    s = str(value).upper()
    m = re.search(r"\bBLOCO\s+([A-Z0-9]+)\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\bBL\s*([A-Z0-9]+)\b", s)
    if m:
        return m.group(1)
    return None


def choose_outlier_strategy(n_rows: int) -> str:
    if n_rows <= SMALL_BASE_MAX:
        return OUTLIER_STRATEGY_SMALL
    if n_rows <= MEDIUM_BASE_MAX:
        return OUTLIER_STRATEGY_MEDIUM
    return OUTLIER_STRATEGY_LARGE


def apply_outlier_control(df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
    work = df.copy()
    if col not in work.columns:
        return work

    if strategy == "iqr":
        q1 = work[col].quantile(0.25)
        q3 = work[col].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return work
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        return work[(work[col].isna()) | ((work[col] >= low) & (work[col] <= high))]

    if strategy == "quantile_clip":
        low = work[col].quantile(0.01)
        high = work[col].quantile(0.99)
        work[col] = work[col].clip(lower=low, upper=high)
        return work

    return work


def prepare_base():
    if not BASE_INCREMENTAL_PATH.exists():
        raise FileNotFoundError(f"Base incremental não encontrada: {BASE_INCREMENTAL_PATH}")

    df = pd.read_csv(BASE_INCREMENTAL_PATH, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower() for c in df.columns]

    raw_rows = len(df)

    for col in ["bairro", "quadra", "descricao", "arquivo_origem", "portal", "cidade", "tipo"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)

    for col in ["preco", "valor_m2", "area_util", "quartos", "vagas", "latitude", "longitude"]:
        if col in df.columns:
            df[col + "_num"] = df[col].apply(to_float_br)

    if "data_coleta" in df.columns:
        df["data_coleta_dt"] = pd.to_datetime(df["data_coleta"], errors="coerce", dayfirst=True)
    elif "data_processamento" in df.columns:
        df["data_coleta_dt"] = pd.to_datetime(df["data_processamento"], errors="coerce")
    else:
        df["data_coleta_dt"] = pd.NaT

    df["mes_ref"] = df["data_coleta_dt"].dt.to_period("M").astype("string")

    if "bairro" in df.columns:
        df["bairro_padronizado"] = df["bairro"].str.title()
    else:
        df["bairro_padronizado"] = None

    if "quadra" in df.columns:
        df["grupo_quadra"] = df["quadra"].apply(extract_group_quadra)
        df["bloco_padronizado"] = df["quadra"].apply(extract_bloco)
    else:
        df["grupo_quadra"] = None
        df["bloco_padronizado"] = None

    if "vagas_num" in df.columns:
        df["tem_vaga"] = np.where(df["vagas_num"].fillna(0) > 0, "Com Vaga", "Sem Vaga")
    else:
        df["tem_vaga"] = "Sem Vaga"

    if "area_util_num" in df.columns:
        df["faixa_metragem"] = pd.cut(
            df["area_util_num"],
            bins=METRAGEM_BINS,
            labels=METRAGEM_LABELS,
            include_lowest=True,
            right=False,
        )
    else:
        df["faixa_metragem"] = None

    if "preco_num" in df.columns and "area_util_num" in df.columns:
        df["valor_m2_calc"] = df["preco_num"] / df["area_util_num"].replace(0, np.nan)
    elif "valor_m2_num" in df.columns:
        df["valor_m2_calc"] = df["valor_m2_num"]
    else:
        df["valor_m2_calc"] = np.nan

    strategy = choose_outlier_strategy(len(df))
    before_outlier = len(df)
    df = apply_outlier_control(df, "valor_m2_calc", strategy)
    after_outlier = len(df)

    if "link" in df.columns:
        df = df.sort_values(by=["data_coleta_dt"], ascending=True).drop_duplicates(subset=["link"], keep="last")
    elif "codigo" in df.columns:
        df = df.sort_values(by=["data_coleta_dt"], ascending=True).drop_duplicates(subset=["codigo"], keep="last")

    BASE_ANALITICA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASE_ANALITICA_PATH, index=False, encoding="utf-8-sig")

    append_pipeline_log(
        PIPELINE_LOG_PATH,
        {
            "step": "prepare_base",
            "source_rows": raw_rows,
            "rows_after_outlier": after_outlier,
            "rows_removed_outlier": before_outlier - after_outlier,
            "rows_final": len(df),
            "outlier_strategy": strategy,
            "output_path": str(BASE_ANALITICA_PATH),
        },
    )

    print(f"Base analítica salva em: {BASE_ANALITICA_PATH}")
    print(f"Linhas finais: {len(df)}")
    print(f"Estratégia de outlier usada: {strategy}")


if __name__ == "__main__":
    prepare_base()
