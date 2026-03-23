import pandas as pd

from config import BASE_ANALITICA_PATH, MOVING_AVG_WINDOW, PIPELINE_LOG_PATH
from logger_store import append_pipeline_log


def load_base():
    if not BASE_ANALITICA_PATH.exists():
        raise FileNotFoundError("Rode primeiro o prepare_base.py para gerar a base analítica.")
    return pd.read_csv(BASE_ANALITICA_PATH, encoding="utf-8-sig")


def filter_scope(df, bairro=None, quadra=None, bloco=None):
    work = df.copy()
    if bairro:
        work = work[work["bairro_padronizado"].fillna("").str.lower() == bairro.strip().lower()]
    if quadra and "grupo_quadra" in work.columns:
        work = work[work["grupo_quadra"].fillna("").str.lower() == quadra.strip().lower()]
    if bloco and "bloco_padronizado" in work.columns:
        work = work[work["bloco_padronizado"].fillna("").str.lower() == bloco.strip().lower()]
    return work


def monthly_series(df, group_cols):
    if df.empty:
        return pd.DataFrame()

    base = (
        df.dropna(subset=["mes_ref", "valor_m2_calc"])
          .groupby(["mes_ref", *group_cols], dropna=False, as_index=False)
          .agg(valor_m2_medio=("valor_m2_calc", "mean"), qtd=("valor_m2_calc", "size"))
          .sort_values("mes_ref")
    )

    sort_cols = [*group_cols, "mes_ref"]
    base = base.sort_values(sort_cols)
    base["mm3_valor_m2"] = (
        base.groupby(group_cols)["valor_m2_medio"]
            .transform(lambda s: s.rolling(MOVING_AVG_WINDOW, min_periods=1).mean())
    )
    return base


def summary_last_month(df):
    if df.empty:
        return {}
    valid = df.dropna(subset=["mes_ref", "valor_m2_calc"]).copy()
    if valid.empty:
        return {}
    ultimo_mes = sorted(valid["mes_ref"].dropna().unique())[-1]
    ref = valid[valid["mes_ref"] == ultimo_mes].copy()

    return {
        "mes_ref": ultimo_mes,
        "valor_m2_geral": ref["valor_m2_calc"].mean(),
        "valor_m2_com_vaga": ref.loc[ref["tem_vaga"] == "Com Vaga", "valor_m2_calc"].mean(),
        "valor_m2_sem_vaga": ref.loc[ref["tem_vaga"] == "Sem Vaga", "valor_m2_calc"].mean(),
        "amostra": len(ref),
    }


def build_study_payload(bairro=None, quadra=None, bloco=None):
    df = load_base()
    scoped = filter_scope(df, bairro=bairro, quadra=quadra, bloco=bloco)

    payload = {
        "scope": {
            "bairro": bairro,
            "quadra": quadra,
            "bloco": bloco,
            "linhas_filtradas": len(scoped),
        },
        "headline": summary_last_month(scoped),
        "series_vaga": monthly_series(scoped, ["tem_vaga"]),
        "series_quartos": monthly_series(scoped.dropna(subset=["quartos_num"]), ["quartos_num", "tem_vaga"]),
        "series_metragem": monthly_series(scoped.dropna(subset=["faixa_metragem"]), ["faixa_metragem", "tem_vaga"]),
        "series_quadra": monthly_series(scoped.dropna(subset=["grupo_quadra"]), ["grupo_quadra", "tem_vaga"]) if not quadra else pd.DataFrame(),
    }

    append_pipeline_log(
        PIPELINE_LOG_PATH,
        {
            "step": "build_study_payload",
            "bairro": bairro,
            "quadra": quadra,
            "bloco": bloco,
            "linhas_filtradas": len(scoped),
            "headline_mes_ref": payload["headline"].get("mes_ref") if payload["headline"] else None,
        },
    )
    return payload


if __name__ == "__main__":
    payload = build_study_payload(bairro="Asa Norte")
    print(payload["scope"])
    print(payload["headline"])
