import pandas as pd
import streamlit as st

from config import BASE_ANALITICA_PATH
from generate_report import build_study_payload

st.set_page_config(page_title="Estudo Imobiliário", layout="wide")
st.title("Estudo Imobiliário — v1")

if not BASE_ANALITICA_PATH.exists():
    st.warning("A base analítica ainda não existe. Rode antes: python src/prepare_base.py")
    st.stop()

df = pd.read_csv(BASE_ANALITICA_PATH, encoding="utf-8-sig")

bairros = sorted([x for x in df["bairro_padronizado"].dropna().unique().tolist() if str(x).strip()])
bairro = st.selectbox("Bairro", [""] + bairros)

if bairro:
    quadras_disponiveis = sorted(
        [x for x in df.loc[df["bairro_padronizado"] == bairro, "grupo_quadra"].dropna().unique().tolist() if str(x).strip()]
    )
else:
    quadras_disponiveis = []

quadra = st.selectbox("Quadra", [""] + quadras_disponiveis)

if quadra:
    blocos_disponiveis = sorted(
        [x for x in df.loc[df["grupo_quadra"] == quadra, "bloco_padronizado"].dropna().unique().tolist() if str(x).strip()]
    )
else:
    blocos_disponiveis = []

bloco = st.selectbox("Bloco", [""] + blocos_disponiveis)

if st.button("Gerar estudo-base"):
    payload = build_study_payload(
        bairro=bairro or None,
        quadra=quadra or None,
        bloco=bloco or None,
    )

    st.subheader("Escopo")
    st.json(payload["scope"])

    st.subheader("Resumo do último mês")
    st.json(payload["headline"])

    st.subheader("Série por vaga")
    st.dataframe(payload["series_vaga"], use_container_width=True)

    st.subheader("Série por quartos")
    st.dataframe(payload["series_quartos"], use_container_width=True)

    st.subheader("Série por metragem")
    st.dataframe(payload["series_metragem"], use_container_width=True)

    if quadra:
        st.info("Como a quadra foi selecionada, a divisão por quadra foi ocultada.")
    else:
        st.subheader("Série por quadra")
        st.dataframe(payload["series_quadra"], use_container_width=True)
