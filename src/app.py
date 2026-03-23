from __future__ import annotations

from importlib.util import find_spec
from io import BytesIO

import pandas as pd

if find_spec("streamlit") is None:
    raise ModuleNotFoundError(
        "Streamlit não está instalado no ambiente. Rode `python -m pip install -r requirements.txt` "
        "e execute a interface com `streamlit run src/app.py`."
    )

import streamlit as st

from generate_pdf import generate_report_pdf, load_data

st.set_page_config(page_title="Estudo Imobiliário", layout="wide")
st.title("Estudo Imobiliário")
st.caption("Selecione o recorte do estudo para gerar o PDF analítico completo.")

st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    .metric-box {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 1rem;
        background: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    df, base_path = load_data()
except Exception as exc:
    st.warning(f"Não foi possível carregar a base analítica: {exc}")
    st.stop()

bairros = sorted([x for x in df["bairro_padronizado"].dropna().unique().tolist() if str(x).strip()])

col1, col2, col3 = st.columns(3)
with col1:
    bairro = st.selectbox("Bairro", [""] + bairros, index=1 if len(bairros) == 1 else 0)

if bairro:
    scoped_bairro = df[df["bairro_padronizado"].fillna("") == bairro]
    quadras_disponiveis = sorted(
        [x for x in scoped_bairro["grupo_quadra"].dropna().unique().tolist() if str(x).strip()]
    )
else:
    scoped_bairro = pd.DataFrame(columns=df.columns)
    quadras_disponiveis = []

with col2:
    quadra = st.selectbox("Quadra", [""] + quadras_disponiveis, disabled=not bairro)

if quadra:
    scoped_quadra = scoped_bairro[scoped_bairro["grupo_quadra"].fillna("") == quadra]
    blocos_disponiveis = sorted(
        [x for x in scoped_quadra["bloco_padronizado"].dropna().unique().tolist() if str(x).strip()]
    )
else:
    scoped_quadra = scoped_bairro
    blocos_disponiveis = []

with col3:
    bloco = st.selectbox("Bloco", [""] + blocos_disponiveis, disabled=not quadra)

preview_scope = scoped_quadra.copy()
if bloco:
    preview_scope = preview_scope[preview_scope["bloco_padronizado"].fillna("") == bloco]

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Base carregada", f"{len(df):,}".replace(",", "."))
metric_2.metric("Registros no recorte", f"{len(preview_scope):,}".replace(",", "."))
metric_3.metric("Meses disponíveis", int(preview_scope["mes_ref"].nunique()) if not preview_scope.empty else 0)

st.info(f"Base utilizada: {base_path}")

if bairro:
    with st.expander("Prévia do recorte analítico", expanded=True):
        st.write(
            {
                "bairro": bairro,
                "quadra": quadra or None,
                "bloco": bloco or None,
                "registros": len(preview_scope),
                "meses": sorted(preview_scope["mes_ref"].dropna().astype(str).unique().tolist()),
            }
        )
        st.dataframe(preview_scope.head(50), use_container_width=True)
else:
    st.warning("Selecione ao menos um bairro para habilitar a geração do estudo.")

button_col1, button_col2 = st.columns([1, 1])
with button_col1:
    gerar_pdf = st.button("Gerar PDF", type="primary", disabled=not bairro)
with button_col2:
    limpar = st.button("Limpar filtros")

if limpar:
    st.rerun()

if gerar_pdf and bairro:
    with st.spinner("Gerando relatório em PDF..."):
        result = generate_report_pdf(
            bairro=bairro,
            quadra=quadra or None,
            bloco=bloco or None,
        )

    pdf_bytes = BytesIO(result.pdf_path.read_bytes())
    st.success(f"PDF gerado com sucesso em: {result.pdf_path}")
    st.download_button(
        label="Baixar PDF",
        data=pdf_bytes,
        file_name=result.pdf_path.name,
        mime="application/pdf",
    )

    st.subheader("Resumo do último mês")
    st.json(result.indicators.overview_last_month)
