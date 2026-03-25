from __future__ import annotations

import argparse
import os
import sys
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

from flask import Flask, render_template, request, send_file
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"
GENERATE_PDF_PATH = Path(os.environ.get("PROJECT_GENERATE_PDF_PATH", APP_DIR / "generate_pdf.py")).resolve()
GENERATE_PDF_SPEC = spec_from_file_location("project_generate_pdf", GENERATE_PDF_PATH)
if GENERATE_PDF_SPEC is None or GENERATE_PDF_SPEC.loader is None:
    raise ImportError(f"Não foi possível carregar o módulo local: {GENERATE_PDF_PATH}")

GENERATE_PDF_MODULE = module_from_spec(GENERATE_PDF_SPEC)
sys.modules[GENERATE_PDF_SPEC.name] = GENERATE_PDF_MODULE
GENERATE_PDF_SPEC.loader.exec_module(GENERATE_PDF_MODULE)

def _fallback_format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fallback_choose_default_bairro(df: pd.DataFrame) -> str:
    bairros = sorted([value for value in df["bairro_padronizado"].dropna().astype(str).tolist() if value.strip()])
    if not bairros:
        raise ValueError("Não há bairros válidos na base analítica para gerar um relatório padrão.")
    return bairros[0]


def _build_generate_report_fallback(module):
    required = ["FilterScope", "load_data", "filter_data", "compute_indicators", "generate_charts", "build_pdf"]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "O módulo generate_pdf.py não expõe a interface necessária para o app web. "
            f"Itens ausentes: {missing}"
        )

    def _generate_report_pdf(
        bairro: str,
        quadra: str | None = None,
        bloco: str | None = None,
        input_path: str | None = None,
        output_path: str | None = None,
    ):
        if output_path is None:
            raise ValueError("output_path é obrigatório no fallback de geração de PDF.")

        df, base_path = module.load_data(input_path)
        scoped = module.filter_data(df, bairro=bairro, quadra=quadra, bloco=bloco)
        if scoped.empty:
            raise ValueError("Nenhum registro encontrado para os filtros informados.")

        filters = module.FilterScope(bairro=bairro, quadra=quadra, bloco=bloco)
        indicators = module.compute_indicators(scoped, filters)
        output_file = Path(output_path)

        with tempfile.TemporaryDirectory(prefix="study_charts_") as tmpdir:
            charts = module.generate_charts(indicators, Path(tmpdir))
            module.build_pdf(indicators, charts, output_file)

        return SimpleNamespace(pdf_path=output_file, indicators=indicators, base_path=base_path)

    return _generate_report_pdf


load_data = getattr(GENERATE_PDF_MODULE, "load_data")
generate_report_pdf = getattr(GENERATE_PDF_MODULE, "generate_report_pdf", _build_generate_report_fallback(GENERATE_PDF_MODULE))
choose_default_bairro = getattr(GENERATE_PDF_MODULE, "choose_default_bairro", _fallback_choose_default_bairro)
format_currency = getattr(GENERATE_PDF_MODULE, "format_currency", _fallback_format_currency)

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))


def _options_from_series(series: pd.Series) -> list[str]:
    return sorted([value for value in series.dropna().astype(str).unique().tolist() if value.strip()])


def _scope_preview(df: pd.DataFrame, bairro: str, quadra: str | None = None, bloco: str | None = None) -> dict:
    scoped = df[df["bairro_padronizado"].fillna("") == bairro].copy()
    if quadra:
        scoped = scoped[scoped["grupo_quadra"].fillna("") == quadra]
    if bloco:
        scoped = scoped[scoped["bloco_padronizado"].fillna("") == bloco]

    return {
        "bairro": bairro,
        "quadra": quadra,
        "bloco": bloco,
        "registros": len(scoped),
        "meses": sorted(scoped["mes_ref"].dropna().astype(str).unique().tolist()),
    }


def _get_form_state(df: pd.DataFrame, form: dict[str, str]) -> dict:
    bairros = _options_from_series(df["bairro_padronizado"])
    bairro = (form.get("bairro") or "").strip()
    if not bairro and bairros:
        bairro = bairros[0]

    scoped_bairro = df[df["bairro_padronizado"].fillna("") == bairro] if bairro else pd.DataFrame(columns=df.columns)
    quadras = _options_from_series(scoped_bairro["grupo_quadra"]) if not scoped_bairro.empty else []
    quadra = (form.get("quadra") or "").strip()
    if quadra not in quadras:
        quadra = ""

    scoped_quadra = scoped_bairro
    if quadra:
        scoped_quadra = scoped_bairro[scoped_bairro["grupo_quadra"].fillna("") == quadra]
    blocos = _options_from_series(scoped_quadra["bloco_padronizado"]) if not scoped_quadra.empty else []
    bloco = (form.get("bloco") or "").strip()
    if bloco not in blocos:
        bloco = ""

    preview = _scope_preview(df, bairro=bairro, quadra=quadra or None, bloco=bloco or None) if bairro else None

    return {
        "bairros": bairros,
        "quadras": quadras,
        "blocos": blocos,
        "bairro": bairro,
        "quadra": quadra,
        "bloco": bloco,
        "preview": preview,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    input_path = (request.form.get("input_path") or request.args.get("input") or "").strip() or None
    error = None
    info = None

    try:
        df, base_path = load_data(input_path)
    except Exception as exc:
        return render_template(
            "index.html",
            error=f"Não foi possível carregar a base analítica: {exc}",
            info=None,
            base_path=input_path or "(padrão do projeto)",
            bairros=[],
            quadras=[],
            blocos=[],
            bairro="",
            quadra="",
            bloco="",
            preview=None,
            input_path=input_path or "",
            last_month=None,
        )

    form_state = _get_form_state(df, request.form if request.method == "POST" else request.args)

    if request.method == "POST" and request.form.get("action") == "preview":
        info = "Prévia atualizada com sucesso."

    if request.method == "POST" and request.form.get("action") == "generate":
        if not form_state["bairro"]:
            error = "Selecione um bairro para gerar o relatório."
        elif form_state["preview"] and form_state["preview"]["registros"] == 0:
            error = "Nenhum registro encontrado para os filtros selecionados."
        else:
            try:
                with tempfile.NamedTemporaryFile(prefix="estudo_imobiliario_", suffix=".pdf", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                result = generate_report_pdf(
                    bairro=form_state["bairro"],
                    quadra=form_state["quadra"] or None,
                    bloco=form_state["bloco"] or None,
                    input_path=input_path,
                    output_path=str(tmp_path),
                )

                pdf_path = Path(result.pdf_path)
                if not pdf_path.exists() or pdf_path.stat().st_size == 0:
                    raise ValueError("A geração do PDF não produziu um arquivo válido.")

                download_name = pdf_path.name
                return send_file(pdf_path, as_attachment=True, download_name=download_name, mimetype="application/pdf")
            except Exception as exc:
                error = f"Erro ao gerar PDF: {exc}"

    last_month = None
    if form_state["bairro"]:
        scoped_preview = form_state["preview"]
        if scoped_preview and scoped_preview["registros"] > 0:
            scoped_df = df[df["bairro_padronizado"].fillna("") == form_state["bairro"]]
            if form_state["quadra"]:
                scoped_df = scoped_df[scoped_df["grupo_quadra"].fillna("") == form_state["quadra"]]
            if form_state["bloco"]:
                scoped_df = scoped_df[scoped_df["bloco_padronizado"].fillna("") == form_state["bloco"]]
            if not scoped_df.empty:
                ref = scoped_df[scoped_df["mes_ref_dt"] == scoped_df["mes_ref_dt"].max()]
                last_month = {
                    "mes_ref": ref["mes_ref"].iloc[0],
                    "valor_m2_geral": format_currency(ref["valor_m2_calc"].mean()),
                    "amostra": len(ref),
                }

    return render_template(
        "index.html",
        error=error,
        info=info,
        base_path=str(base_path),
        input_path=input_path or "",
        last_month=last_month,
        **form_state,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface web local (Flask) para gerar estudo imobiliário em PDF.")
    parser.add_argument("--host", default="127.0.0.1", help="Host do servidor Flask local.")
    parser.add_argument("--port", type=int, default=5000, help="Porta do servidor Flask local.")
    parser.add_argument("--debug", action="store_true", help="Ativa modo debug do Flask.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
