from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from config import BASE_ANALITICA_PATH, MOVING_AVG_WINDOW, PIPELINE_LOG_PATH
from logger_store import append_pipeline_log

MIN_RECORDS_PER_MONTH = 5
REQUIRED_COLUMNS = [
    "bairro_padronizado",
    "grupo_quadra",
    "bloco_padronizado",
    "mes_ref",
    "valor_m2_calc",
    "tem_vaga",
    "quartos_num",
    "faixa_metragem",
]


@dataclass(frozen=True)
class FilterScope:
    bairro: str
    quadra: str | None = None
    bloco: str | None = None

    def subtitle(self) -> str:
        parts = [f"Bairro: {self.bairro}"]
        if self.quadra:
            parts.append(f"Quadra: {self.quadra}")
        if self.bloco:
            parts.append(f"Bloco: {self.bloco}")
        return " | ".join(parts)


@dataclass(frozen=True)
class ChartSpec:
    title: str
    image_path: Path
    summary_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class Indicators:
    filters: FilterScope
    total_records: int
    last_month: str | None
    overview_last_month: dict[str, float | str | None]
    overview_series: pd.DataFrame
    bedrooms_series: pd.DataFrame
    area_series: pd.DataFrame
    quadra_series: pd.DataFrame


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _candidate_paths(explicit_path: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))

    candidates.extend(
        [
            BASE_ANALITICA_PATH,
            _repo_root() / "base_analitica.csv",
            _repo_root() / "data" / "base_analitica.csv",
        ]
    )

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def resolve_base_path(explicit_path: str | None = None) -> Path:
    for candidate in _candidate_paths(explicit_path):
        if candidate.exists():
            return candidate
    searched = "\n - ".join(str(path) for path in _candidate_paths(explicit_path))
    raise FileNotFoundError(
        "Base analítica não encontrada. Caminhos verificados:\n"
        f" - {searched}"
    )


def resolve_log_path() -> Path:
    local_candidates = [
        _repo_root() / "pipeline_history.jsonl",
        _repo_root() / "logs" / "pipeline_history.jsonl",
    ]

    if PIPELINE_LOG_PATH.exists() or PIPELINE_LOG_PATH.parent.exists():
        return PIPELINE_LOG_PATH

    for candidate in local_candidates:
        if candidate.exists() or candidate.parent.exists():
            return candidate

    return local_candidates[0]


def format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def load_data(input_path: str | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the analytical base and normalize key columns used in the report."""
    base_path = resolve_base_path(input_path)
    df = pd.read_csv(base_path, encoding="utf-8-sig")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes na base analítica: {missing}")

    work = df.copy()
    work["valor_m2_calc"] = pd.to_numeric(work["valor_m2_calc"], errors="coerce")
    work["quartos_num"] = pd.to_numeric(work["quartos_num"], errors="coerce")
    work["mes_ref"] = work["mes_ref"].astype("string")
    work = work.dropna(subset=["mes_ref", "valor_m2_calc"])
    work = work[work["valor_m2_calc"] > 0].copy()
    work["mes_ref_dt"] = pd.to_datetime(work["mes_ref"].astype(str) + "-01", errors="coerce")
    work = work.dropna(subset=["mes_ref_dt"]).sort_values("mes_ref_dt")
    return work, base_path


def filter_data(
    df: pd.DataFrame,
    bairro: str,
    quadra: str | None = None,
    bloco: str | None = None,
) -> pd.DataFrame:
    """Apply the report scope filters; bairro is mandatory for the study."""
    if not bairro or not bairro.strip():
        raise ValueError("O filtro de bairro é obrigatório para gerar o estudo imobiliário.")

    work = df.copy()
    work = work[
        work["bairro_padronizado"].fillna("").str.casefold() == bairro.strip().casefold()
    ]

    if quadra:
        work = work[
            work["grupo_quadra"].fillna("").str.casefold() == quadra.strip().casefold()
        ]

    if bloco:
        work = work[
            work["bloco_padronizado"].fillna("").str.casefold() == bloco.strip().casefold()
        ]

    return work.sort_values("mes_ref_dt").reset_index(drop=True)


def _aggregate_series(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str = "valor_m2_calc",
    min_records: int = MIN_RECORDS_PER_MONTH,
) -> pd.DataFrame:
    """Aggregate the series by month and group, keeping only robust monthly samples."""
    if df.empty:
        columns = ["mes_ref", "mes_ref_dt", *group_cols, "valor_m2_medio", "qtd", "mm3_valor_m2"]
        return pd.DataFrame(columns=columns)

    base = (
        df.dropna(subset=["mes_ref", "mes_ref_dt", value_col, *group_cols])
        .groupby(["mes_ref", "mes_ref_dt", *group_cols], as_index=False, dropna=False)
        .agg(valor_m2_medio=(value_col, "mean"), qtd=(value_col, "size"))
    )

    if base.empty:
        return base

    base = base[base["qtd"] >= min_records].copy()
    if base.empty:
        base["mm3_valor_m2"] = pd.Series(dtype=float)
        return base

    sort_cols = [*group_cols, "mes_ref_dt"]
    base = base.sort_values(sort_cols)
    base["mm3_valor_m2"] = (
        base.groupby(group_cols, dropna=False)["valor_m2_medio"]
        .transform(lambda series: series.rolling(MOVING_AVG_WINDOW, min_periods=1).mean())
    )
    return base.reset_index(drop=True)


def _compute_overview_last_month(df: pd.DataFrame) -> dict[str, float | str | None]:
    if df.empty:
        return {
            "mes_ref": None,
            "valor_m2_geral": None,
            "valor_m2_com_vaga": None,
            "valor_m2_sem_vaga": None,
            "amostra": 0,
        }

    last_month = df["mes_ref_dt"].max()
    ref = df[df["mes_ref_dt"] == last_month].copy()
    return {
        "mes_ref": last_month.strftime("%Y-%m"),
        "valor_m2_geral": ref["valor_m2_calc"].mean(),
        "valor_m2_com_vaga": ref.loc[ref["tem_vaga"] == "Com Vaga", "valor_m2_calc"].mean(),
        "valor_m2_sem_vaga": ref.loc[ref["tem_vaga"] == "Sem Vaga", "valor_m2_calc"].mean(),
        "amostra": int(len(ref)),
    }


def compute_indicators(df: pd.DataFrame, filters: FilterScope) -> Indicators:
    """Compute all analytical tables used by the PDF pages."""
    overview_by_vaga = _aggregate_series(df, ["tem_vaga"]).rename(columns={"tem_vaga": "serie"})
    overall = _aggregate_series(df.assign(serie="Geral"), ["serie"])
    overview_series = pd.concat([overall, overview_by_vaga], ignore_index=True, sort=False)

    bedrooms_series = _aggregate_series(df, ["quartos_num", "tem_vaga"])
    area_series = _aggregate_series(df, ["faixa_metragem"])
    quadra_series = (
        _aggregate_series(df, ["grupo_quadra"])
        if not filters.quadra
        else pd.DataFrame(columns=["mes_ref", "mes_ref_dt", "grupo_quadra", "valor_m2_medio", "qtd", "mm3_valor_m2"])
    )

    return Indicators(
        filters=filters,
        total_records=int(len(df)),
        last_month=df["mes_ref"].iloc[-1] if not df.empty else None,
        overview_last_month=_compute_overview_last_month(df),
        overview_series=overview_series,
        bedrooms_series=bedrooms_series,
        area_series=area_series,
        quadra_series=quadra_series,
    )


def _setup_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#bbbbbb")
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Mês")
    ax.set_ylabel("Valor do m² (R$)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def _plot_grouped_series(
    series_df: pd.DataFrame,
    group_cols: list[str],
    title: str,
    output_path: Path,
    line_col: str = "valor_m2_medio",
    moving_col: str = "mm3_valor_m2",
    max_series: int | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor="white")
    _setup_axes(ax, title)

    if series_df.empty:
        ax.text(0.5, 0.5, "Sem dados suficientes para esta análise.", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path

    plot_df = series_df.sort_values([*group_cols, "mes_ref_dt"]).copy()
    if max_series is not None:
        ranking = plot_df.groupby(group_cols, dropna=False)["qtd"].sum().sort_values(ascending=False)
        keep = set(ranking.head(max_series).index.tolist())
        if len(group_cols) == 1:
            plot_df = plot_df[plot_df[group_cols[0]].isin(keep)]
        else:
            keys = plot_df[group_cols].apply(tuple, axis=1)
            plot_df = plot_df[keys.isin(keep)]

    for group_key, group_df in plot_df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label = " | ".join(str(value) for value in group_key)
        ax.plot(group_df["mes_ref_dt"], group_df[line_col], linewidth=2.0, alpha=0.35, label=f"{label} · média")
        ax.plot(group_df["mes_ref_dt"], group_df[moving_col], linewidth=2.6, alpha=0.95, label=f"{label} · MM3")

    ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def generate_charts(indicators: Indicators, charts_dir: Path) -> list[ChartSpec]:
    """Generate all images that will be embedded into the final PDF."""
    charts_dir.mkdir(parents=True, exist_ok=True)

    overview_lines = (
        f"Último mês ({indicators.overview_last_month['mes_ref']}): {format_currency(indicators.overview_last_month['valor_m2_geral'])} /m²",
        f"Com vaga: {format_currency(indicators.overview_last_month['valor_m2_com_vaga'])} · Sem vaga: {format_currency(indicators.overview_last_month['valor_m2_sem_vaga'])}",
        f"Registros no último mês: {indicators.overview_last_month['amostra']}",
    )

    charts = [
        ChartSpec(
            title="Página 1 — Visão geral do valor do m²",
            image_path=_plot_grouped_series(
                indicators.overview_series,
                ["serie"] if "serie" in indicators.overview_series.columns else ["tem_vaga"],
                "Série mensal do valor do m² — geral vs. vaga",
                charts_dir / "page_1_overview.png",
            ),
            summary_lines=overview_lines,
        ),
        ChartSpec(
            title="Página 2 — Evolução por número de quartos",
            image_path=_plot_grouped_series(
                indicators.bedrooms_series.assign(
                    quartos_num=indicators.bedrooms_series["quartos_num"].map(
                        lambda value: f"{int(value)} quarto(s)" if pd.notna(value) else "n/d"
                    )
                ),
                ["quartos_num", "tem_vaga"],
                "Valor do m² por quartos e vaga",
                charts_dir / "page_2_quartos.png",
            ),
        ),
        ChartSpec(
            title="Página 3 — Evolução por faixa de metragem",
            image_path=_plot_grouped_series(
                indicators.area_series,
                ["faixa_metragem"],
                "Valor do m² por faixa de metragem",
                charts_dir / "page_3_metragem.png",
            ),
        ),
    ]

    if not indicators.filters.quadra:
        charts.append(
            ChartSpec(
                title="Página 4 — Evolução por grupo de quadra",
                image_path=_plot_grouped_series(
                    indicators.quadra_series,
                    ["grupo_quadra"],
                    "Valor do m² por grupo de quadra",
                    charts_dir / "page_4_quadra.png",
                    max_series=8,
                ),
            )
        )

    return charts


def _draw_page(
    pdf: canvas.Canvas,
    chart: ChartSpec,
    subtitle: str,
    image_size: tuple[float, float] = (470, 280),
) -> None:
    page_width, page_height = A4
    margin = 50

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(margin, page_height - 50, chart.title)

    pdf.setFont("Helvetica", 10)
    pdf.setFillColorRGB(0.35, 0.35, 0.35)
    pdf.drawString(margin, page_height - 68, subtitle)
    pdf.setFillColorRGB(0, 0, 0)

    text_y = page_height - 95
    if chart.summary_lines:
        text = pdf.beginText(margin, text_y)
        text.setFont("Helvetica", 10)
        for line in chart.summary_lines:
            text.textLine(line)
        pdf.drawText(text)
        text_y -= 36 + (len(chart.summary_lines) - 1) * 12

    image_reader = ImageReader(str(chart.image_path))
    image_width, image_height = image_size
    image_x = (page_width - image_width) / 2
    image_y = max(120, text_y - image_height - 10)
    pdf.drawImage(image_reader, image_x, image_y, width=image_width, height=image_height, preserveAspectRatio=True)

    pdf.setFont("Helvetica-Oblique", 9)
    pdf.setFillColorRGB(0.4, 0.4, 0.4)
    pdf.drawString(margin, 40, "Fonte: base analítica de anúncios tratada pelo pipeline do projeto.")
    pdf.setFillColorRGB(0, 0, 0)
    pdf.showPage()


def build_pdf(indicators: Indicators, charts: Iterable[ChartSpec], output_path: Path) -> Path:
    """Build the final multi-page PDF report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=A4)

    for chart in charts:
        _draw_page(pdf, chart, indicators.filters.subtitle())

    pdf.save()
    return output_path


def choose_default_bairro(df: pd.DataFrame) -> str:
    bairros = sorted({value for value in df["bairro_padronizado"].dropna().tolist() if str(value).strip()})
    if not bairros:
        raise ValueError("Não há bairros válidos na base analítica para gerar um relatório padrão.")
    return bairros[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera um estudo imobiliário em PDF a partir da base analítica.")
    parser.add_argument("--input", help="Caminho opcional para a base analítica CSV.")
    parser.add_argument("--output", help="Caminho opcional para o PDF de saída.")
    parser.add_argument("--bairro", help="Bairro do estudo. Se omitido, usa o primeiro bairro disponível.")
    parser.add_argument("--quadra", help="Filtro opcional de quadra.")
    parser.add_argument("--bloco", help="Filtro opcional de bloco.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, base_path = load_data(args.input)

    bairro = args.bairro or choose_default_bairro(df)
    scoped = filter_data(df, bairro=bairro, quadra=args.quadra, bloco=args.bloco)
    if scoped.empty:
        raise ValueError("Nenhum registro encontrado para os filtros informados.")

    filters = FilterScope(bairro=bairro, quadra=args.quadra, bloco=args.bloco)
    indicators = compute_indicators(scoped, filters)

    default_name = f"estudo_imobiliario_{bairro.lower().replace(' ', '_')}"
    if args.quadra:
        default_name += f"_{args.quadra.lower().replace(' ', '_')}"
    if args.bloco:
        default_name += f"_{args.bloco.lower().replace(' ', '_')}"

    output_path = Path(args.output) if args.output else _repo_root() / "reports" / f"{default_name}.pdf"

    with tempfile.TemporaryDirectory(prefix="study_charts_") as tmpdir:
        charts = generate_charts(indicators, Path(tmpdir))
        pdf_path = build_pdf(indicators, charts, output_path)

    append_pipeline_log(
        resolve_log_path(),
        {
            "step": "generate_pdf",
            "report_type": "estudo_imobiliario_pdf",
            "input_path": str(base_path),
            "output_path": str(pdf_path),
            "bairro": filters.bairro,
            "quadra": filters.quadra,
            "bloco": filters.bloco,
            "records_used": indicators.total_records,
            "mes_ref": indicators.last_month,
        },
    )

    print(f"Relatório gerado com sucesso: {pdf_path}")
    print(f"Filtro aplicado: {filters.subtitle()}")
    print(f"Registros utilizados: {indicators.total_records}")
    if args.bairro is None:
        print(f"Bairro não informado. Bairro padrão utilizado: {bairro}")


if __name__ == "__main__":
    main()
