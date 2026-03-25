from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
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

COLOR_PRIMARY = "#1E88E5"
COLOR_SECONDARY = "#2831A7"
COLOR_ORANGE = "#F06D2F"
COLOR_PURPLE = "#8E24AA"
COLOR_PINK = "#D94FB8"
COLOR_TEXT = "#222222"
COLOR_MUTED = "#666666"
COLOR_GRID = "#E5E5E5"
PAGE_BG = "#EFEFEF"
DEFAULT_PALETTE = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ORANGE, COLOR_PURPLE, COLOR_PINK]
VAGA_COLORS = {"Com Vaga": COLOR_PRIMARY, "Sem Vaga": COLOR_SECONDARY, "Geral": "#404040"}

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 12})


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
class Indicators:
    filters: FilterScope
    total_records: int
    last_month: str | None
    overview_last_month: dict[str, float | str | None]
    overview_series: pd.DataFrame
    bedrooms_series: pd.DataFrame
    area_series: pd.DataFrame
    quadra_series: pd.DataFrame


@dataclass(frozen=True)
class PageCharts:
    header: str
    intro: str
    highlight_lines: tuple[str, ...]
    top_chart_path: Path
    top_caption: str
    bottom_chart_path: Path
    bottom_caption: str


@dataclass(frozen=True)
class ReportResult:
    pdf_path: Path
    indicators: Indicators
    base_path: Path


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

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def resolve_base_path(explicit_path: str | None = None) -> Path:
    for candidate in _candidate_paths(explicit_path):
        if candidate.exists():
            return candidate
    searched = "\n - ".join(str(path) for path in _candidate_paths(explicit_path))
    raise FileNotFoundError(f"Base analítica não encontrada. Caminhos verificados:\n - {searched}")


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


def month_label(value: str | None) -> str:
    if not value or pd.isna(value):
        return "mês indisponível"
    month_names = {
        1: "janeiro",
        2: "fevereiro",
        3: "março",
        4: "abril",
        5: "maio",
        6: "junho",
        7: "julho",
        8: "agosto",
        9: "setembro",
        10: "outubro",
        11: "novembro",
        12: "dezembro",
    }
    dt = pd.to_datetime(f"{value}-01", errors="coerce")
    if pd.isna(dt):
        return str(value)
    return month_names[dt.month]


def load_data(input_path: str | None = None) -> tuple[pd.DataFrame, Path]:
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
    work = work.dropna(subset=["mes_ref_dt"]).sort_values("mes_ref_dt").reset_index(drop=True)
    return work, base_path


def filter_data(
    df: pd.DataFrame,
    bairro: str,
    quadra: str | None = None,
    bloco: str | None = None,
) -> pd.DataFrame:
    if not bairro or not bairro.strip():
        raise ValueError("O filtro de bairro é obrigatório para gerar o estudo imobiliário.")

    work = df.copy()
    work = work[work["bairro_padronizado"].fillna("").str.casefold() == bairro.strip().casefold()]

    if quadra:
        work = work[work["grupo_quadra"].fillna("").str.casefold() == quadra.strip().casefold()]

    if bloco:
        work = work[work["bloco_padronizado"].fillna("").str.casefold() == bloco.strip().casefold()]

    return work.sort_values("mes_ref_dt").reset_index(drop=True)


def _aggregate_series(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str = "valor_m2_calc",
    min_records: int = MIN_RECORDS_PER_MONTH,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["mes_ref", "mes_ref_dt", *group_cols, "valor_m2_medio", "qtd", "mm3_valor_m2"])

    base = (
        df.dropna(subset=["mes_ref", "mes_ref_dt", value_col, *group_cols])
        .groupby(["mes_ref", "mes_ref_dt", *group_cols], as_index=False, dropna=False)
        .agg(valor_m2_medio=(value_col, "mean"), qtd=(value_col, "size"))
    )

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

    last_month_dt = df["mes_ref_dt"].max()
    ref = df[df["mes_ref_dt"] == last_month_dt].copy()
    return {
        "mes_ref": last_month_dt.strftime("%Y-%m"),
        "valor_m2_geral": ref["valor_m2_calc"].mean(),
        "valor_m2_com_vaga": ref.loc[ref["tem_vaga"] == "Com Vaga", "valor_m2_calc"].mean(),
        "valor_m2_sem_vaga": ref.loc[ref["tem_vaga"] == "Sem Vaga", "valor_m2_calc"].mean(),
        "amostra": int(len(ref)),
    }


def compute_indicators(df: pd.DataFrame, filters: FilterScope) -> Indicators:
    overview_by_vaga = _aggregate_series(df, ["tem_vaga"]).rename(columns={"tem_vaga": "serie"})
    overall = _aggregate_series(df.assign(serie="Geral"), ["serie"])
    overview_series = pd.concat([overall, overview_by_vaga], ignore_index=True, sort=False)

    bedrooms_series = _aggregate_series(df, ["quartos_num", "tem_vaga"])
    area_series = _aggregate_series(df, ["faixa_metragem", "tem_vaga"])
    quadra_series = (
        _aggregate_series(df, ["grupo_quadra", "tem_vaga"])
        if not filters.quadra
        else pd.DataFrame(columns=["mes_ref", "mes_ref_dt", "grupo_quadra", "tem_vaga", "valor_m2_medio", "qtd", "mm3_valor_m2"])
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


def choose_default_bairro(df: pd.DataFrame) -> str:
    bairros = sorted({value for value in df["bairro_padronizado"].dropna().tolist() if str(value).strip()})
    if not bairros:
        raise ValueError("Não há bairros válidos na base analítica para gerar um relatório padrão.")
    return bairros[0]


def _series_palette(labels: list[str]) -> dict[str, str]:
    palette: dict[str, str] = {}
    for idx, label in enumerate(labels):
        palette[str(label)] = DEFAULT_PALETTE[idx % len(DEFAULT_PALETTE)]
    return palette


def _last_month_snapshot(series_df: pd.DataFrame, group_col: str, extra_filter: tuple[str, str] | None = None) -> pd.DataFrame:
    if series_df.empty:
        return pd.DataFrame(columns=[group_col, "valor_m2_medio", "mes_ref", "mes_ref_dt", "qtd"])

    work = series_df.copy()
    if extra_filter is not None:
        filter_col, filter_value = extra_filter
        work = work[work[filter_col] == filter_value]

    if work.empty:
        return pd.DataFrame(columns=work.columns)

    last_month_dt = work["mes_ref_dt"].max()
    snapshot = work[work["mes_ref_dt"] == last_month_dt].copy()
    return snapshot.sort_values("valor_m2_medio", ascending=False).reset_index(drop=True)


def _annotate_bar_values(ax: plt.Axes, bars, values: list[float]) -> None:
    max_value = max(values) if values else 0
    offset = max_value * 0.03 if max_value else 0.5
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            format_currency(value),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=10,
            color=COLOR_TEXT,
        )


def _style_bar_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=11, color=COLOR_TEXT, pad=12)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_facecolor("white")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", length=0, labelsize=11, colors=COLOR_MUTED)
    ax.grid(False)


def _style_line_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_facecolor("white")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", length=0, labelsize=12, colors=COLOR_MUTED)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)


def _plot_overview_top(indicators: Indicators, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 2.7), facecolor="white")
    snapshot = pd.DataFrame(
        [
            {"serie": "Com Vaga", "valor": indicators.overview_last_month["valor_m2_com_vaga"]},
            {"serie": "Sem Vaga", "valor": indicators.overview_last_month["valor_m2_sem_vaga"]},
        ]
    ).dropna(subset=["valor"])

    if snapshot.empty:
        ax.text(0.5, 0.5, "Sem dados suficientes para o último mês.", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        colors_used = [VAGA_COLORS.get(label, COLOR_PRIMARY) for label in snapshot["serie"]]
        bars = ax.bar(snapshot["serie"], snapshot["valor"], color=colors_used, width=0.5)
        _annotate_bar_values(ax, bars, snapshot["valor"].tolist())
        _style_bar_axis(ax, f"Valor médio do m² no mês de {month_label(indicators.overview_last_month['mes_ref'])}")
        ax.set_ylim(0, snapshot["valor"].max() * 1.28)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8)
            for color in colors_used
        ]
        ax.legend(handles, snapshot["serie"], loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _plot_line_chart(
    series_df: pd.DataFrame,
    group_cols: list[str],
    output_path: Path,
    title: str,
    label_formatter=None,
    use_vaga_palette: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(10.2, 4.0), facecolor="white")
    _style_line_axis(ax, title)

    if series_df.empty:
        ax.text(0.5, 0.5, "Sem dados suficientes para esta análise.", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path

    plot_df = series_df.sort_values([*group_cols, "mes_ref_dt"]).copy()
    if len(group_cols) == 1:
        plot_df["legend_label"] = plot_df[group_cols[0]].astype(str)
    else:
        plot_df["legend_label"] = plot_df[group_cols].astype(str).agg(" | ".join, axis=1)

    if label_formatter is not None:
        plot_df["legend_label"] = plot_df["legend_label"].map(label_formatter)

    labels = plot_df["legend_label"].drop_duplicates().tolist()
    palette = VAGA_COLORS if use_vaga_palette else _series_palette(labels)

    for label, group in plot_df.groupby("legend_label", sort=False):
        color = palette.get(label, DEFAULT_PALETTE[0])
        ax.plot(group["mes_ref_dt"], group["mm3_valor_m2"], color=color, linewidth=2.2)
        for _, row in group.iterrows():
            ax.text(
                row["mes_ref_dt"],
                row["mm3_valor_m2"] + (plot_df["mm3_valor_m2"].max() * 0.02),
                format_currency(row["mm3_valor_m2"]),
                fontsize=9,
                color=COLOR_TEXT,
                ha="center",
            )

    month_ticks = plot_df[["mes_ref_dt", "mes_ref"]].drop_duplicates().sort_values("mes_ref_dt")
    ax.set_xticks(month_ticks["mes_ref_dt"])
    ax.set_xticklabels([month_label(value) for value in month_ticks["mes_ref"]])

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=palette.get(label, DEFAULT_PALETTE[0]), markersize=7)
        for label in labels
    ]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=min(4, max(1, len(labels))), frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _plot_dual_bar_last_month(
    series_df: pd.DataFrame,
    group_col: str,
    output_path: Path,
    title_left: str,
    title_right: str,
    label_formatter=None,
    max_categories: int | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.2), facecolor="white")
    snapshots = {
        "Com Vaga": _last_month_snapshot(series_df, group_col, ("tem_vaga", "Com Vaga")),
        "Sem Vaga": _last_month_snapshot(series_df, group_col, ("tem_vaga", "Sem Vaga")),
    }

    categories = []
    for snapshot in snapshots.values():
        if snapshot.empty:
            continue
        categories.extend(snapshot[group_col].astype(str).tolist())
    category_order = list(dict.fromkeys(categories))
    if max_categories is not None:
        category_order = category_order[:max_categories]
    palette = _series_palette(category_order)

    for ax, (vaga_label, snapshot), title in zip(axes, snapshots.items(), [title_left, title_right]):
        _style_bar_axis(ax, title)
        if snapshot.empty:
            ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", fontsize=11)
            ax.axis("off")
            continue

        if max_categories is not None:
            snapshot = snapshot.head(max_categories)

        labels = snapshot[group_col].astype(str).tolist()
        if label_formatter is not None:
            labels = [label_formatter(label) for label in labels]
        values = snapshot["valor_m2_medio"].tolist()
        colors_used = [palette.get(str(value), DEFAULT_PALETTE[0]) for value in snapshot[group_col].astype(str)]
        bars = ax.bar(labels, values, color=colors_used, width=0.55)
        _annotate_bar_values(ax, bars, values)
        ax.set_ylim(0, max(values) * 1.3)

        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=color, markersize=7)
            for color in colors_used
        ]
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=min(4, len(labels)), frameon=False, handlelength=0)

    fig.tight_layout(w_pad=1.4)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _page_intro_overview(indicators: Indicators) -> str:
    bairro = indicators.filters.bairro
    return (
        f"Este é um estudo gerado com base na inteligência imobiliária do projeto sobre as variações "
        f"do valor do m² em {bairro}. O material consolida os anúncios filtrados e destaca o comportamento "
        f"mais recente do mercado para apoiar análise comercial e tomada de decisão."
    )


def _page_highlights_overview(indicators: Indicators) -> tuple[str, ...]:
    last_month = indicators.overview_last_month["mes_ref"]
    return (
        f"O valor médio do m² no último mês ({month_label(last_month)}) foi de {format_currency(indicators.overview_last_month['valor_m2_geral'])}.",
        f"Para imóveis com vaga de garagem esse valor foi {format_currency(indicators.overview_last_month['valor_m2_com_vaga'])}.",
        f"Já para os imóveis sem vaga o valor observado foi {format_currency(indicators.overview_last_month['valor_m2_sem_vaga'])}.",
    )


def _page_intro_rooms() -> str:
    return (
        "A quantidade de quartos também interfere no valor do imóvel. Nesta página observamos como esse "
        "fator influencia o preço do metro quadrado, separando os imóveis com e sem vaga de garagem."
    )


def _page_intro_area() -> str:
    return (
        "Outro fator importante é a metragem do imóvel. Nesta página avaliamos como as diferenças de tamanho "
        "nominal total influenciam no valor do metro quadrado ao longo do tempo."
    )


def _page_intro_quadra() -> str:
    return (
        "A localização do imóvel também causa variação nos preços. Nesta página analisamos como essa "
        "variação ocorre entre grupos de quadra dentro do recorte selecionado."
    )


def generate_charts(indicators: Indicators, charts_dir: Path) -> list[PageCharts]:
    charts_dir.mkdir(parents=True, exist_ok=True)

    overview_page = PageCharts(
        header=f"Report {indicators.filters.bairro}",
        intro=_page_intro_overview(indicators),
        highlight_lines=_page_highlights_overview(indicators),
        top_chart_path=_plot_overview_top(indicators, charts_dir / "page_1_top.png"),
        top_caption=(
            f"O gráfico acima apresenta a média do valor do m² em {indicators.filters.bairro} no mês mais recente, "
            "comparando imóveis com e sem vaga de garagem."
        ),
        bottom_chart_path=_plot_line_chart(
            indicators.overview_series,
            ["serie"],
            charts_dir / "page_1_bottom.png",
            "Evolução do valor do m² nos últimos meses",
            use_vaga_palette=True,
        ),
        bottom_caption="O gráfico abaixo mostra a evolução mensal suavizada por média móvel de 3 períodos.",
    )

    room_label = lambda text: f"Quartos {int(float(text))}" if str(text).replace('.', '', 1).isdigit() else str(text)
    rooms_page = PageCharts(
        header="Valor do m² por número de quartos",
        intro=_page_intro_rooms(),
        highlight_lines=(),
        top_chart_path=_plot_dual_bar_last_month(
            indicators.bedrooms_series,
            "quartos_num",
            charts_dir / "page_2_top.png",
            "Valor do m² por n. de quartos com garagem",
            "Valor do m² por n. de quartos sem garagem",
            label_formatter=room_label,
        ),
        top_caption="Acima, o valor do m² por quartos aparece separado por imóveis com e sem garagem no último mês disponível.",
        bottom_chart_path=_plot_line_chart(
            indicators.bedrooms_series.assign(quartos_label=indicators.bedrooms_series["quartos_num"].astype(str).map(room_label)),
            ["quartos_label", "tem_vaga"],
            charts_dir / "page_2_bottom.png",
            "Evolução do valor do m² por número de quartos",
        ),
        bottom_caption="Abaixo, a evolução mensal é apresentada por número de quartos e tipo de vaga.",
    )

    area_page = PageCharts(
        header="Valor do m² por metragem",
        intro=_page_intro_area(),
        highlight_lines=(),
        top_chart_path=_plot_dual_bar_last_month(
            indicators.area_series,
            "faixa_metragem",
            charts_dir / "page_3_top.png",
            "Valor de m² por metragem com garagem",
            "Valor de m² por metragem sem garagem",
            max_categories=5,
        ),
        top_caption="O gráfico acima mostra o recorte do último mês por faixa de metragem, separado entre imóveis com e sem vaga.",
        bottom_chart_path=_plot_line_chart(
            indicators.area_series.assign(serie=indicators.area_series["faixa_metragem"].astype(str) + " | " + indicators.area_series["tem_vaga"].astype(str)),
            ["serie"],
            charts_dir / "page_3_bottom.png",
            "Evolução do valor do m² por metragem",
        ),
        bottom_caption="O gráfico abaixo explicita a evolução das faixas de metragem ao longo dos meses.",
    )

    pages = [overview_page, rooms_page, area_page]

    if not indicators.filters.quadra:
        quadra_page = PageCharts(
            header="Valor do m² por quadra",
            intro=_page_intro_quadra(),
            highlight_lines=(),
            top_chart_path=_plot_dual_bar_last_month(
                indicators.quadra_series,
                "grupo_quadra",
                charts_dir / "page_4_top.png",
                "Valor do m² por quadra com garagem",
                "Valor do m² por quadra sem garagem",
                max_categories=4,
            ),
            top_caption="Acima, a média do valor do m² por grupo de quadra é apresentada no último mês com dados suficientes.",
            bottom_chart_path=_plot_line_chart(
                indicators.quadra_series.assign(serie=indicators.quadra_series["grupo_quadra"].astype(str) + " | " + indicators.quadra_series["tem_vaga"].astype(str)),
                ["serie"],
                charts_dir / "page_4_bottom.png",
                "Evolução do valor do m² por quadra",
            ),
            bottom_caption="Abaixo, a evolução mensal mostra como o comportamento de preço varia entre as quadras do recorte.",
        )
        pages.append(quadra_page)

    return pages


def _draw_brand(pdf: canvas.Canvas, page_width: float, page_height: float) -> None:
    x = page_width - 78
    y = page_height - 48
    pdf.setFillColor(colors.HexColor("#FFC400"))
    pdf.rect(x, y - 6, 18, 18, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor(COLOR_SECONDARY))
    pdf.setFont("Helvetica", 36)
    pdf.drawString(x - 2, y - 10, "6")
    pdf.drawString(x + 18, y - 10, "1")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(x - 2, y - 28, "IMÓVEIS")
    pdf.setFillColor(colors.black)


def _draw_wrapped_text(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font_name: str = "Helvetica",
    font_size: int = 12,
    align: str = "left",
    color: str = COLOR_TEXT,
    leading: float | None = None,
) -> float:
    lines = simpleSplit(text, font_name, font_size, width)
    leading = leading or font_size * 1.35
    pdf.setFillColor(colors.HexColor(color))
    pdf.setFont(font_name, font_size)
    current_y = y
    for line in lines:
        if align == "center":
            pdf.drawCentredString(x + width / 2, current_y, line)
        else:
            pdf.drawString(x, current_y, line)
        current_y -= leading
    pdf.setFillColor(colors.black)
    return current_y


def _draw_page(pdf: canvas.Canvas, page: PageCharts, subtitle: str) -> None:
    page_width, page_height = A4
    margin = 54
    content_width = page_width - (margin * 2)

    pdf.setFillColor(colors.HexColor(PAGE_BG))
    pdf.rect(0, 0, page_width, page_height, stroke=0, fill=1)
    pdf.setFillColor(colors.black)

    _draw_brand(pdf, page_width, page_height)

    current_y = page_height - 68
    title_font_size = 36 if page.header.startswith("Report ") else 22
    current_y = _draw_wrapped_text(
        pdf,
        page.header,
        margin,
        current_y,
        content_width,
        font_name="Helvetica",
        font_size=title_font_size,
        align="center" if page.header.startswith("Report ") else "left",
        leading=title_font_size * 1.05,
    )
    current_y -= 8
    current_y = _draw_wrapped_text(
        pdf,
        subtitle,
        margin,
        current_y,
        content_width,
        font_name="Helvetica",
        font_size=10,
        align="center" if page.header.startswith("Report ") else "left",
        color=COLOR_MUTED,
    )
    current_y -= 16
    current_y = _draw_wrapped_text(
        pdf,
        page.intro,
        margin,
        current_y,
        content_width,
        font_name="Helvetica",
        font_size=12,
        align="center" if page.header.startswith("Report ") else "left",
    )
    current_y -= 6

    for highlight in page.highlight_lines:
        current_y = _draw_wrapped_text(
            pdf,
            highlight,
            margin,
            current_y,
            content_width,
            font_name="Helvetica-Bold",
            font_size=12,
            align="center",
        )
    current_y -= 10

    top_image_height = 150
    bottom_image_height = 205
    image_x = margin + 10
    image_width = content_width - 20

    pdf.drawImage(ImageReader(str(page.top_chart_path)), image_x, current_y - top_image_height, width=image_width, height=top_image_height, preserveAspectRatio=True, mask="auto")
    current_y -= top_image_height + 12
    current_y = _draw_wrapped_text(pdf, page.top_caption, margin, current_y, content_width, font_name="Helvetica", font_size=11)
    current_y -= 10

    pdf.drawImage(ImageReader(str(page.bottom_chart_path)), image_x, current_y - bottom_image_height, width=image_width, height=bottom_image_height, preserveAspectRatio=True, mask="auto")
    current_y -= bottom_image_height + 10
    _draw_wrapped_text(pdf, page.bottom_caption, margin, max(current_y, 80), content_width, font_name="Helvetica", font_size=11)

    pdf.setFont("Helvetica", 11)
    pdf.setFillColor(colors.HexColor(COLOR_TEXT))
    pdf.drawCentredString(page_width / 2, 38, "61IMÓVEIS.COM")
    pdf.setLineWidth(0.6)
    pdf.setStrokeColor(colors.HexColor("#c9c9c9"))
    pdf.line(margin, 52, page_width - margin, 52)
    pdf.setFillColor(colors.black)
    pdf.showPage()


def build_pdf(indicators: Indicators, charts: Iterable[PageCharts], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=A4)

    for chart in charts:
        _draw_page(pdf, chart, indicators.filters.subtitle())

    pdf.save()
    return output_path


def generate_report_pdf(
    bairro: str,
    quadra: str | None = None,
    bloco: str | None = None,
    input_path: str | None = None,
    output_path: str | None = None,
) -> ReportResult:
    df, base_path = load_data(input_path)
    scoped = filter_data(df, bairro=bairro, quadra=quadra, bloco=bloco)
    if scoped.empty:
        raise ValueError("Nenhum registro encontrado para os filtros informados.")

    filters = FilterScope(bairro=bairro, quadra=quadra, bloco=bloco)
    indicators = compute_indicators(scoped, filters)

    default_name = f"estudo_imobiliario_{bairro.lower().replace(' ', '_')}"
    if quadra:
        default_name += f"_{quadra.lower().replace(' ', '_')}"
    if bloco:
        default_name += f"_{bloco.lower().replace(' ', '_')}"

    pdf_path = Path(output_path) if output_path else _repo_root() / "reports" / f"{default_name}.pdf"

    with tempfile.TemporaryDirectory(prefix="study_charts_") as tmpdir:
        charts = generate_charts(indicators, Path(tmpdir))
        build_pdf(indicators, charts, pdf_path)

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

    return ReportResult(pdf_path=pdf_path, indicators=indicators, base_path=base_path)


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
    df, _ = load_data(args.input)
    bairro = args.bairro or choose_default_bairro(df)
    result = generate_report_pdf(
        bairro=bairro,
        quadra=args.quadra,
        bloco=args.bloco,
        input_path=args.input,
        output_path=args.output,
    )

    print(f"Relatório gerado com sucesso: {result.pdf_path}")
    print(f"Filtro aplicado: {result.indicators.filters.subtitle()}")
    print(f"Registros utilizados: {result.indicators.total_records}")
    if args.bairro is None:
        print(f"Bairro não informado. Bairro padrão utilizado: {bairro}")


if __name__ == "__main__":
    main()
