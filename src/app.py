from __future__ import annotations

import argparse
import os
import sys
import tempfile
import tkinter as tk
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from types import SimpleNamespace

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
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
            "O módulo generate_pdf.py não expõe a interface necessária para o app. "
            f"Itens ausentes: {missing}"
        )

    def _generate_report_pdf(
        bairro: str,
        quadra: str | None = None,
        bloco: str | None = None,
        input_path: str | None = None,
        output_path: str | None = None,
    ):
        df, base_path = module.load_data(input_path)
        scoped = module.filter_data(df, bairro=bairro, quadra=quadra, bloco=bloco)
        if scoped.empty:
            raise ValueError("Nenhum registro encontrado para os filtros informados.")

        filters = module.FilterScope(bairro=bairro, quadra=quadra, bloco=bloco)
        indicators = module.compute_indicators(scoped, filters)

        default_name = f"estudo_imobiliario_{bairro.lower().replace(' ', '_')}"
        if quadra:
            default_name += f"_{quadra.lower().replace(' ', '_')}"
        if bloco:
            default_name += f"_{bloco.lower().replace(' ', '_')}"

        pdf_path = Path(output_path) if output_path else APP_DIR.parent / "reports" / f"{default_name}.pdf"

        with tempfile.TemporaryDirectory(prefix="study_charts_") as tmpdir:
            charts = module.generate_charts(indicators, Path(tmpdir))
            module.build_pdf(indicators, charts, pdf_path)

        return SimpleNamespace(pdf_path=pdf_path, indicators=indicators, base_path=base_path)

    return _generate_report_pdf


choose_default_bairro = getattr(GENERATE_PDF_MODULE, "choose_default_bairro", _fallback_choose_default_bairro)
format_currency = getattr(GENERATE_PDF_MODULE, "format_currency", _fallback_format_currency)
load_data = getattr(GENERATE_PDF_MODULE, "load_data")
generate_report_pdf = getattr(GENERATE_PDF_MODULE, "generate_report_pdf", _build_generate_report_fallback(GENERATE_PDF_MODULE))


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


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


def _print_cli_preview(preview: dict, base_path: Path) -> None:
    print("\nPrévia do recorte analítico")
    print(f"- Base utilizada: {base_path}")
    print(f"- Bairro: {preview['bairro']}")
    print(f"- Quadra: {preview['quadra'] or 'todas'}")
    print(f"- Bloco: {preview['bloco'] or 'todos'}")
    print(f"- Registros: {_format_int(preview['registros'])}")
    print(f"- Meses disponíveis: {', '.join(preview['meses']) if preview['meses'] else 'nenhum'}")


def run_cli(args: argparse.Namespace) -> None:
    df, base_path = load_data(args.input)

    if args.listar:
        print("Bairros disponíveis:")
        for bairro in _options_from_series(df["bairro_padronizado"]):
            print(f"- {bairro}")
        return

    bairro = args.bairro or choose_default_bairro(df)
    quadra = args.quadra
    bloco = args.bloco

    preview = _scope_preview(df, bairro=bairro, quadra=quadra, bloco=bloco)
    _print_cli_preview(preview, base_path)

    if preview["registros"] == 0:
        raise ValueError("Nenhum registro encontrado para os filtros selecionados.")

    result = generate_report_pdf(
        bairro=bairro,
        quadra=quadra,
        bloco=bloco,
        input_path=args.input,
        output_path=args.output,
    )

    print("\nPDF gerado com sucesso")
    print(f"- Arquivo: {result.pdf_path}")
    print(f"- Registros usados: {_format_int(result.indicators.total_records)}")
    print(f"- Mês de referência: {result.indicators.last_month}")
    print(f"- Valor médio geral do m²: {format_currency(result.indicators.overview_last_month.get('valor_m2_geral'))}")


class AppGUI:
    def __init__(self, root: tk.Tk, input_path: str | None = None):
        self.root = root
        self.root.title("Estudo Imobiliário — Gerador de PDF")
        self.root.geometry("760x430")

        self.df, self.base_path = load_data(input_path)

        self.bairro_var = tk.StringVar()
        self.quadra_var = tk.StringVar()
        self.bloco_var = tk.StringVar()
        self.output_var = tk.StringVar(value="")

        self._build_layout()
        self._load_bairros()

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        ttk.Label(main, text="Gerador de Estudo Imobiliário em PDF", font=("Arial", 15, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(main, text=f"Base: {self.base_path}", foreground="#444444").grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 14))

        ttk.Label(main, text="Bairro").grid(row=2, column=0, sticky="w")
        self.bairro_combo = ttk.Combobox(main, textvariable=self.bairro_var, state="readonly", width=30)
        self.bairro_combo.grid(row=3, column=0, sticky="we", padx=(0, 8))
        self.bairro_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_bairro_change())

        ttk.Label(main, text="Quadra (opcional)").grid(row=2, column=1, sticky="w")
        self.quadra_combo = ttk.Combobox(main, textvariable=self.quadra_var, state="readonly", width=30)
        self.quadra_combo.grid(row=3, column=1, sticky="we", padx=8)
        self.quadra_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_quadra_change())

        ttk.Label(main, text="Bloco (opcional)").grid(row=2, column=2, sticky="w")
        self.bloco_combo = ttk.Combobox(main, textvariable=self.bloco_var, state="readonly", width=30)
        self.bloco_combo.grid(row=3, column=2, sticky="we", padx=(8, 0))
        self.bloco_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_preview())

        output_row = ttk.Frame(main)
        output_row.grid(row=4, column=0, columnspan=3, sticky="we", pady=(18, 8))
        output_row.columnconfigure(0, weight=1)
        ttk.Label(output_row, text="Arquivo de saída (opcional)").grid(row=0, column=0, sticky="w")
        ttk.Entry(output_row, textvariable=self.output_var).grid(row=1, column=0, sticky="we", padx=(0, 8))
        ttk.Button(output_row, text="Selecionar", command=self._choose_output_path).grid(row=1, column=1)

        self.preview_label = ttk.Label(main, text="", justify="left", foreground="#222222")
        self.preview_label.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 18))

        actions = ttk.Frame(main)
        actions.grid(row=6, column=0, columnspan=3, sticky="e")
        ttk.Button(actions, text="Atualizar prévia", command=self._refresh_preview).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Gerar PDF", command=self._generate_pdf).pack(side="left")

        for col in (0, 1, 2):
            main.columnconfigure(col, weight=1)

    def _load_bairros(self) -> None:
        bairros = _options_from_series(self.df["bairro_padronizado"])
        if not bairros:
            raise ValueError("Não há bairros disponíveis na base analítica.")

        self.bairro_combo["values"] = bairros
        self.bairro_var.set(bairros[0])
        self._on_bairro_change()

    def _on_bairro_change(self) -> None:
        bairro = self.bairro_var.get()
        scoped = self.df[self.df["bairro_padronizado"].fillna("") == bairro]
        quadras = ["(todas)"] + _options_from_series(scoped["grupo_quadra"])
        self.quadra_combo["values"] = quadras
        self.quadra_var.set("(todas)")
        self._on_quadra_change()

    def _on_quadra_change(self) -> None:
        bairro = self.bairro_var.get()
        quadra = self._quadra_value()
        scoped = self.df[self.df["bairro_padronizado"].fillna("") == bairro]
        if quadra:
            scoped = scoped[scoped["grupo_quadra"].fillna("") == quadra]

        blocos = ["(todos)"] + _options_from_series(scoped["bloco_padronizado"])
        self.bloco_combo["values"] = blocos
        self.bloco_var.set("(todos)")
        self._refresh_preview()

    def _quadra_value(self) -> str | None:
        value = self.quadra_var.get().strip()
        return None if value in {"", "(todas)"} else value

    def _bloco_value(self) -> str | None:
        value = self.bloco_var.get().strip()
        return None if value in {"", "(todos)"} else value

    def _choose_output_path(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Salvar relatório PDF",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
        )
        if selected:
            self.output_var.set(selected)

    def _refresh_preview(self) -> None:
        preview = _scope_preview(
            self.df,
            bairro=self.bairro_var.get(),
            quadra=self._quadra_value(),
            bloco=self._bloco_value(),
        )
        text = (
            f"Bairro: {preview['bairro']}\n"
            f"Quadra: {preview['quadra'] or 'todas'}\n"
            f"Bloco: {preview['bloco'] or 'todos'}\n"
            f"Registros: {_format_int(preview['registros'])}\n"
            f"Meses disponíveis: {', '.join(preview['meses']) if preview['meses'] else 'nenhum'}"
        )
        self.preview_label.config(text=text)

    def _generate_pdf(self) -> None:
        try:
            result = generate_report_pdf(
                bairro=self.bairro_var.get(),
                quadra=self._quadra_value(),
                bloco=self._bloco_value(),
                output_path=self.output_var.get().strip() or None,
            )
        except Exception as exc:
            messagebox.showerror("Erro ao gerar relatório", str(exc))
            return

        info = (
            f"PDF gerado com sucesso!\n\n"
            f"Arquivo: {result.pdf_path}\n"
            f"Registros usados: {_format_int(result.indicators.total_records)}\n"
            f"Mês de referência: {result.indicators.last_month}\n"
            f"Valor médio geral do m²: {format_currency(result.indicators.overview_last_month.get('valor_m2_geral'))}"
        )
        messagebox.showinfo("Relatório gerado", info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface gráfica para gerar o estudo imobiliário em PDF.")
    parser.add_argument("--cli", action="store_true", help="Executa em modo linha de comando (sem janela).")
    parser.add_argument("--input", help="Caminho opcional para a base analítica CSV.")
    parser.add_argument("--output", help="Caminho opcional para o PDF de saída (modo CLI).")
    parser.add_argument("--bairro", help="Bairro do estudo (modo CLI).")
    parser.add_argument("--quadra", help="Filtro opcional de quadra (modo CLI).")
    parser.add_argument("--bloco", help="Filtro opcional de bloco (modo CLI).")
    parser.add_argument("--listar", action="store_true", help="Lista os bairros disponíveis e encerra (modo CLI).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cli:
        run_cli(args)
        return

    root = tk.Tk()
    try:
        app = AppGUI(root, input_path=args.input)
        if args.bairro:
            if args.bairro in app.bairro_combo["values"]:
                app.bairro_var.set(args.bairro)
                app._on_bairro_change()
        if args.quadra:
            values = list(app.quadra_combo["values"])
            if args.quadra in values:
                app.quadra_var.set(args.quadra)
                app._on_quadra_change()
        if args.bloco:
            values = list(app.bloco_combo["values"])
            if args.bloco in values:
                app.bloco_var.set(args.bloco)
                app._refresh_preview()
        if args.output:
            app.output_var.set(args.output)
    except Exception as exc:
        messagebox.showerror("Erro ao iniciar aplicação", str(exc))
        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    main()
