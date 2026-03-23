from __future__ import annotations

import argparse
import sys

import pandas as pd

from generate_pdf import choose_default_bairro, format_currency, generate_report_pdf, load_data


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _options_from_series(series: pd.Series) -> list[str]:
    return sorted([value for value in series.dropna().astype(str).unique().tolist() if value.strip()])


def _print_menu(title: str, options: list[str], allow_skip: bool = False) -> None:
    print(f"\n{title}")
    if allow_skip:
        print("  [0] Pular este filtro")
    for idx, option in enumerate(options, start=1):
        print(f"  [{idx}] {option}")


def _prompt_option(title: str, options: list[str], allow_skip: bool = False) -> str | None:
    if not options:
        return None

    while True:
        _print_menu(title, options, allow_skip=allow_skip)
        raw = input("Escolha uma opção pelo número: ").strip()

        if allow_skip and raw in {"", "0"}:
            return None

        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]

        print("Opção inválida. Tente novamente.")


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


def _interactive_filters(df: pd.DataFrame) -> tuple[str, str | None, str | None]:
    print("Estudo Imobiliário — seleção interativa")
    print("Escolha o recorte do relatório para gerar o PDF.\n")

    bairros = _options_from_series(df["bairro_padronizado"])
    if not bairros:
        raise ValueError("Não há bairros disponíveis na base analítica.")

    bairro = _prompt_option("Bairros disponíveis", bairros, allow_skip=False)
    assert bairro is not None

    scoped_bairro = df[df["bairro_padronizado"].fillna("") == bairro]
    quadras = _options_from_series(scoped_bairro["grupo_quadra"])
    quadra = _prompt_option("Quadras disponíveis", quadras, allow_skip=True)

    bloco = None
    if quadra:
        scoped_quadra = scoped_bairro[scoped_bairro["grupo_quadra"].fillna("") == quadra]
        blocos = _options_from_series(scoped_quadra["bloco_padronizado"])
        bloco = _prompt_option("Blocos disponíveis", blocos, allow_skip=True)

    return bairro, quadra, bloco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface de terminal para gerar o estudo imobiliário em PDF.")
    parser.add_argument("--input", help="Caminho opcional para a base analítica CSV.")
    parser.add_argument("--output", help="Caminho opcional para o PDF de saída.")
    parser.add_argument("--bairro", help="Bairro do estudo.")
    parser.add_argument("--quadra", help="Filtro opcional de quadra.")
    parser.add_argument("--bloco", help="Filtro opcional de bloco.")
    parser.add_argument("--listar", action="store_true", help="Lista os bairros disponíveis e encerra.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, base_path = load_data(args.input)

    if args.listar:
        print("Bairros disponíveis:")
        for bairro in _options_from_series(df["bairro_padronizado"]):
            print(f"- {bairro}")
        return

    if args.bairro:
        bairro = args.bairro
        quadra = args.quadra
        bloco = args.bloco
    elif sys.stdin.isatty():
        bairro, quadra, bloco = _interactive_filters(df)
    else:
        bairro = choose_default_bairro(df)
        quadra = None
        bloco = None
        print(f"Bairro não informado em modo não interativo. Utilizando bairro padrão: {bairro}")

    preview = _scope_preview(df, bairro=bairro, quadra=quadra, bloco=bloco)
    print("\nPrévia do recorte analítico")
    print(f"- Base utilizada: {base_path}")
    print(f"- Bairro: {preview['bairro']}")
    print(f"- Quadra: {preview['quadra'] or 'todas'}")
    print(f"- Bloco: {preview['bloco'] or 'todos'}")
    print(f"- Registros: {_format_int(preview['registros'])}")
    print(f"- Meses disponíveis: {', '.join(preview['meses']) if preview['meses'] else 'nenhum'}")

    if preview["registros"] == 0:
        raise ValueError("Nenhum registro encontrado para os filtros selecionados.")

    if sys.stdin.isatty():
        confirm = input("\nGerar o PDF com esse recorte? [s/N]: ").strip().lower()
        if confirm not in {"s", "sim", "y", "yes"}:
            print("Operação cancelada pelo usuário.")
            return

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


if __name__ == "__main__":
    main()
