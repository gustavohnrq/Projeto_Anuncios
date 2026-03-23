# Projeto — Estudo Imobiliário v1

Primeira versão do projeto para gerar estudos em PDF a partir da base incremental de imóveis.

## Objetivo desta v1
Esta versão prepara a fundação do projeto:

- padronização da base incremental
- filtro por bairro / quadra / bloco
- criação de base analítica mensal
- suavização por média móvel de 3 meses
- estrutura inicial para clusterização de conservação
- registro incremental de execuções e decisões do pipeline
- interface inicial em Streamlit para escolha dos filtros

Nesta etapa, o PDF ainda está em modo **rascunho funcional**:
ele já pode ser gerado depois, mas a prioridade aqui é montar a fundação analítica.

## Estrutura
- `src/prepare_base.py` — limpeza, padronização e base analítica
- `src/generate_report.py` — cálculos dos indicadores do estudo
- `src/app.py` — interface inicial em Streamlit
- `src/config.py` — parâmetros centrais do projeto
- `src/logger_store.py` — grava histórico de execuções
- `requirements.txt` — dependências do projeto

## Como instalar no Mac
Crie e ative o ambiente virtual:

```bash
cd /Users/macbook/Desktop/Corretagem_2026/Coletas
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

## Como usar a preparação da base
Edite os caminhos absolutos dentro de `src/config.py`.

Depois rode:

```bash
python src/prepare_base.py
```

Isso irá:
1. ler a base incremental
2. padronizar colunas
3. tratar tipos
4. calcular `valor_m2_calc`
5. criar colunas analíticas
6. salvar a base tratada em `data/base_analitica.csv`
7. registrar a execução em `logs/pipeline_history.jsonl`

## Como abrir a interface
```bash
streamlit run src/app.py
```

## Etapas seguintes
1. validar a base tratada
2. validar o filtro de bairro / quadra / bloco
3. revisar categorias de metragem
4. implantar remoção de outliers adaptativa
5. implantar clusterização de conservação
6. gerar PDF final com layout equivalente ao estudo de referência
