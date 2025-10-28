
import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from miadp_laplace import MIADPAnonymizer

def infer_task_type(y: pd.Series) -> str:
    """
    Heuristic to infer classification vs regression:
    - If y is numeric and has <= 20 unique values or dtype is not float -> classification.
    - Else regression.
    """
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique <= 20 and not pd.api.types.is_float_dtype(y):
            return "classification"
        # If it's integer with small cardinality, also classify
        if pd.api.types.is_integer_dtype(y) and nunique <= 50:
            return "classification"
        return "regression"
    else:
        return "classification"

def _auto_find_csv(data_dir: Path) -> Path:
    """Find a single CSV inside data directory (excluding already anonymized *_anon.csv)."""
    if not data_dir.exists():
        raise SystemExit(f"Diretório de dados não existe: {data_dir}")
    csvs = [p for p in data_dir.glob("*.csv") if not p.name.endswith("_anon.csv")]
    if len(csvs) == 0:
        raise SystemExit("Nenhum arquivo CSV encontrado em ./data (exceto *_anon.csv).")
    if len(csvs) > 1:
        names = ", ".join(p.name for p in csvs)
        raise SystemExit(f"Mais de um CSV encontrado em ./data. Especifique com --input. Arquivos: {names}")
    return csvs[0]

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MIADP (Laplace) — Anonimiza um dataset adaptativamente usando Informação Mútua.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", required=False, help="Caminho do CSV de entrada. Se omitido será detectado automaticamente em ./data.")
    parser.add_argument("--output", "-o", required=False, help="Caminho do CSV anonimizado de saída. Se omitido usa <nome>_anon.csv em ./data.")
    parser.add_argument("--target", "-y", required=True, help="Nome da coluna alvo (target) no CSV. Todas as demais colunas serão tratadas como features para anonimização.")
    parser.add_argument("--epsilon", "-e", required=True, type=float, help="Epsilon global (>0). Menor => mais ruído.")
    parser.add_argument("--task", "-t", choices=["classification", "regression", "auto"], default="auto", help="Tipo de tarefa para estimar MI. 'auto' tenta inferir.")
    parser.add_argument("--sep", default=",", help="Separador do CSV.")
    parser.add_argument("--encoding", default="utf-8", help="Encoding do CSV.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed aleatória.")
    parser.add_argument("--report", default=None, help="Caminho opcional para salvar relatório de MI/epsilon.")
    args = parser.parse_args(argv)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    input_path = Path(args.input) if args.input else _auto_find_csv(data_dir)
    if not input_path.exists():
        raise SystemExit(f"Arquivo de entrada não existe: {input_path}")

    # Pasta de resultados
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # definir output sempre em results
    stem = input_path.stem
    if args.output:
        # Se usuário forneceu caminho absoluto ou relativo, usamos nome base e jogamos em results se for só nome
        provided = Path(args.output)
        if provided.is_absolute() and provided.parent != Path('.'):
            output_path = provided  # respeita caminho absoluto
        else:
            output_path = results_dir / provided.name
    else:
        output_path = results_dir / f"{stem}_anon.csv"

    # Load CSV
    df = pd.read_csv(input_path, sep=args.sep, encoding=args.encoding)

    if args.target not in df.columns:
        raise SystemExit(f"Coluna target '{args.target}' não encontrada no dataset {input_path.name}.")

    y = df[args.target]
    # Todas as colunas exceto target são features
    X = df.drop(columns=[args.target])
    if X.shape[1] == 0:
        raise SystemExit("Não há colunas de features (apenas target). Adicione colunas de atributos para anonimizar.")

    # Infer task if needed
    task_type = args.task
    if task_type == "auto":
        task_type = infer_task_type(y)

    # Build anonymizer
    anonymizer = MIADPAnonymizer(
        epsilon=args.epsilon,
        random_state=args.random_state,
        task_type=task_type
    )

    # Fit and transform
    X_anon = anonymizer.fit_transform(X, y)

    # Reattach target unchanged
    out_df = X_anon.copy()
    out_df[args.target] = y.values

    # Save output
    out_df.to_csv(output_path, index=False, sep=args.sep, encoding=args.encoding)

    # Optional report
    if args.report:
        report_df = anonymizer.export_report()
        report_path = Path(args.report)
        if not report_path.is_absolute() or report_path.parent == Path('.'):
            report_path = results_dir / report_path.name
        report_df.to_csv(report_path, index=False, sep=args.sep, encoding=args.encoding)
        print(f"Relatório salvo em: {report_path}")

    print(f"Dataset anonimizado salvo em: {output_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
