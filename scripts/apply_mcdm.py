"""
Script para aplicar métodos MCDM al Frente de Pareto.

Este script:
1. Carga el Frente de Pareto desde un archivo CSV
2. Convierte los objetivos a matriz de decisión
3. Aplica los 8 métodos MCDM
4. Genera un reporte comparativo con rankings y consenso

Uso:
    python scripts/apply_mcdm.py --experiment experiment_001 --image-id 103
    python scripts/apply_mcdm.py --pareto-file results/experiment_001/103_pareto.csv
"""

import sys
import argparse
from pathlib import Path

# Agregar el directorio src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from mcdm import (
    SMARTER, TOPSIS, BellmanZadeh, PROMETHEEII,
    GRA, VIKOR, CODAS, MABAC
)


@dataclass
class MCDMResult:
    """Resultado de un método MCDM."""
    method_name: str
    best_index: int
    rankings: np.ndarray
    best_params: np.ndarray
    best_objectives: np.ndarray


def load_pareto_front(pareto_file: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Carga el Frente de Pareto desde un archivo CSV.
    
    Args:
        pareto_file: Ruta al archivo CSV del Pareto.
    
    Returns:
        Tupla con (matriz_decision, parametros_clahe, dataframe_completo).
        - matriz_decision: Array (n_soluciones, 3) con objetivos [H, SSIM, VQI]
        - parametros_clahe: Array (n_soluciones, 3) con params [Rx, Ry, Clip]
        - df: DataFrame completo con todos los datos
    """
    df = pd.read_csv(pareto_file)
    
    # Extraer columnas de objetivos
    objective_cols = [col for col in df.columns if col.startswith('objective_')]
    param_cols = [col for col in df.columns if col.startswith('param_')]
    
    # Matriz de decisión (objetivos)
    decision_matrix = df[objective_cols].values
    
    # Parámetros CLAHE
    params = df[param_cols].values
    
    return decision_matrix, params, df


def apply_all_mcdm_methods(
    decision_matrix: np.ndarray,
    params: np.ndarray,
    weights: Optional[np.ndarray] = None,
    criteria_types: Optional[List[str]] = None
) -> List[MCDMResult]:
    """
    Aplica todos los métodos MCDM a la matriz de decisión.
    
    Args:
        decision_matrix: Matriz de decisión (n_alternativas x n_criterios).
        params: Parámetros CLAHE de cada alternativa.
        weights: Pesos de los criterios. Por defecto [0.4, 0.35, 0.25].
        criteria_types: Tipos de criterios. Por defecto todos 'benefit'.
    
    Returns:
        Lista de MCDMResult con resultados de cada método.
    """
    if weights is None:
        # Pesos por defecto según el dominio del problema
        # Entropía: 0.40, SSIM: 0.35, VQI: 0.25
        weights = np.array([0.40, 0.35, 0.25])
    
    if criteria_types is None:
        # Todos los criterios son de beneficio (maximizar)
        criteria_types = ['benefit', 'benefit', 'benefit']
    
    # Definir los 8 métodos MCDM
    methods = [
        ('SMARTER', SMARTER(weights=weights.copy(), criteria_types=criteria_types.copy())),
        ('TOPSIS', TOPSIS(weights=weights.copy(), criteria_types=criteria_types.copy())),
        ('Bellman-Zadeh', BellmanZadeh(weights=weights.copy(), criteria_types=criteria_types.copy())),
        ('PROMETHEE II', PROMETHEEII(weights=weights.copy(), criteria_types=criteria_types.copy())),
        ('GRA', GRA(weights=weights.copy(), criteria_types=criteria_types.copy())),
        ('VIKOR', VIKOR(weights=weights.copy(), criteria_types=criteria_types.copy())),
        ('CODAS', CODAS(weights=weights.copy(), criteria_types=criteria_types.copy())),
        ('MABAC', MABAC(weights=weights.copy(), criteria_types=criteria_types.copy())),
    ]
    
    results = []
    
    for method_name, method in methods:
        try:
            best_idx, rankings = method.select(decision_matrix.copy())
            
            result = MCDMResult(
                method_name=method_name,
                best_index=best_idx,
                rankings=rankings,
                best_params=params[best_idx],
                best_objectives=decision_matrix[best_idx]
            )
            results.append(result)
            
        except Exception as e:
            print(f"Error al aplicar {method_name}: {e}")
            continue
    
    return results


def calculate_consensus(results: List[MCDMResult]) -> Dict[int, int]:
    """
    Calcula el consenso entre métodos MCDM.
    
    Cuenta cuántas veces cada alternativa fue seleccionada como la mejor.
    
    Args:
        results: Lista de resultados MCDM.
    
    Returns:
        Diccionario {índice_alternativa: frecuencia_selección}.
    """
    consensus = {}
    for result in results:
        idx = result.best_index
        consensus[idx] = consensus.get(idx, 0) + 1
    
    # Ordenar por frecuencia descendente
    consensus = dict(sorted(consensus.items(), key=lambda x: x[1], reverse=True))
    
    return consensus


def calculate_borda_count(results: List[MCDMResult], n_alternatives: int) -> np.ndarray:
    """
    Calcula el conteo de Borda para ranking agregado.
    
    Cada método asigna puntos basados en su ranking:
    - 1er lugar: n-1 puntos
    - 2do lugar: n-2 puntos
    - ...
    - último lugar: 0 puntos
    
    Args:
        results: Lista de resultados MCDM.
        n_alternatives: Número total de alternativas.
    
    Returns:
        Array con puntuación Borda de cada alternativa.
    """
    borda_scores = np.zeros(n_alternatives)
    
    for result in results:
        # Obtener ranking ordenado (índices de mejor a peor)
        # Para métodos que maximizan el score
        sorted_indices = np.argsort(result.rankings)[::-1]
        
        # Asignar puntos Borda
        for rank, idx in enumerate(sorted_indices):
            borda_scores[idx] += (n_alternatives - 1 - rank)
    
    return borda_scores


def generate_report(
    results: List[MCDMResult],
    decision_matrix: np.ndarray,
    params: np.ndarray,
    output_dir: Path
) -> str:
    """
    Genera un reporte detallado de los resultados MCDM.
    
    Args:
        results: Lista de resultados MCDM.
        decision_matrix: Matriz de decisión original.
        params: Parámetros CLAHE.
        output_dir: Directorio para guardar el reporte.
    
    Returns:
        Ruta al archivo de reporte generado.
    """
    n_alternatives = decision_matrix.shape[0]
    
    # Calcular consenso y Borda
    consensus = calculate_consensus(results)
    borda_scores = calculate_borda_count(results, n_alternatives)
    best_borda_idx = int(np.argmax(borda_scores))
    
    # Crear reporte como texto
    lines = []
    lines.append("=" * 80)
    lines.append("REPORTE DE DECISIÓN MULTICRITERIO (MCDM)")
    lines.append("=" * 80)
    lines.append("")
    
    # Información del problema
    lines.append("INFORMACIÓN DEL PROBLEMA")
    lines.append("-" * 40)
    lines.append(f"Número de alternativas: {n_alternatives}")
    lines.append(f"Número de criterios: {decision_matrix.shape[1]}")
    lines.append(f"Criterios: Entropía (H), SSIM, VQI")
    lines.append(f"Pesos: [0.40, 0.35, 0.25]")
    lines.append(f"Tipos: [beneficio, beneficio, beneficio]")
    lines.append("")
    
    # Resultados por método
    lines.append("RESULTADOS POR MÉTODO MCDM")
    lines.append("-" * 40)
    
    for result in results:
        lines.append(f"\n{result.method_name}:")
        lines.append(f"  Mejor alternativa: #{result.best_index}")
        lines.append(f"  Parámetros CLAHE: Rx={result.best_params[0]:.0f}, "
                    f"Ry={result.best_params[1]:.0f}, "
                    f"Clip={result.best_params[2]:.4f}")
        lines.append(f"  Objetivos: H={result.best_objectives[0]:.4f}, "
                    f"SSIM={result.best_objectives[1]:.4f}, "
                    f"VQI={result.best_objectives[2]:.4f}")
    
    # Análisis de consenso
    lines.append("\n" + "=" * 80)
    lines.append("ANÁLISIS DE CONSENSO")
    lines.append("-" * 40)
    
    lines.append("\nFrecuencia de selección por alternativa:")
    for idx, freq in consensus.items():
        pct = (freq / len(results)) * 100
        lines.append(f"  Alternativa #{idx}: {freq}/{len(results)} métodos ({pct:.1f}%)")
        lines.append(f"    Params: Rx={params[idx, 0]:.0f}, "
                    f"Ry={params[idx, 1]:.0f}, Clip={params[idx, 2]:.4f}")
        lines.append(f"    Objs: H={decision_matrix[idx, 0]:.4f}, "
                    f"SSIM={decision_matrix[idx, 1]:.4f}, "
                    f"VQI={decision_matrix[idx, 2]:.4f}")
    
    # Ranking agregado por Borda
    lines.append("\n" + "-" * 40)
    lines.append("RANKING AGREGADO (Método de Borda)")
    lines.append("-" * 40)
    
    # Top 10 por Borda
    top_borda = np.argsort(borda_scores)[::-1][:10]
    lines.append("\nTop 10 alternativas por puntuación Borda:")
    for rank, idx in enumerate(top_borda, 1):
        lines.append(f"  {rank}. Alternativa #{idx} - Puntuación: {borda_scores[idx]:.0f}")
        lines.append(f"     Params: Rx={params[idx, 0]:.0f}, "
                    f"Ry={params[idx, 1]:.0f}, Clip={params[idx, 2]:.4f}")
    
    # Recomendación final
    lines.append("\n" + "=" * 80)
    lines.append("RECOMENDACIÓN FINAL")
    lines.append("=" * 80)
    
    # La alternativa con más consenso
    best_consensus_idx = list(consensus.keys())[0]
    best_consensus_freq = consensus[best_consensus_idx]
    
    if best_consensus_freq >= len(results) // 2:
        # Mayoría de métodos coinciden
        final_idx = best_consensus_idx
        lines.append(f"\nPor CONSENSO MAYORITARIO ({best_consensus_freq}/{len(results)} métodos):")
    else:
        # Usar Borda como desempate
        final_idx = best_borda_idx
        lines.append(f"\nPor RANKING BORDA (sin consenso mayoritario):")
    
    lines.append(f"\n  ★ MEJOR ALTERNATIVA: #{final_idx}")
    lines.append(f"  ═══════════════════════════════════")
    lines.append(f"  Parámetros CLAHE:")
    lines.append(f"    - Rx (filas región): {params[final_idx, 0]:.0f}")
    lines.append(f"    - Ry (cols región): {params[final_idx, 1]:.0f}")
    lines.append(f"    - Clip limit: {params[final_idx, 2]:.4f}")
    lines.append(f"  Métricas de calidad:")
    lines.append(f"    - Entropía: {decision_matrix[final_idx, 0]:.4f} bits")
    lines.append(f"    - SSIM: {decision_matrix[final_idx, 1]:.4f}")
    lines.append(f"    - VQI: {decision_matrix[final_idx, 2]:.4f}")
    lines.append("")
    
    # Guardar reporte
    report_content = "\n".join(lines)
    report_file = output_dir / "mcdm_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(report_content)
    
    return str(report_file)


def save_results_csv(
    results: List[MCDMResult],
    decision_matrix: np.ndarray,
    params: np.ndarray,
    output_dir: Path
) -> str:
    """
    Guarda los resultados en formato CSV para análisis posterior.
    
    Args:
        results: Lista de resultados MCDM.
        decision_matrix: Matriz de decisión.
        params: Parámetros CLAHE.
        output_dir: Directorio de salida.
    
    Returns:
        Ruta al archivo CSV generado.
    """
    n_alternatives = decision_matrix.shape[0]
    
    # Crear DataFrame con rankings de cada método
    data = {
        'solution_id': range(n_alternatives),
        'Rx': params[:, 0],
        'Ry': params[:, 1],
        'Clip': params[:, 2],
        'Entropy': decision_matrix[:, 0],
        'SSIM': decision_matrix[:, 1],
        'VQI': decision_matrix[:, 2],
    }
    
    # Agregar rankings de cada método
    for result in results:
        col_name = f'ranking_{result.method_name.replace(" ", "_").replace("-", "_")}'
        data[col_name] = result.rankings
    
    # Agregar ranking Borda
    borda_scores = calculate_borda_count(results, n_alternatives)
    data['borda_score'] = borda_scores
    
    # Crear DataFrame y guardar
    df = pd.DataFrame(data)
    
    # Agregar columna de ranking global basado en Borda
    df['global_rank'] = df['borda_score'].rank(ascending=False, method='min').astype(int)
    
    # Ordenar por ranking global
    df = df.sort_values('global_rank')
    
    csv_file = output_dir / "mcdm_rankings.csv"
    df.to_csv(csv_file, index=False)
    
    return str(csv_file)


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description='Aplica métodos MCDM al Frente de Pareto'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='experiment_001',
        help='Nombre del directorio del experimento'
    )
    parser.add_argument(
        '--image-id',
        type=str,
        default='103',
        help='ID de la imagen procesada'
    )
    parser.add_argument(
        '--pareto-file',
        type=str,
        help='Ruta directa al archivo CSV del Pareto (override experiment/image-id)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='0.40,0.35,0.25',
        help='Pesos de los criterios separados por coma (H,SSIM,VQI)'
    )
    
    args = parser.parse_args()
    
    # Determinar ruta del archivo Pareto
    project_root = Path(__file__).parent.parent
    
    if args.pareto_file:
        pareto_file = Path(args.pareto_file)
    else:
        pareto_file = project_root / "results" / args.experiment / f"{args.image_id}_pareto.csv"
    
    if not pareto_file.exists():
        print(f"Error: No se encontró el archivo {pareto_file}")
        sys.exit(1)
    
    # Directorio de salida
    output_dir = pareto_file.parent
    
    print(f"Cargando Frente de Pareto desde: {pareto_file}")
    
    # Cargar datos
    decision_matrix, params, df = load_pareto_front(pareto_file)
    
    print(f"Número de soluciones en el Pareto: {decision_matrix.shape[0]}")
    print(f"Número de criterios: {decision_matrix.shape[1]}")
    
    # Parsear pesos
    weights = np.array([float(w) for w in args.weights.split(',')])
    weights = weights / np.sum(weights)  # Normalizar
    
    print(f"Pesos normalizados: {weights}")
    
    # Aplicar métodos MCDM
    print("\nAplicando métodos MCDM...")
    results = apply_all_mcdm_methods(
        decision_matrix=decision_matrix,
        params=params,
        weights=weights,
        criteria_types=['benefit', 'benefit', 'benefit']
    )
    
    print(f"Métodos aplicados exitosamente: {len(results)}/8")
    
    # Generar reportes
    print("\nGenerando reportes...")
    report_file = generate_report(results, decision_matrix, params, output_dir)
    csv_file = save_results_csv(results, decision_matrix, params, output_dir)
    
    print(f"\nArchivos generados:")
    print(f"  - Reporte: {report_file}")
    print(f"  - Rankings CSV: {csv_file}")


if __name__ == '__main__':
    main()
