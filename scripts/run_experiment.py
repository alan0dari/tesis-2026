"""
Script para ejecutar experimentación con muestra representativa.

Este script realiza el experimento completo del framework:
1. Calcula el tamaño de muestra estadísticamente válido
2. Selecciona imágenes aleatoriamente de forma estratificada
3. Procesa cada imagen con SMPSO-CLAHE
4. Aplica los 8 métodos MCDM
5. Consolida resultados y genera análisis estadístico

Uso:
    # Calcular tamaño de muestra recomendado
    python scripts/run_experiment.py --calculate-sample --population 598
    
    # Ejecutar experimento completo
    python scripts/run_experiment.py --data-dir data/original --sample-size 60
    
    # Ejecutar con configuración personalizada
    python scripts/run_experiment.py --data-dir data/original --sample-size 100 \
        --particles 50 --iterations 75 --workers 4
"""

import sys
import argparse
import random
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Agregar directorio raíz del proyecto al path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats

# Suprimir warnings de matplotlib en procesos paralelos
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# CÁLCULO DE TAMAÑO DE MUESTRA
# ============================================================================

def calculate_sample_size(
    population: int,
    confidence_level: float = 0.95,
    margin_of_error: float = 0.10,
    expected_proportion: float = 0.5
) -> Dict[str, any]:
    """
    Calcula el tamaño de muestra para una población finita.
    
    Fórmula de Cochran con corrección de población finita:
    n = (N * Z² * p * (1-p)) / (e² * (N-1) + Z² * p * (1-p))
    
    Args:
        population: Tamaño de la población (N=598 imágenes).
        confidence_level: Nivel de confianza (0.90, 0.95, 0.99).
        margin_of_error: Margen de error aceptable (0.05=5%, 0.10=10%).
        expected_proportion: Proporción esperada (0.5 = máxima variabilidad).
    
    Returns:
        Diccionario con cálculos y recomendaciones.
    """
    # Valor Z según nivel de confianza
    z_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_values.get(confidence_level, 1.96)
    
    N = population
    p = expected_proportion
    e = margin_of_error
    
    # Fórmula de Cochran con corrección de población finita
    numerator = N * (z ** 2) * p * (1 - p)
    denominator = (e ** 2) * (N - 1) + (z ** 2) * p * (1 - p)
    
    sample_size = int(np.ceil(numerator / denominator))
    
    # Recomendaciones para diferentes escenarios
    scenarios = {}
    for conf in [0.90, 0.95, 0.99]:
        for err in [0.05, 0.10, 0.15]:
            z_sc = z_values[conf]
            num = N * (z_sc ** 2) * p * (1 - p)
            den = (err ** 2) * (N - 1) + (z_sc ** 2) * p * (1 - p)
            n_sc = int(np.ceil(num / den))
            scenarios[f"conf_{int(conf*100)}_err_{int(err*100)}"] = n_sc
    
    return {
        'population': population,
        'confidence_level': confidence_level,
        'margin_of_error': margin_of_error,
        'z_value': z,
        'sample_size': sample_size,
        'percentage_of_population': round(sample_size / population * 100, 2),
        'scenarios': scenarios,
        'formula': 'n = (N × Z² × p × (1-p)) / (e² × (N-1) + Z² × p × (1-p))'
    }


def print_sample_size_recommendations(population: int):
    """Imprime tabla con recomendaciones de tamaño de muestra."""
    print("\n" + "=" * 70)
    print("CÁLCULO DE TAMAÑO DE MUESTRA REPRESENTATIVA")
    print("=" * 70)
    print(f"\nPoblación total: {population} imágenes")
    print(f"Fórmula: Cochran con corrección de población finita")
    print("\nTabla de tamaños de muestra recomendados:")
    print("-" * 70)
    print(f"{'Confianza':<15} {'Error ±5%':<15} {'Error ±10%':<15} {'Error ±15%':<15}")
    print("-" * 70)
    
    for conf in [0.90, 0.95, 0.99]:
        row = f"{int(conf*100)}%".ljust(15)
        for err in [0.05, 0.10, 0.15]:
            result = calculate_sample_size(population, conf, err)
            n = result['sample_size']
            pct = result['percentage_of_population']
            row += f"{n} ({pct}%)".ljust(15)
        print(row)
    
    print("-" * 70)
    
    # Recomendación específica para tesis
    rec = calculate_sample_size(population, 0.95, 0.10)
    print(f"\n★ RECOMENDACIÓN PARA TESIS:")
    print(f"  Confianza: 95%, Error: ±10%")
    print(f"  Tamaño de muestra: {rec['sample_size']} imágenes")
    print(f"  Porcentaje de población: {rec['percentage_of_population']}%")
    
    return rec['sample_size']


# ============================================================================
# SELECCIÓN DE MUESTRA
# ============================================================================

def select_sample_images(
    data_dir: Path,
    sample_size: int,
    extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tif', '.tiff'],
    seed: int = None,
    stratify_by_size: bool = True
) -> List[Path]:
    """
    Selecciona una muestra aleatoria de imágenes.
    
    Args:
        data_dir: Directorio con las imágenes.
        sample_size: Número de imágenes a seleccionar.
        extensions: Extensiones de archivo válidas.
        seed: Semilla para reproducibilidad.
        stratify_by_size: Si True, estratifica por tamaño de archivo.
    
    Returns:
        Lista de rutas a las imágenes seleccionadas.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Encontrar todas las imágenes
    all_images = []
    for ext in extensions:
        all_images.extend(data_dir.glob(f"*{ext}"))
        all_images.extend(data_dir.glob(f"*{ext.upper()}"))
    
    all_images = list(set(all_images))  # Eliminar duplicados
    
    if len(all_images) < sample_size:
        print(f"Advertencia: Solo hay {len(all_images)} imágenes, "
              f"se usarán todas en lugar de {sample_size}")
        return all_images
    
    if stratify_by_size:
        # Estratificar por tamaño de archivo (proxy de resolución)
        sizes = [(img, img.stat().st_size) for img in all_images]
        sizes.sort(key=lambda x: x[1])
        
        # Dividir en terciles
        n = len(sizes)
        tercile_size = n // 3
        
        small = [s[0] for s in sizes[:tercile_size]]
        medium = [s[0] for s in sizes[tercile_size:2*tercile_size]]
        large = [s[0] for s in sizes[2*tercile_size:]]
        
        # Seleccionar proporcionalmente de cada estrato
        n_per_stratum = sample_size // 3
        remainder = sample_size % 3
        
        selected = []
        selected.extend(random.sample(small, min(n_per_stratum, len(small))))
        selected.extend(random.sample(medium, min(n_per_stratum + (1 if remainder > 0 else 0), len(medium))))
        selected.extend(random.sample(large, min(n_per_stratum + (1 if remainder > 1 else 0), len(large))))
        
        # Completar si falta
        while len(selected) < sample_size:
            remaining = [img for img in all_images if img not in selected]
            if remaining:
                selected.append(random.choice(remaining))
            else:
                break
        
        return selected[:sample_size]
    else:
        return random.sample(all_images, sample_size)


# ============================================================================
# PROCESAMIENTO DE IMAGEN (para paralelización)
# ============================================================================

def process_single_image_for_experiment(args: tuple) -> Dict:
    """
    Procesa una imagen individual para el experimento.
    
    Esta función es llamada en paralelo para cada imagen.
    
    Args:
        args: Tupla con (image_path, output_dir, config)
    
    Returns:
        Diccionario con resultados o error.
    """
    image_path, output_dir, config = args
    
    try:
        # Configurar path para imports (necesario para multiprocessing en Windows)
        import sys
        from pathlib import Path
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        import cv2
        from src.optimization.smpso import SMPSOImageOptimizer
        from src.optimization.pareto import export_pareto_front
        from src.utils.degradation import apply_random_degradation, get_image_quality_metrics
        from src.mcdm import (SMARTER, TOPSIS, BellmanZadeh, PROMETHEEII,
                              GRA, VIKOR, CODAS, MABAC)
        
        image_id = Path(image_path).stem
        img_output_dir = Path(output_dir) / image_id
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        # 1. Cargar imagen
        original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if original is None:
            return {'image_id': image_id, 'status': 'error', 'error': 'No se pudo cargar'}
        
        # 2. Aplicar degradación
        degraded, deg_type, deg_params = apply_random_degradation(
            original, seed=config.get('seed')
        )
        
        # 3. Ejecutar SMPSO
        optimizer = SMPSOImageOptimizer(
            image=degraded,
            n_particles=config['particles'],
            max_iterations=config['iterations'],
            verbose=False,
            seed=config.get('seed')
        )
        pareto_front = optimizer.run()
        
        # 4. Obtener solución de compromiso
        compromise = pareto_front.get_compromise_solution()
        enhanced = optimizer.get_enhanced_image(compromise)
        
        # 5. Aplicar métodos MCDM
        decision_matrix = pareto_front.get_decision_matrix()
        params_matrix = np.array([s.parameters for s in pareto_front.to_list()])
        
        weights = np.array(config.get('weights', [0.40, 0.35, 0.25]))
        criteria_types = ['benefit', 'benefit', 'benefit']
        
        mcdm_results = {}
        methods = [
            ('SMARTER', SMARTER), ('TOPSIS', TOPSIS),
            ('BellmanZadeh', BellmanZadeh), ('PROMETHEEII', PROMETHEEII),
            ('GRA', GRA), ('VIKOR', VIKOR),
            ('CODAS', CODAS), ('MABAC', MABAC)
        ]
        
        for name, MethodClass in methods:
            try:
                method = MethodClass(weights=weights.copy(), criteria_types=criteria_types.copy())
                best_idx, rankings = method.select(decision_matrix.copy())
                mcdm_results[name] = {
                    'best_index': int(best_idx),
                    'best_params': params_matrix[best_idx].tolist(),
                    'best_objectives': decision_matrix[best_idx].tolist()
                }
            except Exception as e:
                mcdm_results[name] = {'error': str(e)}
        
        # 6. Calcular consenso
        votes = {}
        for name, result in mcdm_results.items():
            if 'best_index' in result:
                idx = result['best_index']
                votes[idx] = votes.get(idx, 0) + 1
        
        consensus_idx = max(votes, key=votes.get) if votes else compromise.index
        consensus_count = votes.get(consensus_idx, 0)
        
        # 7. Guardar resultados
        cv2.imwrite(str(img_output_dir / "original.png"), original)
        cv2.imwrite(str(img_output_dir / "degraded.png"), degraded)
        cv2.imwrite(str(img_output_dir / "enhanced.png"), enhanced)
        
        # Guardar Pareto
        export_pareto_front(pareto_front.to_list(), str(img_output_dir / "pareto.csv"))
        
        elapsed_time = time.time() - start_time
        
        # Resultado
        result = {
            'image_id': image_id,
            'status': 'success',
            'processing_time': elapsed_time,
            'image_size': list(original.shape),
            'degradation_type': deg_type.value,
            'pareto_size': len(pareto_front),
            'compromise': {
                'params': compromise.parameters.tolist(),
                'objectives': compromise.objectives.tolist()
            },
            'mcdm_results': mcdm_results,
            'consensus': {
                'index': int(consensus_idx),
                'votes': consensus_count,
                'total_methods': len([r for r in mcdm_results.values() if 'best_index' in r])
            },
            'metrics': {
                'original': get_image_quality_metrics(original),
                'degraded': get_image_quality_metrics(degraded),
                'enhanced': get_image_quality_metrics(enhanced)
            }
        }
        
        # Guardar metadata individual
        with open(img_output_dir / "result.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        import traceback
        return {
            'image_id': Path(image_path).stem,
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


# ============================================================================
# EJECUCIÓN DEL EXPERIMENTO
# ============================================================================

def run_experiment(
    data_dir: Path,
    output_dir: Path,
    sample_size: int,
    particles: int = 30,
    iterations: int = 50,
    workers: int = 1,
    seed: int = None,
    weights: List[float] = None
) -> Dict:
    """
    Ejecuta el experimento completo.
    
    Args:
        data_dir: Directorio con las imágenes.
        output_dir: Directorio para resultados.
        sample_size: Número de imágenes a procesar.
        particles: Partículas para SMPSO.
        iterations: Iteraciones para SMPSO.
        workers: Número de procesos paralelos.
        seed: Semilla para reproducibilidad.
        weights: Pesos para MCDM [H, SSIM, VQI].
    
    Returns:
        Diccionario con resumen del experimento.
    """
    experiment_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear directorio del experimento
    experiment_dir = output_dir / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("EXPERIMENTO: FRAMEWORK SMPSO-CLAHE + MCDM")
    print("=" * 70)
    print(f"\nConfiguración:")
    print(f"  Directorio de datos: {data_dir}")
    print(f"  Tamaño de muestra: {sample_size}")
    print(f"  Partículas SMPSO: {particles}")
    print(f"  Iteraciones SMPSO: {iterations}")
    print(f"  Procesos paralelos: {workers}")
    print(f"  Semilla: {seed}")
    print(f"  Pesos MCDM: {weights or [0.40, 0.35, 0.25]}")
    print(f"  Directorio de salida: {experiment_dir}")
    
    # Seleccionar muestra
    print(f"\n[1/4] Seleccionando muestra de {sample_size} imágenes...")
    selected_images = select_sample_images(
        data_dir, sample_size, seed=seed, stratify_by_size=True
    )
    print(f"      Imágenes seleccionadas: {len(selected_images)}")
    
    # Guardar lista de imágenes seleccionadas
    with open(experiment_dir / "sample_images.txt", 'w') as f:
        for img in selected_images:
            f.write(f"{img}\n")
    
    # Configuración para workers
    config = {
        'particles': particles,
        'iterations': iterations,
        'seed': seed,
        'weights': weights or [0.40, 0.35, 0.25]
    }
    
    # Preparar argumentos para procesamiento paralelo
    args_list = [(str(img), str(experiment_dir / "images"), config) 
                 for img in selected_images]
    
    # Procesar imágenes
    print(f"\n[2/4] Procesando {len(selected_images)} imágenes...")
    results = []
    
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_image_for_experiment, args): args[0] 
                      for args in args_list}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                results.append(result)
                status = "✓" if result['status'] == 'success' else "✗"
                print(f"      [{completed}/{len(selected_images)}] {status} {result['image_id']}")
                if result['status'] == 'error':
                    print(f"          Error: {result.get('error', 'Unknown')}")
                    if 'traceback' in result:
                        print(f"          Traceback:\n{result['traceback']}")
    else:
        for i, args in enumerate(args_list, 1):
            result = process_single_image_for_experiment(args)
            results.append(result)
            status = "✓" if result['status'] == 'success' else "✗"
            print(f"      [{i}/{len(selected_images)}] {status} {result['image_id']}")
            if result['status'] == 'error':
                print(f"          Error: {result.get('error', 'Unknown')}")
                if 'traceback' in result:
                    print(f"          Traceback:\n{result['traceback']}")
    
    # Consolidar resultados
    print(f"\n[3/4] Consolidando resultados...")
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    print(f"      Exitosos: {len(successful)}")
    print(f"      Fallidos: {len(failed)}")
    
    # Generar análisis estadístico
    print(f"\n[4/4] Generando análisis estadístico...")
    analysis = generate_statistical_analysis(successful, experiment_dir)
    
    # Guardar resumen del experimento
    experiment_summary = {
        'timestamp': timestamp,
        'config': {
            'data_dir': str(data_dir),
            'sample_size': sample_size,
            'actual_processed': len(successful),
            'particles': particles,
            'iterations': iterations,
            'workers': workers,
            'seed': seed,
            'weights': weights or [0.40, 0.35, 0.25]
        },
        'results': {
            'successful': len(successful),
            'failed': len(failed),
            'failed_images': [r['image_id'] for r in failed]
        },
        'analysis': analysis,
        'total_time_seconds': time.time() - experiment_start
    }
    
    with open(experiment_dir / "experiment_summary.json", 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    # Imprimir resumen
    print("\n" + "=" * 70)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 70)
    print(f"\nTiempo total: {experiment_summary['total_time_seconds']/60:.2f} minutos")
    print(f"Imágenes procesadas: {len(successful)}/{len(selected_images)}")
    print(f"\nResultados guardados en: {experiment_dir}")
    
    return experiment_summary


# ============================================================================
# ANÁLISIS ESTADÍSTICO
# ============================================================================

def generate_statistical_analysis(results: List[Dict], output_dir: Path) -> Dict:
    """
    Genera análisis estadístico de los resultados.
    
    Args:
        results: Lista de resultados exitosos.
        output_dir: Directorio de salida.
    
    Returns:
        Diccionario con análisis estadístico.
    """
    if not results:
        return {'error': 'No hay resultados para analizar'}
    
    # Extraer datos
    data = []
    for r in results:
        row = {
            'image_id': r['image_id'],
            'degradation_type': r['degradation_type'],
            'pareto_size': r['pareto_size'],
            'processing_time': r['processing_time'],
            # Métricas de compromiso
            'compromise_H': r['compromise']['objectives'][0],
            'compromise_SSIM': r['compromise']['objectives'][1],
            'compromise_VQI': r['compromise']['objectives'][2],
            # Parámetros CLAHE
            'compromise_Rx': r['compromise']['params'][0],
            'compromise_Ry': r['compromise']['params'][1],
            'compromise_Clip': r['compromise']['params'][2],
            # Consenso MCDM
            'consensus_votes': r['consensus']['votes'],
            'consensus_total': r['consensus']['total_methods'],
            # Métricas de mejora
            'entropy_improvement': r['metrics']['enhanced']['entropy'] - r['metrics']['degraded']['entropy'],
            'contrast_improvement': r['metrics']['enhanced']['contrast'] - r['metrics']['degraded']['contrast'],
        }
        
        # Agregar selección de cada método MCDM
        for method_name, method_result in r['mcdm_results'].items():
            if 'best_index' in method_result:
                row[f'{method_name}_selection'] = method_result['best_index']
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Guardar datos crudos
    df.to_csv(output_dir / "experiment_data.csv", index=False)
    
    # Estadísticas descriptivas
    stats_summary = {}
    
    # 1. Métricas de las soluciones de compromiso
    metrics_cols = ['compromise_H', 'compromise_SSIM', 'compromise_VQI']
    for col in metrics_cols:
        stats_summary[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median(),
            'ci_95': calculate_confidence_interval(df[col].values)
        }
    
    # 2. Parámetros CLAHE óptimos
    params_cols = ['compromise_Rx', 'compromise_Ry', 'compromise_Clip']
    for col in params_cols:
        stats_summary[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'mode': df[col].mode().values[0] if not df[col].mode().empty else None,
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    # 3. Tamaño del Frente de Pareto
    stats_summary['pareto_size'] = {
        'mean': df['pareto_size'].mean(),
        'std': df['pareto_size'].std(),
        'min': int(df['pareto_size'].min()),
        'max': int(df['pareto_size'].max())
    }
    
    # 4. Tiempo de procesamiento
    stats_summary['processing_time'] = {
        'mean_seconds': df['processing_time'].mean(),
        'std_seconds': df['processing_time'].std(),
        'total_minutes': df['processing_time'].sum() / 60
    }
    
    # 5. Análisis de consenso MCDM
    mcdm_methods = ['SMARTER', 'TOPSIS', 'BellmanZadeh', 'PROMETHEEII', 
                    'GRA', 'VIKOR', 'CODAS', 'MABAC']
    
    # Contar acuerdos entre métodos
    agreement_matrix = np.zeros((len(mcdm_methods), len(mcdm_methods)))
    for _, row in df.iterrows():
        selections = [row.get(f'{m}_selection') for m in mcdm_methods]
        for i, sel_i in enumerate(selections):
            for j, sel_j in enumerate(selections):
                if sel_i is not None and sel_j is not None and sel_i == sel_j:
                    agreement_matrix[i, j] += 1
    
    agreement_matrix = agreement_matrix / len(df) * 100  # Porcentaje
    
    agreement_df = pd.DataFrame(
        agreement_matrix,
        index=mcdm_methods,
        columns=mcdm_methods
    )
    agreement_df.to_csv(output_dir / "mcdm_agreement_matrix.csv")
    
    stats_summary['mcdm_agreement'] = {
        'mean_agreement': np.mean(agreement_matrix[np.triu_indices(len(mcdm_methods), k=1)]),
        'min_agreement': np.min(agreement_matrix[np.triu_indices(len(mcdm_methods), k=1)]),
        'max_agreement': np.max(agreement_matrix[np.triu_indices(len(mcdm_methods), k=1)])
    }
    
    # 6. Análisis por tipo de degradación
    degradation_analysis = df.groupby('degradation_type').agg({
        'compromise_H': ['mean', 'std'],
        'compromise_SSIM': ['mean', 'std'],
        'compromise_VQI': ['mean', 'std'],
        'entropy_improvement': ['mean', 'std'],
        'contrast_improvement': ['mean', 'std']
    }).round(4)
    
    degradation_analysis.to_csv(output_dir / "analysis_by_degradation.csv")
    stats_summary['by_degradation'] = degradation_analysis.to_dict()
    
    # 7. Mejora promedio
    stats_summary['improvement'] = {
        'entropy_mean': df['entropy_improvement'].mean(),
        'entropy_std': df['entropy_improvement'].std(),
        'contrast_mean': df['contrast_improvement'].mean(),
        'contrast_std': df['contrast_improvement'].std()
    }
    
    # Generar reporte en texto
    generate_text_report(stats_summary, df, output_dir)
    
    return stats_summary


def calculate_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Calcula intervalo de confianza para la media."""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - h, mean + h)


def generate_text_report(stats: Dict, df: pd.DataFrame, output_dir: Path):
    """Genera reporte de texto con resultados para la tesis."""
    lines = []
    lines.append("=" * 80)
    lines.append("REPORTE DE RESULTADOS DEL EXPERIMENTO")
    lines.append("Framework SMPSO-CLAHE con MCDM para mejora de radiografías dentales")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("1. RESUMEN GENERAL")
    lines.append("-" * 40)
    lines.append(f"   Imágenes procesadas: {len(df)}")
    lines.append(f"   Tiempo total: {stats['processing_time']['total_minutes']:.2f} minutos")
    lines.append(f"   Tiempo promedio por imagen: {stats['processing_time']['mean_seconds']:.2f} ± "
                f"{stats['processing_time']['std_seconds']:.2f} segundos")
    lines.append("")
    
    lines.append("2. MÉTRICAS DE CALIDAD (Solución de Compromiso)")
    lines.append("-" * 40)
    for metric, label in [('compromise_H', 'Entropía (H)'), 
                          ('compromise_SSIM', 'SSIM'), 
                          ('compromise_VQI', 'VQI')]:
        s = stats[metric]
        ci = s['ci_95']
        lines.append(f"   {label}:")
        lines.append(f"      Media ± DE: {s['mean']:.4f} ± {s['std']:.4f}")
        lines.append(f"      IC 95%: [{ci[0]:.4f}, {ci[1]:.4f}]")
        lines.append(f"      Rango: [{s['min']:.4f}, {s['max']:.4f}]")
    lines.append("")
    
    lines.append("3. PARÁMETROS CLAHE ÓPTIMOS")
    lines.append("-" * 40)
    for param, label in [('compromise_Rx', 'Rx (filas)'), 
                         ('compromise_Ry', 'Ry (columnas)'), 
                         ('compromise_Clip', 'Clip limit')]:
        s = stats[param]
        lines.append(f"   {label}: {s['mean']:.2f} ± {s['std']:.2f} (moda: {s['mode']})")
    lines.append("")
    
    lines.append("4. TAMAÑO DEL FRENTE DE PARETO")
    lines.append("-" * 40)
    s = stats['pareto_size']
    lines.append(f"   Promedio: {s['mean']:.1f} ± {s['std']:.1f} soluciones")
    lines.append(f"   Rango: [{s['min']}, {s['max']}]")
    lines.append("")
    
    lines.append("5. MEJORA DE CALIDAD")
    lines.append("-" * 40)
    s = stats['improvement']
    lines.append(f"   Mejora en entropía: {s['entropy_mean']:.4f} ± {s['entropy_std']:.4f}")
    lines.append(f"   Mejora en contraste: {s['contrast_mean']:.4f} ± {s['contrast_std']:.4f}")
    lines.append("")
    
    lines.append("6. CONSENSO ENTRE MÉTODOS MCDM")
    lines.append("-" * 40)
    s = stats['mcdm_agreement']
    lines.append(f"   Acuerdo promedio: {s['mean_agreement']:.1f}%")
    lines.append(f"   Rango de acuerdo: [{s['min_agreement']:.1f}%, {s['max_agreement']:.1f}%]")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("Archivos generados:")
    lines.append("  - experiment_data.csv: Datos completos del experimento")
    lines.append("  - mcdm_agreement_matrix.csv: Matriz de acuerdo entre métodos")
    lines.append("  - analysis_by_degradation.csv: Análisis por tipo de degradación")
    lines.append("=" * 80)
    
    with open(output_dir / "experiment_report.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    print("\n".join(lines))


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ejecutar experimento con muestra representativa',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Ver recomendaciones de tamaño de muestra
  python scripts/run_experiment.py --calculate-sample --population 598
  
  # Ejecutar experimento con muestra de 60 imágenes
  python scripts/run_experiment.py --data-dir data/original --sample-size 60
  
  # Ejecutar con configuración personalizada
  python scripts/run_experiment.py --data-dir data/original --sample-size 100 \\
      --particles 50 --iterations 75 --workers 4 --seed 42
        """
    )
    
    parser.add_argument(
        '--calculate-sample',
        action='store_true',
        help='Solo calcular tamaño de muestra recomendado'
    )
    parser.add_argument(
        '--population',
        type=int,
        default=598,
        help='Tamaño de la población total (default: 598)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Directorio con las imágenes originales'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directorio base para resultados (default: results)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Número de imágenes a procesar'
    )
    parser.add_argument(
        '--particles',
        type=int,
        default=30,
        help='Número de partículas SMPSO (default: 30)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=50,
        help='Número de iteraciones SMPSO (default: 50)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Procesos paralelos (default: 1, usar más si hay RAM suficiente)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad (default: 42)'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='0.40,0.35,0.25',
        help='Pesos MCDM separados por coma (default: 0.40,0.35,0.25)'
    )
    
    args = parser.parse_args()
    
    # Solo calcular tamaño de muestra
    if args.calculate_sample:
        recommended = print_sample_size_recommendations(args.population)
        return
    
    # Validar argumentos para experimento
    if not args.data_dir:
        parser.error("Se requiere --data-dir para ejecutar el experimento")
    
    if not args.sample_size:
        # Calcular tamaño recomendado
        result = calculate_sample_size(args.population, 0.95, 0.10)
        args.sample_size = result['sample_size']
        print(f"Usando tamaño de muestra recomendado: {args.sample_size}")
    
    # Parsear pesos
    weights = [float(w) for w in args.weights.split(',')]
    
    # Ejecutar experimento
    run_experiment(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        sample_size=args.sample_size,
        particles=args.particles,
        iterations=args.iterations,
        workers=args.workers,
        seed=args.seed,
        weights=weights
    )


if __name__ == '__main__':
    main()
