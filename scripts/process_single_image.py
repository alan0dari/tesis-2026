"""
Script para procesar una radiografia con SMPSO-CLAHE.

Este script:
1. Carga una radiografia panoramica
2. Aplica degradacion controlada (simulando problemas de visualizacion)
3. Ejecuta SMPSO para encontrar parametros optimos de CLAHE
4. Genera y guarda el Frente de Pareto 3D
5. Muestra comparacion visual de resultados

Uso:
    python scripts/process_single_image.py --image path/to/image.jpg
    python scripts/process_single_image.py --image path/to/image.jpg --degradation low_contrast
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Agregar directorio raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.smpso import SMPSOImageOptimizer
from src.optimization.pareto import visualize_pareto_front_3d, export_pareto_front
from src.utils.degradation import (
    DegradationType,
    apply_degradation,
    apply_random_degradation,
    create_degradation_set,
    get_image_quality_metrics
)
from src.metrics.entropy import calculate_entropy
from src.metrics.ssim import calculate_ssim
from src.metrics.vqi import calculate_vqi


def load_image(image_path: str) -> np.ndarray:
    """Carga una imagen en escala de grises."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    return image


def create_comparison_figure(
    original: np.ndarray,
    degraded: np.ndarray,
    enhanced: np.ndarray,
    degradation_info: str,
    solution_params: np.ndarray,
    solution_objectives: np.ndarray,
    save_path: str = None
) -> None:
    """Crea figura comparativa con las tres versiones de la imagen."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Imagen original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original', fontsize=14)
    metrics_orig = get_image_quality_metrics(original)
    axes[0].set_xlabel(f"Contraste: {metrics_orig['contrast']:.3f}\n"
                       f"Entropia: {metrics_orig['entropy']:.3f}")
    axes[0].axis('off')
    
    # Imagen degradada
    axes[1].imshow(degraded, cmap='gray')
    axes[1].set_title(f'Degradada\n({degradation_info})', fontsize=14)
    metrics_deg = get_image_quality_metrics(degraded)
    axes[1].set_xlabel(f"Contraste: {metrics_deg['contrast']:.3f}\n"
                       f"Entropia: {metrics_deg['entropy']:.3f}")
    axes[1].axis('off')
    
    # Imagen mejorada con CLAHE
    axes[2].imshow(enhanced, cmap='gray')
    axes[2].set_title(f'Mejorada (CLAHE)\n'
                      f'Rx={solution_params[0]:.0f}, Ry={solution_params[1]:.0f}, '
                      f'Clip={solution_params[2]:.2f}', fontsize=14)
    axes[2].set_xlabel(f"H: {solution_objectives[0]:.4f}\n"
                       f"SSIM: {solution_objectives[1]:.4f}\n"
                       f"VQI: {solution_objectives[2]:.2f}")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
        plt.close(fig)  # Cerrar figura si se guarda
    else:
        plt.show()


def create_pareto_3d_figure(
    pareto_front,
    title: str = "Frente de Pareto 3D",
    save_path: str = None
) -> None:
    """Crea visualizacion 3D del Frente de Pareto."""
    
    objectives = pareto_front.get_decision_matrix()
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot de las soluciones
    scatter = ax.scatter(
        objectives[:, 0],  # Entropia
        objectives[:, 1],  # SSIM
        objectives[:, 2],  # VQI
        c=objectives[:, 2],  # Color por VQI
        cmap='viridis',
        s=100,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Marcar solucion de compromiso
    compromise = pareto_front.get_compromise_solution()
    ax.scatter(
        [compromise.objectives[0]],
        [compromise.objectives[1]],
        [compromise.objectives[2]],
        c='red',
        s=300,
        marker='*',
        label='Solucion de compromiso',
        edgecolors='black',
        linewidth=1
    )
    
    # Etiquetas y titulo
    ax.set_xlabel('Entropia (H)', fontsize=12, labelpad=10)
    ax.set_ylabel('SSIM', fontsize=12, labelpad=10)
    ax.set_zlabel('VQI', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('VQI', fontsize=11)
    
    ax.legend(loc='upper left')
    
    # Mejor angulo de visualizacion
    ax.view_init(elev=25, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafico 3D guardado en: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def create_histograms_comparison(
    original: np.ndarray,
    degraded: np.ndarray,
    enhanced: np.ndarray,
    save_path: str = None
) -> None:
    """Crea comparacion de histogramas."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, img, title in zip(
        axes,
        [original, degraded, enhanced],
        ['Original', 'Degradada', 'Mejorada (CLAHE)']
    ):
        hist, bins = np.histogram(img.flatten(), bins=256, range=(0, 256))
        ax.fill_between(range(256), hist, alpha=0.7)
        ax.set_xlim(0, 255)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Nivel de gris')
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Histogramas guardados en: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def process_image(
    image_path: str,
    degradation_type: str = None,
    n_particles: int = 30,
    max_iterations: int = 50,
    output_dir: str = None,
    seed: int = None
) -> dict:
    """
    Procesa una imagen con el flujo completo SMPSO-CLAHE.
    
    Args:
        image_path: Ruta a la imagen.
        degradation_type: Tipo de degradacion a aplicar (None = aleatorio).
        n_particles: Numero de particulas para SMPSO.
        max_iterations: Numero de iteraciones para SMPSO.
        output_dir: Directorio para guardar resultados.
        seed: Semilla para reproducibilidad.
    
    Returns:
        Diccionario con resultados del procesamiento.
    """
    print("=" * 70)
    print("PROCESAMIENTO DE RADIOGRAFIA CON SMPSO-CLAHE")
    print("=" * 70)
    
    # 1. Cargar imagen
    print(f"\n[1/5] Cargando imagen: {image_path}")
    original = load_image(image_path)
    print(f"      Tamano: {original.shape}")
    print(f"      Rango: [{original.min()}, {original.max()}]")
    
    # 2. Aplicar degradacion
    print(f"\n[2/5] Aplicando degradacion...")
    if degradation_type:
        deg_type = DegradationType(degradation_type)
        # Usar parametros predeterminados moderados
        if deg_type == DegradationType.LOW_CONTRAST:
            degraded = apply_degradation(original, deg_type, factor=0.4)
            deg_params = {'factor': 0.4}
        elif deg_type == DegradationType.UNDEREXPOSURE:
            degraded = apply_degradation(original, deg_type, gamma=2.2, offset=-25)
            deg_params = {'gamma': 2.2, 'offset': -25}
        elif deg_type == DegradationType.OVEREXPOSURE:
            degraded = apply_degradation(original, deg_type, gamma=0.5, saturation_threshold=235)
            deg_params = {'gamma': 0.5, 'saturation_threshold': 235}
        elif deg_type == DegradationType.POOR_LOCAL_CONTRAST:
            degraded = apply_degradation(original, deg_type, blur_kernel=13, contrast_reduction=0.6)
            deg_params = {'blur_kernel': 13, 'contrast_reduction': 0.6}
        elif deg_type == DegradationType.SKEWED_HISTOGRAM:
            degraded = apply_degradation(original, deg_type, skew_direction='dark', intensity=0.7)
            deg_params = {'skew_direction': 'dark', 'intensity': 0.7}
    else:
        degraded, deg_type, deg_params = apply_random_degradation(original, seed=seed)
    
    print(f"      Tipo: {deg_type.value}")
    print(f"      Parametros: {deg_params}")
    
    # Metricas de la imagen degradada
    metrics_degraded = get_image_quality_metrics(degraded)
    print(f"      Contraste resultante: {metrics_degraded['contrast']:.3f}")
    print(f"      Entropia resultante: {metrics_degraded['entropy']:.3f}")
    
    # 3. Ejecutar SMPSO
    print(f"\n[3/5] Ejecutando SMPSO...")
    print(f"      Particulas: {n_particles}")
    print(f"      Iteraciones: {max_iterations}")
    
    optimizer = SMPSOImageOptimizer(
        image=degraded,
        n_particles=n_particles,
        max_iterations=max_iterations,
        verbose=True,
        seed=seed
    )
    
    pareto_front = optimizer.run()
    
    # 4. Obtener mejor solucion
    print(f"\n[4/5] Analizando Frente de Pareto...")
    print(f"      Soluciones encontradas: {len(pareto_front)}")
    
    # Solucion de compromiso
    compromise = pareto_front.get_compromise_solution()
    print(f"\n      Solucion de compromiso:")
    print(f"        Rx={compromise.parameters[0]:.0f}, "
          f"Ry={compromise.parameters[1]:.0f}, "
          f"Clip={compromise.parameters[2]:.2f}")
    print(f"        H={compromise.objectives[0]:.4f}, "
          f"SSIM={compromise.objectives[1]:.4f}, "
          f"VQI={compromise.objectives[2]:.2f}")
    
    # Mejor por cada objetivo
    best_entropy = pareto_front.get_best_by_objective(0)
    best_ssim = pareto_front.get_best_by_objective(1)
    best_vqi = pareto_front.get_best_by_objective(2)
    
    print(f"\n      Mejor por Entropia: H={best_entropy.objectives[0]:.4f}")
    print(f"      Mejor por SSIM: SSIM={best_ssim.objectives[1]:.4f}")
    print(f"      Mejor por VQI: VQI={best_vqi.objectives[2]:.2f}")
    
    # Generar imagen mejorada con solucion de compromiso
    enhanced = optimizer.get_enhanced_image(compromise)
    
    # 5. Guardar resultados
    print(f"\n[5/5] Generando visualizaciones...")
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = Path(image_path).stem
        
        # Guardar imagenes
        cv2.imwrite(str(output_path / f"{image_name}_original.png"), original)
        cv2.imwrite(str(output_path / f"{image_name}_degraded.png"), degraded)
        cv2.imwrite(str(output_path / f"{image_name}_enhanced.png"), enhanced)
        
        # Guardar figura comparativa
        comparison_path = str(output_path / f"{image_name}_comparison.png")
        create_comparison_figure(
            original, degraded, enhanced,
            f"{deg_type.value}",
            compromise.parameters,
            compromise.objectives,
            save_path=comparison_path
        )
        
        # Guardar Pareto 3D
        pareto_path = str(output_path / f"{image_name}_pareto3d.png")
        create_pareto_3d_figure(
            pareto_front,
            title=f"Frente de Pareto 3D - {image_name}",
            save_path=pareto_path
        )
        
        # Guardar histogramas
        hist_path = str(output_path / f"{image_name}_histograms.png")
        create_histograms_comparison(
            original, degraded, enhanced,
            save_path=hist_path
        )
        
        # Exportar datos del Pareto a CSV
        csv_path = str(output_path / f"{image_name}_pareto.csv")
        export_pareto_front(pareto_front.to_list(), csv_path)
        
        # Guardar metadata
        metadata = {
            'image_path': str(image_path),
            'image_size': list(original.shape),
            'degradation_type': deg_type.value,
            'degradation_params': deg_params,
            'smpso_params': {
                'n_particles': n_particles,
                'max_iterations': max_iterations,
                'seed': seed
            },
            'pareto_size': len(pareto_front),
            'compromise_solution': {
                'parameters': compromise.parameters.tolist(),
                'objectives': compromise.objectives.tolist()
            },
            'timestamp': timestamp
        }
        
        with open(output_path / f"{image_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n      Resultados guardados en: {output_path}")
    else:
        # Solo mostrar visualizaciones
        create_comparison_figure(
            original, degraded, enhanced,
            f"{deg_type.value}",
            compromise.parameters,
            compromise.objectives
        )
        create_pareto_3d_figure(pareto_front)
        create_histograms_comparison(original, degraded, enhanced)
    
    print("\n" + "=" * 70)
    print("PROCESAMIENTO COMPLETADO")
    print("=" * 70)
    
    return {
        'original': original,
        'degraded': degraded,
        'enhanced': enhanced,
        'pareto_front': pareto_front,
        'compromise': compromise,
        'degradation_type': deg_type,
        'degradation_params': deg_params
    }


def main():
    parser = argparse.ArgumentParser(
        description='Procesar radiografia con SMPSO-CLAHE'
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Ruta a la imagen de radiografia'
    )
    parser.add_argument(
        '--degradation', '-d',
        type=str,
        choices=['low_contrast', 'underexposure', 'overexposure', 
                 'poor_local_contrast', 'skewed_histogram'],
        default=None,
        help='Tipo de degradacion a aplicar (default: aleatorio)'
    )
    parser.add_argument(
        '--particles', '-p',
        type=int,
        default=30,
        help='Numero de particulas (default: 30)'
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=50,
        help='Numero de iteraciones (default: 50)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Directorio de salida para resultados'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Semilla para reproducibilidad'
    )
    
    args = parser.parse_args()
    
    process_image(
        image_path=args.image,
        degradation_type=args.degradation,
        n_particles=args.particles,
        max_iterations=args.iterations,
        output_dir=args.output,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
