"""
Script para generar visualizaciones avanzadas del Frente de Pareto.

Genera:
- Vista multi-ángulo (4 perspectivas) - estática PNG
- Proyecciones 2D con imágenes degradada/mejorada - estática PNG
- Partículas interactivas 3D - HTML
- Superficie triangulada con partículas - HTML
- Codificación por parámetros CLAHE - HTML
"""

import sys
import os
import argparse

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import cv2
from dataclasses import dataclass

# Importar visualizaciones
from src.utils.pareto_visualization import (
    create_multi_angle_pareto,
    create_pareto_with_projections_and_images,
    create_interactive_pareto_particles,
    create_interactive_pareto_triangulated,
    create_pareto_by_clahe_params
)


@dataclass
class Solution:
    """Representa una solución del Frente de Pareto."""
    parameters: np.ndarray
    objectives: np.ndarray
    crowding_distance: float = 0.0


class MockParetoFront:
    """Clase para cargar un Frente de Pareto desde CSV."""
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.solutions = []
        
        for _, row in self.df.iterrows():
            sol = Solution(
                parameters=np.array([row['param_0'], row['param_1'], row['param_2']]),
                objectives=np.array([row['objective_0'], row['objective_1'], row['objective_2']])
            )
            self.solutions.append(sol)
    
    def __iter__(self):
        return iter(self.solutions)
    
    def __len__(self):
        return len(self.solutions)
    
    def get_decision_matrix(self) -> np.ndarray:
        """Retorna matriz de objetivos."""
        return np.array([s.objectives for s in self.solutions])
    
    def get_compromise_solution(self) -> Solution:
        """Retorna solución de compromiso (distancia mínima al punto ideal)."""
        matrix = self.get_decision_matrix()
        
        min_vals = matrix.min(axis=0)
        max_vals = matrix.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        
        normalized = (matrix - min_vals) / ranges
        ideal_point = np.ones(matrix.shape[1])
        distances = np.sqrt(np.sum((normalized - ideal_point) ** 2, axis=1))
        
        best_idx = np.argmin(distances)
        return self.solutions[best_idx]


def main():
    """Genera todas las visualizaciones del Frente de Pareto."""
    
    parser = argparse.ArgumentParser(description='Generar visualizaciones del Frente de Pareto')
    parser.add_argument('--experiment', type=str, default='results/experiment_001',
                        help='Directorio del experimento')
    parser.add_argument('--image-id', type=str, default='103',
                        help='ID de la imagen procesada')
    args = parser.parse_args()
    
    # Rutas
    experiment_dir = args.experiment
    image_id = args.image_id
    csv_path = os.path.join(experiment_dir, f"{image_id}_pareto.csv")
    degraded_path = os.path.join(experiment_dir, f"{image_id}_degraded.png")
    enhanced_path = os.path.join(experiment_dir, f"{image_id}_enhanced.png")
    output_dir = os.path.join(experiment_dir, "visualizaciones")
    
    # Verificar archivos
    if not os.path.exists(csv_path):
        print(f"ERROR: No se encontró {csv_path}")
        print("Primero ejecuta process_single_image.py para generar datos")
        return
    
    print("=" * 65)
    print("GENERACIÓN DE VISUALIZACIONES DEL FRENTE DE PARETO")
    print("=" * 65)
    
    # Cargar Frente de Pareto
    print(f"\nCargando datos desde: {csv_path}")
    pareto = MockParetoFront(csv_path)
    print(f"Soluciones cargadas: {len(pareto)}")
    
    # Cargar imágenes si existen
    degraded_image = None
    enhanced_image = None
    
    if os.path.exists(degraded_path):
        degraded_image = cv2.imread(degraded_path, cv2.IMREAD_GRAYSCALE)
        print(f"Imagen degradada cargada: {degraded_path}")
    else:
        print(f"ADVERTENCIA: No se encontró imagen degradada: {degraded_path}")
    
    if os.path.exists(enhanced_path):
        enhanced_image = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)
        print(f"Imagen mejorada cargada: {enhanced_path}")
    else:
        print(f"ADVERTENCIA: No se encontró imagen mejorada: {enhanced_path}")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # =========================================================================
    # VISUALIZACIONES ESTÁTICAS (matplotlib PNG)
    # =========================================================================
    
    print("\n" + "-" * 65)
    print("VISUALIZACIONES ESTÁTICAS")
    print("-" * 65)
    
    # 1. Vista multi-ángulo
    print("\n[1/5] Generando vista multi-ángulo...")
    create_multi_angle_pareto(
        pareto,
        title=f"Frente de Pareto - Imagen {image_id} (4 Perspectivas)",
        save_path=os.path.join(output_dir, "pareto_multiangulo.png")
    )
    
    # 2. Proyecciones 2D con imágenes
    if degraded_image is not None and enhanced_image is not None:
        print("\n[2/5] Generando proyecciones con imágenes...")
        create_pareto_with_projections_and_images(
            pareto,
            degraded_image=degraded_image,
            enhanced_image=enhanced_image,
            title=f"Frente de Pareto con Proyecciones e Imágenes - Imagen {image_id}",
            save_path=os.path.join(output_dir, "pareto_proyecciones.png")
        )
    else:
        print("\n[2/5] OMITIDO: Faltan imágenes para proyecciones")
    
    # =========================================================================
    # VISUALIZACIONES INTERACTIVAS (Plotly HTML)
    # =========================================================================
    
    print("\n" + "-" * 65)
    print("VISUALIZACIONES INTERACTIVAS")
    print("-" * 65)
    
    # 3. Partículas interactivas (tamaño reducido a la mitad)
    print("\n[3/5] Generando partículas interactivas...")
    create_interactive_pareto_particles(
        pareto,
        title=f"Frente de Pareto Interactivo - Imagen {image_id}",
        save_path=os.path.join(output_dir, "pareto_interactivo.html"),
        color_by='vqi',
        marker_size=5,       # Reducido a la mitad (antes era ~10)
        compromise_size=8    # Reducido a la mitad (antes era ~15)
    )
    
    # 4. Superficie triangulada (tamaño reducido a la mitad)
    print("\n[4/5] Generando superficie triangulada...")
    create_interactive_pareto_triangulated(
        pareto,
        title=f"Superficie Triangulada del Frente de Pareto - Imagen {image_id}",
        save_path=os.path.join(output_dir, "pareto_triangulado.html"),
        surface_opacity=0.6,
        color_by='clip_limit',
        marker_size=5,       # Reducido a la mitad
        compromise_size=8    # Reducido a la mitad
    )
    
    # 5. Por parámetros CLAHE
    print("\n[5/5] Generando visualización por parámetros CLAHE...")
    create_pareto_by_clahe_params(
        pareto,
        title=f"Frente de Pareto por Parámetros CLAHE - Imagen {image_id}",
        save_path=os.path.join(output_dir, "pareto_por_parametros.html"),
        marker_size_range=(3, 10)  # Reducido (antes era 6-20)
    )
    
    # =========================================================================
    # RESUMEN
    # =========================================================================
    
    print("\n" + "=" * 65)
    print("✅ VISUALIZACIONES GENERADAS EXITOSAMENTE")
    print("=" * 65)
    print(f"\nArchivos en: {output_dir}/")
    print("\nEstáticas (PNG):")
    print("  • pareto_multiangulo.png     - 4 perspectivas del Pareto")
    print("  • pareto_proyecciones.png    - 3D + proyecciones 2D + imágenes entrada/salida")
    print("\nInteractivas (HTML):")
    print("  • pareto_interactivo.html    - Partículas con tooltips")
    print("  • pareto_triangulado.html    - Superficie triangulada + partículas")
    print("  • pareto_por_parametros.html - Color=Clip, Tamaño=Área")
    
    # Estadísticas
    compromise = pareto.get_compromise_solution()
    matrix = pareto.get_decision_matrix()
    
    print("\n" + "=" * 65)
    print("ESTADÍSTICAS DEL FRENTE DE PARETO")
    print("=" * 65)
    print(f"Total de soluciones: {len(pareto)}")
    print(f"\nRangos de objetivos:")
    print(f"  Entropía: [{matrix[:, 0].min():.4f}, {matrix[:, 0].max():.4f}]")
    print(f"  SSIM:     [{matrix[:, 1].min():.4f}, {matrix[:, 1].max():.4f}]")
    print(f"  VQI:      [{matrix[:, 2].min():.2f}, {matrix[:, 2].max():.2f}]")
    print(f"\nSolución de compromiso:")
    print(f"  Parámetros CLAHE: Rx={compromise.parameters[0]:.0f}, "
          f"Ry={compromise.parameters[1]:.0f}, Clip={compromise.parameters[2]:.3f}")
    print(f"  Objetivos: H={compromise.objectives[0]:.4f}, "
          f"SSIM={compromise.objectives[1]:.4f}, VQI={compromise.objectives[2]:.2f}")


if __name__ == "__main__":
    main()
