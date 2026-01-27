"""
Módulo de optimización multiobjetivo.

Este módulo contiene:
- SMPSO: Algoritmo Speed-constrained Multi-objective PSO genérico
- SMPSOImageOptimizer: SMPSO especializado para optimización de imágenes con CLAHE
- ParetoFront: Gestión del Frente de Pareto
- Solution: Representación de una solución
- Particle: Representación de una partícula

Funciones auxiliares:
- is_dominated: Verifica dominancia de Pareto
- build_pareto_front: Construye Frente de Pareto desde lista de soluciones
- visualize_pareto_front_3d: Visualización 3D del Frente de Pareto
- export_pareto_front: Exporta el Frente de Pareto a CSV
"""

from src.optimization.smpso import (
    SMPSO,
    SMPSOImageOptimizer,
    Particle
)

from src.optimization.pareto import (
    ParetoFront,
    Solution,
    is_dominated,
    build_pareto_front,
    calculate_hypervolume,
    calculate_spacing,
    visualize_pareto_front_2d,
    visualize_pareto_front_3d,
    export_pareto_front
)

__all__ = [
    # Clases principales
    'SMPSO',
    'SMPSOImageOptimizer',
    'ParetoFront',
    'Solution',
    'Particle',
    
    # Funciones auxiliares
    'is_dominated',
    'build_pareto_front',
    'calculate_hypervolume',
    'calculate_spacing',
    'visualize_pareto_front_2d',
    'visualize_pareto_front_3d',
    'export_pareto_front'
]
