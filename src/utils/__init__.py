"""
Módulo de utilidades para procesamiento y visualización.
"""

from src.utils.pareto_visualization import (
    # Estáticas (matplotlib)
    create_multi_angle_pareto,
    create_pareto_with_projections_and_images,
    # Interactivas (Plotly)
    create_interactive_pareto_particles,
    create_interactive_pareto_triangulated,
    create_pareto_by_clahe_params,
    create_complete_pareto_report,
    PLOTLY_AVAILABLE
)

__all__ = [
    'create_multi_angle_pareto',
    'create_pareto_with_projections_and_images',
    'create_interactive_pareto_particles',
    'create_interactive_pareto_triangulated',
    'create_pareto_by_clahe_params',
    'create_complete_pareto_report',
    'PLOTLY_AVAILABLE'
]
