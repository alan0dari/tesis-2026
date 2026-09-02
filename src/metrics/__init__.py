"""
Módulo de métricas de evaluación de imágenes.

Métricas de las funciones objetivo del framework:
- Entropía de Shannon (no-reference): cantidad de información.
- SSIM (full-reference): fidelidad estructural respecto a la entrada.
- VIF (full-reference): fidelidad de información visual (Sheikh & Bovik 2006).

`vqi` se conserva solo como referencia histórica de versiones anteriores
del framework; no forma parte de las funciones objetivo actuales.
"""

from src.metrics.entropy import calculate_entropy, calculate_entropy_normalized
from src.metrics.ssim import calculate_ssim
from src.metrics.vif import calculate_vif

__all__ = [
    'calculate_entropy',
    'calculate_entropy_normalized',
    'calculate_ssim',
    'calculate_vif',
]
