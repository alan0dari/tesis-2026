"""
Módulo de métodos de decisión multicriterio (MCDM).

Incluye 8 métodos MCDM para selección de la mejor alternativa
del Frente de Pareto:
- SMARTER: Simple Multi-Attribute Rating Technique using Exploiting Ranks
- TOPSIS: Technique for Order Preference by Similarity to Ideal Solution
- Bellman-Zadeh: Método difuso de intersección
- PROMETHEE II: Preference Ranking Organization Method for Enrichment Evaluations
- GRA: Grey Relational Analysis
- VIKOR: VIseKriterijumska Optimizacija I Kompromisno Resenje
- CODAS: COmbinative Distance-based ASsessment
- MABAC: Multi-Attributive Border Approximation area Comparison
"""

from .base import MCDMMethod
from .smarter import SMARTER
from .topsis import TOPSIS
from .bellman_zadeh import BellmanZadeh
from .promethee_ii import PROMETHEEII
from .gra import GRA
from .vikor import VIKOR
from .codas import CODAS
from .mabac import MABAC

__all__ = [
    'MCDMMethod',
    'SMARTER',
    'TOPSIS',
    'BellmanZadeh',
    'PROMETHEEII',
    'GRA',
    'VIKOR',
    'CODAS',
    'MABAC',
]
