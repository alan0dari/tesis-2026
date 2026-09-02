"""
Tests para las métricas de calidad del Frente de Pareto
(hipervolumen exacto y spacing).
"""

import pytest
import numpy as np

import sys
sys.path.insert(0, '.')

from src.optimization.pareto import calculate_hypervolume, calculate_spacing


def _front(*pts):
    return [{'objectives': np.array(p, dtype=float)} for p in pts]


class TestHipervolumen:
    """Casos analíticos verificables por inclusión-exclusión."""

    def test_un_punto_3d(self):
        assert calculate_hypervolume(_front((2, 2, 2)), np.zeros(3)) == pytest.approx(8.0)

    def test_punto_dominado_no_agrega(self):
        hv = calculate_hypervolume(_front((2, 2, 2), (1, 1, 1)), np.zeros(3))
        assert hv == pytest.approx(8.0)

    def test_dos_no_dominados_3d(self):
        # 2*2*1 + 1*1*(2-1) = 5
        hv = calculate_hypervolume(_front((2, 2, 1), (1, 1, 2)), np.zeros(3))
        assert hv == pytest.approx(5.0)

    def test_tres_no_dominados_inclusion_exclusion(self):
        # |A|+|B|+|C| - |AB| - |AC| - |BC| + |ABC| = 3+3+3-1-1-1+1 = 7
        hv = calculate_hypervolume(
            _front((3, 1, 1), (1, 3, 1), (1, 1, 3)), np.zeros(3))
        assert hv == pytest.approx(7.0)

    def test_escalera_2d(self):
        hv = calculate_hypervolume(_front((1, 5), (2, 3), (3, 1)), np.zeros(2))
        assert hv == pytest.approx(3 * 1 + 2 * (3 - 1) + 1 * (5 - 3))

    def test_minimizacion(self):
        hv = calculate_hypervolume(_front((1, 1, 1)), np.array([2.0, 2, 2]),
                                   maximize=False)
        assert hv == pytest.approx(1.0)

    def test_referencia_no_dominada(self):
        # Punto que no domina a la referencia no contribuye
        hv = calculate_hypervolume(_front((1, 1, 1)), np.array([2.0, 0, 0]))
        assert hv == 0.0

    def test_frente_vacio(self):
        assert calculate_hypervolume([], np.zeros(3)) == 0.0

    def test_empates_en_z(self):
        # Dos puntos con la misma z: 2*1*1 + 1*(2-1)*1 = 3
        hv = calculate_hypervolume(_front((2, 1, 1), (1, 2, 1)), np.zeros(3))
        assert hv == pytest.approx(3.0)


class TestSpacing:
    def test_uniforme_es_cero(self):
        s = calculate_spacing(_front((1, 3), (2, 2), (3, 1)))
        assert s == pytest.approx(0.0)

    def test_menos_de_dos_soluciones(self):
        assert calculate_spacing(_front((1, 1))) == 0.0
