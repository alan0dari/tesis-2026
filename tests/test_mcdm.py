"""
Tests para los métodos de decisión multicriterio (MCDM).
"""

import pytest
import numpy as np
from mcdm.smarter import SMARTER
from mcdm.topsis import TOPSIS
from mcdm.bellman_zadeh import BellmanZadeh
from mcdm.promethee_ii import PROMETHEEII
from mcdm.gra import GRA
from mcdm.vikor import VIKOR
from mcdm.codas import CODAS
from mcdm.mabac import MABAC


# Matriz de decisión de prueba
DECISION_MATRIX = np.array([
    [5, 3, 8],
    [7, 5, 6],
    [6, 8, 7],
    [4, 6, 9]
])

WEIGHTS = np.array([0.4, 0.3, 0.3])


class TestSMARTER:
    """Tests para el método SMARTER."""
    
    def test_basic_selection(self):
        """Selección básica debe funcionar."""
        method = SMARTER()
        best_idx, rankings = method.select(DECISION_MATRIX)
        
        assert isinstance(best_idx, (int, np.integer))
        assert 0 <= best_idx < len(DECISION_MATRIX)
        assert len(rankings) == len(DECISION_MATRIX)
    
    def test_with_weights(self):
        """Con pesos especificados."""
        method = SMARTER()
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))
    
    def test_automatic_weights(self):
        """Pesos automáticos ROC."""
        method = SMARTER(use_rank_order_weights=True)
        best_idx, rankings = method.select(DECISION_MATRIX)
        
        assert method.weights is not None
        assert np.isclose(np.sum(method.weights), 1.0)


class TestTOPSIS:
    """Tests para el método TOPSIS."""
    
    def test_basic_selection(self):
        """Selección básica debe funcionar."""
        method = TOPSIS()
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))
        assert len(rankings) == len(DECISION_MATRIX)
        assert np.all(rankings >= 0) and np.all(rankings <= 1)
    
    def test_criteria_types(self):
        """Con tipos de criterios especificados."""
        method = TOPSIS()
        best_idx, rankings = method.select(
            DECISION_MATRIX, 
            weights=WEIGHTS,
            criteria_types=['benefit', 'benefit', 'cost']
        )
        
        assert isinstance(best_idx, (int, np.integer))


class TestBellmanZadeh:
    """Tests para el método Bellman-Zadeh."""
    
    def test_min_aggregation(self):
        """Agregación con operador min."""
        method = BellmanZadeh(aggregation='min')
        best_idx, rankings = method.select(DECISION_MATRIX)
        
        assert isinstance(best_idx, (int, np.integer))
        assert len(rankings) == len(DECISION_MATRIX)
    
    def test_weighted_aggregation(self):
        """Agregación ponderada."""
        method = BellmanZadeh(aggregation='weighted')
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))
    
    def test_invalid_aggregation(self):
        """Debe fallar con agregación inválida."""
        method = BellmanZadeh(aggregation='invalid')
        
        with pytest.raises(ValueError):
            method.select(DECISION_MATRIX)


class TestPROMETHEEII:
    """Tests para el método PROMETHEE II."""
    
    def test_basic_selection(self):
        """Selección básica debe funcionar."""
        method = PROMETHEEII()
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))
        assert len(rankings) == len(DECISION_MATRIX)
    
    def test_different_preference_functions(self):
        """Diferentes funciones de preferencia."""
        for pref_func in ['usual', 'gaussian', 'linear']:
            method = PROMETHEEII(preference_function=pref_func)
            best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
            assert isinstance(best_idx, (int, np.integer))


class TestGRA:
    """Tests para el método GRA."""
    
    def test_basic_selection(self):
        """Selección básica debe funcionar."""
        method = GRA()
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))
        assert len(rankings) == len(DECISION_MATRIX)
    
    def test_zeta_parameter(self):
        """Parámetro zeta diferente."""
        method = GRA(zeta=0.3)
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))


class TestVIKOR:
    """Tests para el método VIKOR."""
    
    def test_basic_selection(self):
        """Selección básica debe funcionar."""
        method = VIKOR()
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))
        assert len(rankings) == len(DECISION_MATRIX)
    
    def test_v_parameter(self):
        """Parámetro v diferente."""
        method = VIKOR(v=0.7)
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))
    
    def test_compromise_solution(self):
        """Solución de compromiso."""
        method = VIKOR()
        method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        compromise = method.get_compromise_solution()
        assert 'best_alternative' in compromise
        assert 'advantage_condition' in compromise


class TestCODAS:
    """Tests para el método CODAS."""
    
    def test_basic_selection(self):
        """Selección básica debe funcionar."""
        method = CODAS()
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))
        assert len(rankings) == len(DECISION_MATRIX)
    
    def test_tau_parameter(self):
        """Parámetro tau diferente."""
        method = CODAS(tau=0.05)
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))


class TestMABAC:
    """Tests para el método MABAC."""
    
    def test_basic_selection(self):
        """Selección básica debe funcionar."""
        method = MABAC()
        best_idx, rankings = method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        assert isinstance(best_idx, (int, np.integer))
        assert len(rankings) == len(DECISION_MATRIX)
    
    def test_border_approximation_area(self):
        """Área de aproximación de borde."""
        method = MABAC()
        method.select(DECISION_MATRIX, weights=WEIGHTS)
        
        baa = method.get_border_approximation_area()
        assert len(baa) == DECISION_MATRIX.shape[1]


def test_all_methods_convergence():
    """Test de que todos los métodos convergen a soluciones válidas."""
    methods = [
        SMARTER(),
        TOPSIS(),
        BellmanZadeh(),
        PROMETHEEII(),
        GRA(),
        VIKOR(),
        CODAS(),
        MABAC()
    ]
    
    selections = []
    
    for method in methods:
        best_idx, _ = method.select(DECISION_MATRIX, weights=WEIGHTS)
        selections.append(best_idx)
    
    # Todas las selecciones deben ser índices válidos
    assert all(0 <= idx < len(DECISION_MATRIX) for idx in selections)
    
    # Debe haber cierto consenso (al menos 2 métodos de acuerdo)
    unique, counts = np.unique(selections, return_counts=True)
    assert np.max(counts) >= 2


def test_consistency_identical_alternatives():
    """Test con alternativas idénticas."""
    # Todas las alternativas son iguales
    matrix = np.ones((4, 3)) * 5.0
    
    method = TOPSIS()
    best_idx, rankings = method.select(matrix, weights=WEIGHTS)
    
    # Los rankings deben ser similares (dentro de tolerancia)
    assert np.allclose(rankings, rankings[0], atol=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
