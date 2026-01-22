"""
Método PROMETHEE II (Preference Ranking Organization Method for Enrichment Evaluations).

PROMETHEE II calcula flujos de preferencia netos para ranking completo de alternativas.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable
from mcdm.base import MCDMMethod


class PROMETHEEII(MCDMMethod):
    """
    Método PROMETHEE II para decisión multicriterio.
    
    PROMETHEE II:
    1. Define funciones de preferencia para cada criterio
    2. Calcula índice de preferencia agregado
    3. Calcula flujos de salida (φ+) y entrada (φ-)
    4. Calcula flujo neto (φ = φ+ - φ-)
    
    Examples:
        >>> import numpy as np
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> promethee = PROMETHEEII()
        >>> best_idx, rankings = promethee.select(matrix)
        >>> print(f"Mejor alternativa: {best_idx}")
    """
    
    def __init__(
        self,
        weights=None,
        criteria_types=None,
        preference_function='gaussian',
        thresholds=None
    ):
        """
        Inicializa PROMETHEE II.
        
        Args:
            weights: Pesos de criterios.
            criteria_types: Tipos de criterios.
            preference_function: Tipo de función de preferencia:
                               'usual', 'u_shape', 'v_shape', 'level', 'linear', 'gaussian'
            thresholds: Umbrales para funciones de preferencia (dict).
        """
        super().__init__(weights, criteria_types)
        self.preference_function = preference_function
        self.thresholds = thresholds or {}
    
    def _normalize(self) -> NDArray[np.float64]:
        """PROMETHEE trabaja con valores originales, no normaliza."""
        # Ajustar criterios de costo a beneficio
        adjusted = self.decision_matrix.copy()
        
        for j in range(self.n_criteria):
            if self.criteria_types[j] == 'cost':
                adjusted[:, j] = -adjusted[:, j]
        
        return adjusted
    
    def _calculate_rankings(self) -> NDArray[np.float64]:
        """
        Calcula flujo neto de preferencia.
        
        φ(a) = φ+(a) - φ-(a)
        """
        # Calcular matriz de preferencias
        preference_matrix = self._calculate_preference_matrix()
        
        # Calcular flujos
        phi_plus = np.mean(preference_matrix, axis=1)  # Flujo positivo
        phi_minus = np.mean(preference_matrix, axis=0)  # Flujo negativo
        
        # Flujo neto
        phi_net = phi_plus - phi_minus
        
        return phi_net
    
    def _calculate_preference_matrix(self) -> NDArray[np.float64]:
        """
        Calcula la matriz de preferencias agregadas π(a,b).
        
        π(a,b) = Σ(w_j × P_j(a,b))
        """
        n = self.n_alternatives
        preference_matrix = np.zeros((n, n))
        
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                
                # Calcular preferencia agregada
                aggregated_pref = 0.0
                
                for j in range(self.n_criteria):
                    diff = self.normalized_matrix[a, j] - self.normalized_matrix[b, j]
                    pref = self._preference_function_value(diff, j)
                    aggregated_pref += self.weights[j] * pref
                
                preference_matrix[a, b] = aggregated_pref
        
        return preference_matrix
    
    def _preference_function_value(
        self,
        diff: float,
        criterion_idx: int
    ) -> float:
        """
        Calcula el valor de la función de preferencia.
        
        Args:
            diff: Diferencia entre dos alternativas en un criterio.
            criterion_idx: Índice del criterio.
        
        Returns:
            Valor de preferencia en [0, 1].
        """
        if self.preference_function == 'usual':
            # Criterio usual: 0 si diff <= 0, 1 si diff > 0
            return 1.0 if diff > 0 else 0.0
        
        elif self.preference_function == 'u_shape':
            # Función en U
            q = self.thresholds.get(f'q_{criterion_idx}', 0.1)
            return 0.0 if abs(diff) <= q else 1.0
        
        elif self.preference_function == 'v_shape':
            # Función en V (lineal)
            p = self.thresholds.get(f'p_{criterion_idx}', 1.0)
            if diff <= 0:
                return 0.0
            elif diff >= p:
                return 1.0
            else:
                return diff / p
        
        elif self.preference_function == 'level':
            # Función de nivel
            q = self.thresholds.get(f'q_{criterion_idx}', 0.1)
            p = self.thresholds.get(f'p_{criterion_idx}', 1.0)
            if diff <= q:
                return 0.0
            elif diff >= p:
                return 1.0
            else:
                return 0.5
        
        elif self.preference_function == 'linear':
            # Función lineal
            q = self.thresholds.get(f'q_{criterion_idx}', 0.1)
            p = self.thresholds.get(f'p_{criterion_idx}', 1.0)
            if diff <= q:
                return 0.0
            elif diff >= p:
                return 1.0
            else:
                return (diff - q) / (p - q)
        
        elif self.preference_function == 'gaussian':
            # Función gaussiana (más suave)
            sigma = self.thresholds.get(f'sigma_{criterion_idx}', 0.5)
            if diff <= 0:
                return 0.0
            else:
                return 1.0 - np.exp(-(diff ** 2) / (2 * sigma ** 2))
        
        else:
            raise ValueError(
                f"Función de preferencia '{self.preference_function}' no válida"
            )
    
    def _get_best_alternative(self, rankings: NDArray[np.float64]) -> int:
        """Retorna el índice con mayor flujo neto."""
        return int(np.argmax(rankings))
    
    def get_partial_ranking(self) -> tuple:
        """
        Obtiene ranking parcial (PROMETHEE I).
        
        Returns:
            Tupla (flujos_positivos, flujos_negativos).
        """
        if self.normalized_matrix is None:
            raise ValueError("Primero debe ejecutar el método select()")
        
        preference_matrix = self._calculate_preference_matrix()
        
        phi_plus = np.mean(preference_matrix, axis=1)
        phi_minus = np.mean(preference_matrix, axis=0)
        
        return phi_plus, phi_minus
