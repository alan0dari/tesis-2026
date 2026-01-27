"""
Método TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).

TOPSIS selecciona la alternativa más cercana a la solución ideal positiva
y más lejana de la solución ideal negativa.
"""

import numpy as np
from numpy.typing import NDArray
from src.mcdm.base import MCDMMethod


class TOPSIS(MCDMMethod):
    """
    Método TOPSIS para decisión multicriterio.
    
    TOPSIS:
    1. Normaliza la matriz de decisión (vectorial)
    2. Calcula solución ideal positiva (PIS) y negativa (NIS)
    3. Calcula distancias a PIS y NIS
    4. Calcula coeficiente de cercanía relativa
    
    Examples:
        >>> import numpy as np
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> weights = np.array([0.4, 0.3, 0.3])
        >>> topsis = TOPSIS()
        >>> best_idx, rankings = topsis.select(matrix, weights)
        >>> print(f"Mejor alternativa: {best_idx}")
    """
    
    def _normalize(self) -> NDArray[np.float64]:
        """Normaliza usando método vectorial."""
        return self._normalize_vector()
    
    def _calculate_rankings(self) -> NDArray[np.float64]:
        """
        Calcula coeficiente de cercanía relativa.
        
        C_i = D_i^- / (D_i^+ + D_i^-)
        
        donde D_i^+ es distancia a PIS y D_i^- es distancia a NIS.
        """
        # Matriz ponderada
        weighted_matrix = self.normalized_matrix * self.weights
        
        # Solución ideal positiva (PIS) y negativa (NIS)
        pis = np.max(weighted_matrix, axis=0)
        nis = np.min(weighted_matrix, axis=0)
        
        # Calcular distancias euclidianas
        d_positive = np.zeros(self.n_alternatives)
        d_negative = np.zeros(self.n_alternatives)
        
        for i in range(self.n_alternatives):
            d_positive[i] = np.sqrt(
                np.sum((weighted_matrix[i, :] - pis) ** 2)
            )
            d_negative[i] = np.sqrt(
                np.sum((weighted_matrix[i, :] - nis) ** 2)
            )
        
        # Coeficiente de cercanía relativa
        # Evitar división por cero
        closeness = np.zeros(self.n_alternatives)
        for i in range(self.n_alternatives):
            denominator = d_positive[i] + d_negative[i]
            if denominator == 0:
                closeness[i] = 0
            else:
                closeness[i] = d_negative[i] / denominator
        
        return closeness
    
    def _get_best_alternative(self, rankings: NDArray[np.float64]) -> int:
        """Retorna el índice con mayor coeficiente de cercanía."""
        return int(np.argmax(rankings))
