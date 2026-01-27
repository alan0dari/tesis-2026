"""
Método SMARTER (Simple Multi-Attribute Rating Technique using Exploiting Ranks).

SMARTER es una simplificación de SMART que utiliza pesos basados en rankings
y una función de utilidad aditiva.
"""

import numpy as np
from numpy.typing import NDArray
from src.mcdm.base import MCDMMethod


class SMARTER(MCDMMethod):
    """
    Método SMARTER para decisión multicriterio.
    
    SMARTER utiliza:
    - Pesos automáticos basados en rankings de importancia
    - Función de utilidad aditiva simple
    - Normalización Max-Min
    
    Examples:
        >>> import numpy as np
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> smarter = SMARTER()
        >>> best_idx, rankings = smarter.select(matrix)
        >>> print(f"Mejor alternativa: {best_idx}")
    """
    
    def __init__(
        self,
        weights=None,
        criteria_types=None,
        use_rank_order_weights=True
    ):
        """
        Inicializa el método SMARTER.
        
        Args:
            weights: Pesos de criterios. Si None y use_rank_order_weights=True,
                    se calculan automáticamente.
            criteria_types: Tipos de criterios ('benefit' o 'cost').
            use_rank_order_weights: Si True, calcula pesos basados en ranking.
        """
        super().__init__(weights, criteria_types)
        self.use_rank_order_weights = use_rank_order_weights
    
    def _normalize(self) -> NDArray[np.float64]:
        """Normaliza usando método Max-Min."""
        return self._normalize_max_min()
    
    def _calculate_rankings(self) -> NDArray[np.float64]:
        """
        Calcula utilidad usando función aditiva simple.
        
        U(A_i) = Σ(w_j × v_ij)
        
        donde w_j son los pesos y v_ij son los valores normalizados.
        """
        # Si se deben usar pesos por ranking y no se proporcionaron pesos
        if self.use_rank_order_weights and self.weights is None:
            self.weights = self._calculate_rank_order_weights()
        
        # Calcular utilidad como suma ponderada
        utilities = np.zeros(self.n_alternatives)
        
        for i in range(self.n_alternatives):
            utilities[i] = np.sum(
                self.weights * self.normalized_matrix[i, :]
            )
        
        return utilities
    
    def _get_best_alternative(self, rankings: NDArray[np.float64]) -> int:
        """Retorna el índice con mayor utilidad."""
        return int(np.argmax(rankings))
    
    def _calculate_rank_order_weights(self) -> NDArray[np.float64]:
        """
        Calcula pesos basados en ranking usando la fórmula de ROC.
        
        ROC (Rank Order Centroid):
        w_j = (1/n) × Σ(1/k) para k desde j hasta n
        
        Returns:
            Array de pesos normalizados.
        """
        n = self.n_criteria
        weights = np.zeros(n)
        
        for j in range(n):
            weights[j] = (1.0 / n) * np.sum(1.0 / np.arange(j + 1, n + 1))
        
        # Normalizar
        weights = weights / np.sum(weights)
        
        return weights
