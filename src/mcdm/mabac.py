"""
Método MABAC (Multi-Attributive Border Approximation area Comparison).

MABAC determina la distancia de cada alternativa al área de aproximación
de borde (BAA) para cada criterio.
"""

import numpy as np
from numpy.typing import NDArray
from mcdm.base import MCDMMethod


class MABAC(MCDMMethod):
    """
    Método MABAC para decisión multicriterio.
    
    MABAC:
    1. Normaliza la matriz de decisión
    2. Calcula matriz de elementos ponderados
    3. Determina el área de aproximación de borde (BAA)
    4. Calcula distancias al BAA
    5. Suma distancias para obtener ranking
    
    Examples:
        >>> import numpy as np
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> mabac = MABAC()
        >>> best_idx, rankings = mabac.select(matrix)
        >>> print(f"Mejor alternativa: {best_idx}")
    """
    
    def _normalize(self) -> NDArray[np.float64]:
        """
        Normaliza usando la fórmula específica de MABAC.
        
        Para beneficio: (x_ij - x_min) / (x_max - x_min)
        Para costo: (x_ij - x_max) / (x_min - x_max)
        """
        normalized = np.zeros_like(self.decision_matrix)
        
        for j in range(self.n_criteria):
            col = self.decision_matrix[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            
            if max_val - min_val == 0:
                normalized[:, j] = 0
            else:
                if self.criteria_types[j] == 'benefit':
                    normalized[:, j] = (col - min_val) / (max_val - min_val)
                else:  # cost
                    normalized[:, j] = (col - max_val) / (min_val - max_val)
        
        return normalized
    
    def _calculate_rankings(self) -> NDArray[np.float64]:
        """
        Calcula las distancias al área de aproximación de borde.
        
        S_i = Σ(Q_ij - G_j)
        
        donde Q_ij es el elemento ponderado y G_j es el BAA del criterio j.
        """
        # Paso 1: Calcular matriz de elementos ponderados
        # Q_ij = w_j × (n_ij + 1)
        weighted_matrix = np.zeros_like(self.normalized_matrix)
        
        for j in range(self.n_criteria):
            weighted_matrix[:, j] = self.weights[j] * (
                self.normalized_matrix[:, j] + 1
            )
        
        # Paso 2: Determinar el área de aproximación de borde (BAA)
        # G_j = (Π Q_ij)^(1/n)  (media geométrica)
        border_approximation = np.zeros(self.n_criteria)
        
        for j in range(self.n_criteria):
            product = np.prod(weighted_matrix[:, j])
            border_approximation[j] = product ** (1.0 / self.n_alternatives)
        
        # Paso 3: Calcular distancias al BAA
        distances = np.zeros(self.n_alternatives)
        
        for i in range(self.n_alternatives):
            for j in range(self.n_criteria):
                distances[i] += (
                    weighted_matrix[i, j] - border_approximation[j]
                )
        
        return distances
    
    def _get_best_alternative(self, rankings: NDArray[np.float64]) -> int:
        """Retorna el índice con mayor suma de distancias al BAA."""
        return int(np.argmax(rankings))
    
    def get_border_approximation_area(self) -> NDArray[np.float64]:
        """
        Obtiene el área de aproximación de borde para cada criterio.
        
        Returns:
            Array con valores de BAA para cada criterio.
        
        Examples:
            >>> mabac = MABAC()
            >>> matrix = np.array([[5, 3], [7, 5], [6, 8]])
            >>> mabac.select(matrix)
            >>> baa = mabac.get_border_approximation_area()
        """
        if self.normalized_matrix is None:
            raise ValueError("Primero debe ejecutar el método select()")
        
        # Calcular matriz de elementos ponderados
        weighted_matrix = np.zeros_like(self.normalized_matrix)
        
        for j in range(self.n_criteria):
            weighted_matrix[:, j] = self.weights[j] * (
                self.normalized_matrix[:, j] + 1
            )
        
        # Calcular BAA
        border_approximation = np.zeros(self.n_criteria)
        
        for j in range(self.n_criteria):
            product = np.prod(weighted_matrix[:, j])
            border_approximation[j] = product ** (1.0 / self.n_alternatives)
        
        return border_approximation
