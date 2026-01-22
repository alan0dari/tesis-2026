"""
Método CODAS (COmbinative Distance-based ASsessment).

CODAS utiliza distancias Euclidiana y Taxicab para evaluar alternativas
basándose en su desviación de la solución ideal negativa.
"""

import numpy as np
from numpy.typing import NDArray
from mcdm.base import MCDMMethod


class CODAS(MCDMMethod):
    """
    Método CODAS para decisión multicriterio.
    
    CODAS:
    1. Normaliza la matriz de decisión
    2. Calcula solución ideal negativa (NIS)
    3. Calcula distancias Euclidiana y Taxicab desde NIS
    4. Calcula índice de evaluación combinando ambas distancias
    
    Examples:
        >>> import numpy as np
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> codas = CODAS()
        >>> best_idx, rankings = codas.select(matrix)
        >>> print(f"Mejor alternativa: {best_idx}")
    """
    
    def __init__(
        self,
        weights=None,
        criteria_types=None,
        tau=0.02
    ):
        """
        Inicializa el método CODAS.
        
        Args:
            weights: Pesos de criterios.
            criteria_types: Tipos de criterios.
            tau: Umbral para determinar cuándo usar distancia Taxicab.
                Típicamente 0.01 a 0.05.
        """
        super().__init__(weights, criteria_types)
        self.tau = tau
    
    def _normalize(self) -> NDArray[np.float64]:
        """Normaliza usando método lineal."""
        normalized = np.zeros_like(self.decision_matrix)
        
        for j in range(self.n_criteria):
            col = self.decision_matrix[:, j]
            
            if self.criteria_types[j] == 'benefit':
                # Para beneficio: x_ij / max(x_j)
                max_val = np.max(col)
                if max_val != 0:
                    normalized[:, j] = col / max_val
                else:
                    normalized[:, j] = 0
            else:
                # Para costo: min(x_j) / x_ij
                min_val = np.min(col)
                for i in range(self.n_alternatives):
                    if col[i] != 0:
                        normalized[i, j] = min_val / col[i]
                    else:
                        normalized[i, j] = 1.0
        
        return normalized
    
    def _calculate_rankings(self) -> NDArray[np.float64]:
        """
        Calcula el índice de evaluación H.
        
        H_i = E_i + Σ(Ψ_ik × T_i)
        
        donde E_i es distancia Euclidiana y T_i es distancia Taxicab.
        """
        # Matriz ponderada
        weighted_matrix = self.normalized_matrix * self.weights
        
        # Solución ideal negativa (NIS)
        nis = np.min(weighted_matrix, axis=0)
        
        # Calcular distancias Euclidiana y Taxicab desde NIS
        euclidean_distances = np.zeros(self.n_alternatives)
        taxicab_distances = np.zeros(self.n_alternatives)
        
        for i in range(self.n_alternatives):
            diff = weighted_matrix[i, :] - nis
            
            # Distancia Euclidiana (L2)
            euclidean_distances[i] = np.sqrt(np.sum(diff ** 2))
            
            # Distancia Taxicab (L1)
            taxicab_distances[i] = np.sum(np.abs(diff))
        
        # Construir matriz de comparación relativa Ψ
        assessment_scores = euclidean_distances.copy()
        
        for i in range(self.n_alternatives):
            for k in range(self.n_alternatives):
                if i == k:
                    continue
                
                # Calcular diferencia relativa
                diff = euclidean_distances[i] - euclidean_distances[k]
                
                # Si la diferencia es significativa, usar distancia Taxicab
                if abs(diff) >= self.tau:
                    psi_ik = 1 if diff > 0 else -1
                    assessment_scores[i] += psi_ik * taxicab_distances[i]
        
        return assessment_scores
    
    def _get_best_alternative(self, rankings: NDArray[np.float64]) -> int:
        """Retorna el índice con mayor índice de evaluación."""
        return int(np.argmax(rankings))
