"""
Método GRA (Grey Relational Analysis) para decisión multicriterio.

GRA evalúa la relación entre alternativas basándose en el grado de similitud
de secuencias de datos.
"""

import numpy as np
from numpy.typing import NDArray
from mcdm.base import MCDMMethod


class GRA(MCDMMethod):
    """
    Método de Análisis Relacional Gris (GRA).
    
    GRA:
    1. Normaliza la matriz de decisión
    2. Define la secuencia de referencia ideal
    3. Calcula coeficientes de relación gris
    4. Calcula grado de relación gris agregado
    
    Examples:
        >>> import numpy as np
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> gra = GRA()
        >>> best_idx, rankings = gra.select(matrix)
        >>> print(f"Mejor alternativa: {best_idx}")
    """
    
    def __init__(
        self,
        weights=None,
        criteria_types=None,
        zeta=0.5
    ):
        """
        Inicializa el método GRA.
        
        Args:
            weights: Pesos de criterios.
            criteria_types: Tipos de criterios.
            zeta: Coeficiente de distinción [0, 1]. Típicamente 0.5.
                 Controla el nivel de distinción entre alternativas.
        """
        super().__init__(weights, criteria_types)
        self.zeta = zeta
    
    def _normalize(self) -> NDArray[np.float64]:
        """Normaliza usando Max-Min."""
        return self._normalize_max_min()
    
    def _calculate_rankings(self) -> NDArray[np.float64]:
        """
        Calcula el grado de relación gris.
        
        γ(x_0, x_i) = Σ(w_j × ξ_ij)
        
        donde ξ_ij es el coeficiente de relación gris.
        """
        # Secuencia de referencia ideal (mejores valores en cada criterio)
        reference_sequence = np.max(self.normalized_matrix, axis=0)
        
        # Calcular diferencias absolutas
        delta = np.abs(self.normalized_matrix - reference_sequence)
        
        # Valores min y max de diferencias
        delta_min = np.min(delta)
        delta_max = np.max(delta)
        
        # Calcular coeficientes de relación gris
        # ξ_ij = (Δ_min + ζ·Δ_max) / (Δ_ij + ζ·Δ_max)
        grey_coefficients = np.zeros_like(delta)
        
        for i in range(self.n_alternatives):
            for j in range(self.n_criteria):
                numerator = delta_min + self.zeta * delta_max
                denominator = delta[i, j] + self.zeta * delta_max
                
                if denominator == 0:
                    grey_coefficients[i, j] = 1.0
                else:
                    grey_coefficients[i, j] = numerator / denominator
        
        # Calcular grado de relación gris (suma ponderada)
        grey_relational_grades = np.zeros(self.n_alternatives)
        
        for i in range(self.n_alternatives):
            grey_relational_grades[i] = np.sum(
                self.weights * grey_coefficients[i, :]
            )
        
        return grey_relational_grades
    
    def _get_best_alternative(self, rankings: NDArray[np.float64]) -> int:
        """Retorna el índice con mayor grado de relación gris."""
        return int(np.argmax(rankings))
