"""
Método difuso de Bellman-Zadeh para decisión multicriterio.

Utiliza teoría de conjuntos difusos para modelar la decisión como
la intersección de objetivos y restricciones difusas.
"""

import numpy as np
from numpy.typing import NDArray
from mcdm.base import MCDMMethod


class BellmanZadeh(MCDMMethod):
    """
    Método de decisión difusa de Bellman-Zadeh.
    
    Este método modela cada criterio como un conjunto difuso y
    la decisión se toma mediante la intersección (operador min)
    de las funciones de pertenencia.
    
    D = G ∩ C = min(μ_G, μ_C)
    
    Examples:
        >>> import numpy as np
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> bellman = BellmanZadeh()
        >>> best_idx, rankings = bellman.select(matrix)
        >>> print(f"Mejor alternativa: {best_idx}")
    """
    
    def __init__(
        self,
        weights=None,
        criteria_types=None,
        aggregation='min'
    ):
        """
        Inicializa el método Bellman-Zadeh.
        
        Args:
            weights: Pesos de criterios (usado para agregación ponderada).
            criteria_types: Tipos de criterios.
            aggregation: Tipo de agregación ('min' o 'weighted').
                        'min': Intersección difusa (operador min)
                        'weighted': Suma ponderada de funciones de pertenencia
        """
        super().__init__(weights, criteria_types)
        self.aggregation = aggregation
    
    def _normalize(self) -> NDArray[np.float64]:
        """
        Normaliza y convierte a funciones de pertenencia [0, 1].
        """
        return self._normalize_max_min()
    
    def _calculate_rankings(self) -> NDArray[np.float64]:
        """
        Calcula el grado de pertenencia de cada alternativa.
        
        Para agregación 'min':
        μ_D(A_i) = min_j(μ_j(A_i))
        
        Para agregación 'weighted':
        μ_D(A_i) = Σ(w_j × μ_j(A_i))
        """
        if self.aggregation == 'min':
            # Intersección difusa: operador min
            membership_degrees = np.min(self.normalized_matrix, axis=1)
        
        elif self.aggregation == 'weighted':
            # Suma ponderada de funciones de pertenencia
            membership_degrees = np.zeros(self.n_alternatives)
            
            for i in range(self.n_alternatives):
                membership_degrees[i] = np.sum(
                    self.weights * self.normalized_matrix[i, :]
                )
        
        else:
            raise ValueError(
                f"Tipo de agregación '{self.aggregation}' no válido. "
                f"Use 'min' o 'weighted'."
            )
        
        return membership_degrees
    
    def _get_best_alternative(self, rankings: NDArray[np.float64]) -> int:
        """Retorna el índice con mayor grado de pertenencia."""
        return int(np.argmax(rankings))
    
    def calculate_alpha_cuts(
        self,
        alpha: float
    ) -> NDArray[np.bool_]:
        """
        Calcula α-cortes del conjunto difuso de decisión.
        
        Un α-corte contiene todas las alternativas con grado de
        pertenencia mayor o igual a α.
        
        Args:
            alpha: Nivel de corte en [0, 1].
        
        Returns:
            Array booleano indicando qué alternativas están en el α-corte.
        
        Examples:
            >>> bellman = BellmanZadeh()
            >>> matrix = np.array([[5, 3], [7, 5], [6, 8]])
            >>> _, rankings = bellman.select(matrix)
            >>> alpha_cut = bellman.calculate_alpha_cuts(0.5)
        """
        if self.normalized_matrix is None:
            raise ValueError("Primero debe ejecutar el método select()")
        
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha debe estar en [0, 1], pero es {alpha}")
        
        rankings = self._calculate_rankings()
        return rankings >= alpha
