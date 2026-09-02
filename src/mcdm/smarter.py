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
        use_rank_order_weights: bool = True,
        criteria_rank=None
    ):
        """
        Inicializa el método SMARTER.

        En su definición canónica, SMARTER deriva los pesos de los criterios
        exclusivamente de su orden de importancia mediante la fórmula ROC
        (Rank Order Centroid); los pesos no se proporcionan externamente.

        Args:
            weights: Pesos explícitos. Solo se usan si
                    use_rank_order_weights=False (variante SMART).
            criteria_types: Tipos de criterios ('benefit' o 'cost').
            use_rank_order_weights: Si True (por defecto, comportamiento
                    canónico), los pesos se calculan con ROC ignorando
                    cualquier peso proporcionado.
            criteria_rank: Orden de importancia de los criterios, como lista de
                    índices de columna de mayor a menor importancia. Si None,
                    se asume que las columnas ya están en orden de importancia
                    decreciente.
        """
        super().__init__(weights, criteria_types)
        self.use_rank_order_weights = use_rank_order_weights
        self.criteria_rank = criteria_rank

    def _normalize(self) -> NDArray[np.float64]:
        """Normaliza usando método Max-Min."""
        return self._normalize_max_min()

    def _calculate_rankings(self) -> NDArray[np.float64]:
        """
        Calcula utilidad usando función aditiva simple.

        U(A_i) = Σ(w_j × v_ij)

        donde w_j son los pesos y v_ij son los valores normalizados.
        """
        # SMARTER canónico: los pesos ROC se derivan del orden de importancia
        # y sustituyen a cualquier peso externo. La clase base ya asignó pesos
        # iguales por defecto, de modo que no basta con comprobar `is None`.
        if self.use_rank_order_weights:
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
        Calcula pesos basados en el orden de importancia mediante ROC.

        ROC (Rank Order Centroid), para el criterio en la posición j-ésima
        del orden de importancia (1-indexado):

            w_j = (1/n) × Σ_{k=j}^{n} (1/k)

        Para n = 3 esto produce (0.611, 0.278, 0.111).

        El vector devuelto está en el orden de las columnas de la matriz de
        decisión: si `criteria_rank` indica el orden de importancia, el peso
        mayor se asigna a la columna listada primero.

        Returns:
            Array de pesos normalizados, indexado por columna de criterio.
        """
        n = self.n_criteria

        # Pesos ROC por posición en el ranking (posición 0 = más importante)
        roc = np.array([
            (1.0 / n) * np.sum(1.0 / np.arange(j + 1, n + 1))
            for j in range(n)
        ])
        roc = roc / np.sum(roc)

        # Mapear cada peso a su columna según el orden de importancia
        rank = self.criteria_rank if self.criteria_rank is not None else range(n)
        weights = np.zeros(n)
        for position, criterion_index in enumerate(rank):
            weights[criterion_index] = roc[position]

        return weights
