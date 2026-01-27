"""
Método VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje).

VIKOR determina un ranking basado en la cercanía a la solución ideal,
enfocándose en soluciones de compromiso.
"""

import numpy as np
from numpy.typing import NDArray
from src.mcdm.base import MCDMMethod


class VIKOR(MCDMMethod):
    """
    Método VIKOR para decisión multicriterio.
    
    VIKOR:
    1. Determina los mejores (f*) y peores (f-) valores para cada criterio
    2. Calcula S (distancia L1) y R (distancia L∞)
    3. Calcula índice Q de compromiso
    4. Ordena alternativas por Q
    
    Examples:
        >>> import numpy as np
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> vikor = VIKOR()
        >>> best_idx, rankings = vikor.select(matrix)
        >>> print(f"Mejor alternativa: {best_idx}")
    """
    
    def __init__(
        self,
        weights=None,
        criteria_types=None,
        v=0.5
    ):
        """
        Inicializa el método VIKOR.
        
        Args:
            weights: Pesos de criterios.
            criteria_types: Tipos de criterios.
            v: Peso de la estrategia de máxima utilidad grupal [0, 1].
               v=0.5 es un compromiso balanceado.
               v>0.5 favorece máxima utilidad grupal.
               v<0.5 favorece mínimo arrepentimiento individual.
        """
        super().__init__(weights, criteria_types)
        self.v = v
    
    def _normalize(self) -> NDArray[np.float64]:
        """VIKOR usa valores originales, se normalizan en el cálculo."""
        # Ajustar criterios de costo
        adjusted = self.decision_matrix.copy()
        
        for j in range(self.n_criteria):
            if self.criteria_types[j] == 'cost':
                # Invertir criterios de costo
                adjusted[:, j] = -adjusted[:, j]
        
        return adjusted
    
    def _calculate_rankings(self) -> NDArray[np.float64]:
        """
        Calcula el índice Q de VIKOR.
        
        Q_i = v × (S_i - S*) / (S- - S*) + (1-v) × (R_i - R*) / (R- - R*)
        """
        # Determinar mejores (f*) y peores (f-) valores
        f_star = np.max(self.normalized_matrix, axis=0)
        f_minus = np.min(self.normalized_matrix, axis=0)
        
        # Calcular S y R para cada alternativa
        S = np.zeros(self.n_alternatives)
        R = np.zeros(self.n_alternatives)
        
        for i in range(self.n_alternatives):
            # S_i: utilidad grupal (norma L1 ponderada)
            # R_i: arrepentimiento individual (norma L∞ ponderada)
            
            for j in range(self.n_criteria):
                if f_star[j] - f_minus[j] == 0:
                    normalized_diff = 0
                else:
                    normalized_diff = self.weights[j] * (
                        (f_star[j] - self.normalized_matrix[i, j]) /
                        (f_star[j] - f_minus[j])
                    )
                
                S[i] += normalized_diff
                R[i] = max(R[i], normalized_diff)
        
        # Determinar S*, S-, R*, R-
        S_star = np.min(S)
        S_minus = np.max(S)
        R_star = np.min(R)
        R_minus = np.max(R)
        
        # Calcular índice Q
        Q = np.zeros(self.n_alternatives)
        
        for i in range(self.n_alternatives):
            # Evitar división por cero
            if S_minus - S_star == 0:
                term1 = 0
            else:
                term1 = self.v * (S[i] - S_star) / (S_minus - S_star)
            
            if R_minus - R_star == 0:
                term2 = 0
            else:
                term2 = (1 - self.v) * (R[i] - R_star) / (R_minus - R_star)
            
            Q[i] = term1 + term2
        
        # VIKOR minimiza Q, así que invertimos para usar argmax
        return -Q
    
    def _get_best_alternative(self, rankings: NDArray[np.float64]) -> int:
        """Retorna el índice con menor Q (mayor -Q)."""
        return int(np.argmax(rankings))
    
    def get_compromise_solution(self) -> dict:
        """
        Obtiene la solución de compromiso con condiciones de aceptabilidad.
        
        Returns:
            Diccionario con información de la solución de compromiso.
        """
        if self.normalized_matrix is None:
            raise ValueError("Primero debe ejecutar el método select()")
        
        Q = -self._calculate_rankings()  # Obtener Q original (positivo)
        
        # Ordenar alternativas por Q
        sorted_indices = np.argsort(Q)
        
        # Condición 1: Ventaja aceptable
        # Q(A2) - Q(A1) >= DQ, donde DQ = 1/(n-1)
        DQ = 1.0 / (self.n_alternatives - 1) if self.n_alternatives > 1 else 0
        
        advantage_condition = (
            len(sorted_indices) < 2 or
            Q[sorted_indices[1]] - Q[sorted_indices[0]] >= DQ
        )
        
        # Condición 2: Estabilidad en la decisión
        # A1 debe ser mejor en S o R
        best_idx = sorted_indices[0]
        
        return {
            'best_alternative': int(best_idx),
            'advantage_condition': advantage_condition,
            'DQ_threshold': DQ,
            'Q_values': Q,
            'ranking': sorted_indices
        }
