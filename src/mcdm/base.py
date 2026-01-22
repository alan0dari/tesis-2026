"""
Clase base para métodos de decisión multicriterio (MCDM).

Define la interfaz común para todos los métodos MCDM implementados.
"""

from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class MCDMMethod(ABC):
    """
    Clase base abstracta para métodos de decisión multicriterio.
    
    Todos los métodos MCDM heredan de esta clase y deben implementar
    los métodos abstractos definidos.
    
    Attributes:
        decision_matrix: Matriz de decisión (alternativas x criterios).
        weights: Pesos de los criterios (suman 1.0).
        criteria_types: Lista indicando si cada criterio es 'benefit' o 'cost'.
        normalized_matrix: Matriz de decisión normalizada.
    """
    
    def __init__(
        self,
        weights: Optional[NDArray[np.float64]] = None,
        criteria_types: Optional[List[str]] = None
    ):
        """
        Inicializa el método MCDM.
        
        Args:
            weights: Pesos de los criterios. Si None, se asignan pesos iguales.
            criteria_types: Lista con 'benefit' o 'cost' para cada criterio.
                           Si None, todos se asumen 'benefit'.
        """
        self.decision_matrix: Optional[NDArray[np.float64]] = None
        self.weights = weights
        self.criteria_types = criteria_types
        self.normalized_matrix: Optional[NDArray[np.float64]] = None
        self.n_alternatives: int = 0
        self.n_criteria: int = 0
    
    def select(
        self,
        decision_matrix: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]] = None,
        criteria_types: Optional[List[str]] = None
    ) -> Tuple[int, NDArray[np.float64]]:
        """
        Selecciona la mejor alternativa usando el método MCDM.
        
        Args:
            decision_matrix: Matriz de decisión (n_alternatives x n_criteria).
            weights: Pesos de los criterios (opcional).
            criteria_types: Tipos de criterios (opcional).
        
        Returns:
            Tupla (índice_mejor_alternativa, ranking_completo).
        
        Examples:
            >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
            >>> method = ConcreteMethod()
            >>> best_idx, rankings = method.select(matrix)
        """
        self.decision_matrix = decision_matrix.astype(np.float64)
        self.n_alternatives, self.n_criteria = decision_matrix.shape
        
        # Configurar pesos
        if weights is not None:
            self.weights = weights
        if self.weights is None:
            self.weights = np.ones(self.n_criteria) / self.n_criteria
        
        # Normalizar pesos
        self.weights = self.weights / np.sum(self.weights)
        
        # Configurar tipos de criterios
        if criteria_types is not None:
            self.criteria_types = criteria_types
        if self.criteria_types is None:
            self.criteria_types = ['benefit'] * self.n_criteria
        
        # Normalizar matriz de decisión
        self.normalized_matrix = self._normalize()
        
        # Aplicar método específico
        rankings = self._calculate_rankings()
        
        # Obtener índice de la mejor alternativa
        best_idx = self._get_best_alternative(rankings)
        
        return best_idx, rankings
    
    @abstractmethod
    def _normalize(self) -> NDArray[np.float64]:
        """
        Normaliza la matriz de decisión.
        
        Cada método MCDM puede usar su propia técnica de normalización.
        
        Returns:
            Matriz normalizada.
        """
        pass
    
    @abstractmethod
    def _calculate_rankings(self) -> NDArray[np.float64]:
        """
        Calcula los rankings/puntuaciones de las alternativas.
        
        Returns:
            Array con la puntuación de cada alternativa.
        """
        pass
    
    @abstractmethod
    def _get_best_alternative(self, rankings: NDArray[np.float64]) -> int:
        """
        Determina el índice de la mejor alternativa.
        
        Args:
            rankings: Array con puntuaciones de alternativas.
        
        Returns:
            Índice de la mejor alternativa.
        """
        pass
    
    def _normalize_max_min(self) -> NDArray[np.float64]:
        """
        Normalización Max-Min.
        
        Para criterios de beneficio: (x - min) / (max - min)
        Para criterios de costo: (max - x) / (max - min)
        
        Returns:
            Matriz normalizada con valores en [0, 1].
        """
        normalized = np.zeros_like(self.decision_matrix)
        
        for j in range(self.n_criteria):
            col = self.decision_matrix[:, j]
            min_val = np.min(col)
            max_val = np.max(col)
            
            if max_val - min_val == 0:
                normalized[:, j] = 1.0
            else:
                if self.criteria_types[j] == 'benefit':
                    normalized[:, j] = (col - min_val) / (max_val - min_val)
                else:  # cost
                    normalized[:, j] = (max_val - col) / (max_val - min_val)
        
        return normalized
    
    def _normalize_vector(self) -> NDArray[np.float64]:
        """
        Normalización vectorial (norma euclidiana).
        
        x_ij_norm = x_ij / sqrt(sum(x_ij^2))
        
        Returns:
            Matriz normalizada.
        """
        normalized = np.zeros_like(self.decision_matrix)
        
        for j in range(self.n_criteria):
            col = self.decision_matrix[:, j]
            norm = np.linalg.norm(col)
            
            if norm == 0:
                normalized[:, j] = 0
            else:
                normalized[:, j] = col / norm
                
                # Invertir para criterios de costo
                if self.criteria_types[j] == 'cost':
                    normalized[:, j] = 1.0 - normalized[:, j]
        
        return normalized
    
    def _normalize_sum(self) -> NDArray[np.float64]:
        """
        Normalización por suma.
        
        x_ij_norm = x_ij / sum(x_ij)
        
        Returns:
            Matriz normalizada.
        """
        normalized = np.zeros_like(self.decision_matrix)
        
        for j in range(self.n_criteria):
            col = self.decision_matrix[:, j]
            col_sum = np.sum(col)
            
            if col_sum == 0:
                normalized[:, j] = 0
            else:
                normalized[:, j] = col / col_sum
                
                # Invertir para criterios de costo
                if self.criteria_types[j] == 'cost':
                    max_val = np.max(normalized[:, j])
                    normalized[:, j] = max_val - normalized[:, j]
        
        return normalized
    
    def get_ranking_order(self, rankings: NDArray[np.float64]) -> NDArray[np.int32]:
        """
        Obtiene el orden de las alternativas según sus rankings.
        
        Args:
            rankings: Array con puntuaciones de alternativas.
        
        Returns:
            Array con índices ordenados de mejor a peor alternativa.
        """
        # Por defecto, mayor puntuación = mejor
        return np.argsort(rankings)[::-1]
    
    def get_method_name(self) -> str:
        """
        Obtiene el nombre del método MCDM.
        
        Returns:
            Nombre del método.
        """
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        """Representación en string del método."""
        return f"{self.get_method_name()}(n_criteria={self.n_criteria})"
