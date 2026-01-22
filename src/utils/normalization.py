"""
Métodos de normalización para MCDM.

Proporciona diferentes técnicas de normalización de matrices de decisión
utilizadas en métodos de decisión multicriterio.
"""

from typing import List, Optional
import numpy as np
from numpy.typing import NDArray


def normalize_max_min(
    matrix: NDArray[np.float64],
    criteria_types: Optional[List[str]] = None
) -> NDArray[np.float64]:
    """
    Normalización Max-Min.
    
    Para criterios de beneficio: (x - min) / (max - min)
    Para criterios de costo: (max - x) / (max - min)
    
    Args:
        matrix: Matriz de decisión (alternativas x criterios).
        criteria_types: Lista con 'benefit' o 'cost' para cada criterio.
    
    Returns:
        Matriz normalizada con valores en [0, 1].
    
    Examples:
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> normalized = normalize_max_min(matrix)
    """
    n_alternatives, n_criteria = matrix.shape
    
    if criteria_types is None:
        criteria_types = ['benefit'] * n_criteria
    
    normalized = np.zeros_like(matrix)
    
    for j in range(n_criteria):
        col = matrix[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        
        if max_val - min_val == 0:
            normalized[:, j] = 1.0
        else:
            if criteria_types[j] == 'benefit':
                normalized[:, j] = (col - min_val) / (max_val - min_val)
            else:  # cost
                normalized[:, j] = (max_val - col) / (max_val - min_val)
    
    return normalized


def normalize_vector(
    matrix: NDArray[np.float64],
    criteria_types: Optional[List[str]] = None
) -> NDArray[np.float64]:
    """
    Normalización vectorial (norma euclidiana).
    
    x_ij_norm = x_ij / sqrt(sum(x_ij^2))
    
    Args:
        matrix: Matriz de decisión.
        criteria_types: Lista con 'benefit' o 'cost' para cada criterio.
    
    Returns:
        Matriz normalizada.
    
    Examples:
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> normalized = normalize_vector(matrix)
    """
    n_alternatives, n_criteria = matrix.shape
    
    if criteria_types is None:
        criteria_types = ['benefit'] * n_criteria
    
    normalized = np.zeros_like(matrix)
    
    for j in range(n_criteria):
        col = matrix[:, j]
        norm = np.linalg.norm(col)
        
        if norm == 0:
            normalized[:, j] = 0
        else:
            normalized[:, j] = col / norm
            
            # Invertir para criterios de costo
            if criteria_types[j] == 'cost':
                max_val = np.max(normalized[:, j])
                normalized[:, j] = max_val - normalized[:, j]
    
    return normalized


def normalize_sum(
    matrix: NDArray[np.float64],
    criteria_types: Optional[List[str]] = None
) -> NDArray[np.float64]:
    """
    Normalización por suma.
    
    x_ij_norm = x_ij / sum(x_ij)
    
    Args:
        matrix: Matriz de decisión.
        criteria_types: Lista con 'benefit' o 'cost' para cada criterio.
    
    Returns:
        Matriz normalizada.
    
    Examples:
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> normalized = normalize_sum(matrix)
    """
    n_alternatives, n_criteria = matrix.shape
    
    if criteria_types is None:
        criteria_types = ['benefit'] * n_criteria
    
    normalized = np.zeros_like(matrix)
    
    for j in range(n_criteria):
        col = matrix[:, j]
        col_sum = np.sum(col)
        
        if col_sum == 0:
            normalized[:, j] = 0
        else:
            normalized[:, j] = col / col_sum
            
            # Invertir para criterios de costo
            if criteria_types[j] == 'cost':
                max_val = np.max(normalized[:, j])
                normalized[:, j] = max_val - normalized[:, j]
    
    return normalized


def normalize_linear(
    matrix: NDArray[np.float64],
    criteria_types: Optional[List[str]] = None
) -> NDArray[np.float64]:
    """
    Normalización lineal.
    
    Para beneficio: x_ij / max(x_j)
    Para costo: min(x_j) / x_ij
    
    Args:
        matrix: Matriz de decisión.
        criteria_types: Lista con 'benefit' o 'cost' para cada criterio.
    
    Returns:
        Matriz normalizada.
    
    Examples:
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> normalized = normalize_linear(matrix)
    """
    n_alternatives, n_criteria = matrix.shape
    
    if criteria_types is None:
        criteria_types = ['benefit'] * n_criteria
    
    normalized = np.zeros_like(matrix)
    
    for j in range(n_criteria):
        col = matrix[:, j]
        
        if criteria_types[j] == 'benefit':
            max_val = np.max(col)
            if max_val != 0:
                normalized[:, j] = col / max_val
            else:
                normalized[:, j] = 0
        else:  # cost
            min_val = np.min(col)
            for i in range(n_alternatives):
                if col[i] != 0:
                    normalized[i, j] = min_val / col[i]
                else:
                    normalized[i, j] = 1.0
    
    return normalized


def normalize_enhanced(
    matrix: NDArray[np.float64],
    criteria_types: Optional[List[str]] = None
) -> NDArray[np.float64]:
    """
    Normalización mejorada considerando valores ideales y anti-ideales.
    
    n_ij = (x_ij - x_j^-) / (x_j^+ - x_j^-)
    
    Args:
        matrix: Matriz de decisión.
        criteria_types: Lista con 'benefit' o 'cost' para cada criterio.
    
    Returns:
        Matriz normalizada con valores en [0, 1].
    
    Examples:
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> normalized = normalize_enhanced(matrix)
    """
    n_alternatives, n_criteria = matrix.shape
    
    if criteria_types is None:
        criteria_types = ['benefit'] * n_criteria
    
    normalized = np.zeros_like(matrix)
    
    for j in range(n_criteria):
        col = matrix[:, j]
        
        if criteria_types[j] == 'benefit':
            ideal = np.max(col)
            anti_ideal = np.min(col)
        else:  # cost
            ideal = np.min(col)
            anti_ideal = np.max(col)
        
        if ideal - anti_ideal == 0:
            normalized[:, j] = 1.0
        else:
            normalized[:, j] = (col - anti_ideal) / (ideal - anti_ideal)
            
            # Invertir para criterios de costo
            if criteria_types[j] == 'cost':
                normalized[:, j] = 1.0 - normalized[:, j]
    
    return normalized


def normalize_z_score(
    matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Normalización por Z-score (estandarización).
    
    z_ij = (x_ij - μ_j) / σ_j
    
    Args:
        matrix: Matriz de decisión.
    
    Returns:
        Matriz normalizada (z-scores).
    
    Examples:
        >>> matrix = np.array([[5, 3, 8], [7, 5, 6], [6, 8, 7]])
        >>> normalized = normalize_z_score(matrix)
    """
    n_alternatives, n_criteria = matrix.shape
    normalized = np.zeros_like(matrix)
    
    for j in range(n_criteria):
        col = matrix[:, j]
        mean = np.mean(col)
        std = np.std(col)
        
        if std == 0:
            normalized[:, j] = 0
        else:
            normalized[:, j] = (col - mean) / std
    
    return normalized


def select_normalization_method(
    method_name: str
) -> callable:
    """
    Selecciona un método de normalización por nombre.
    
    Args:
        method_name: Nombre del método de normalización.
    
    Returns:
        Función de normalización.
    
    Raises:
        ValueError: Si el método no existe.
    
    Examples:
        >>> normalize_func = select_normalization_method('max_min')
        >>> matrix = np.array([[5, 3], [7, 5]])
        >>> normalized = normalize_func(matrix)
    """
    methods = {
        'max_min': normalize_max_min,
        'vector': normalize_vector,
        'sum': normalize_sum,
        'linear': normalize_linear,
        'enhanced': normalize_enhanced,
        'z_score': normalize_z_score
    }
    
    if method_name not in methods:
        raise ValueError(
            f"Método '{method_name}' no válido. "
            f"Métodos disponibles: {list(methods.keys())}"
        )
    
    return methods[method_name]
