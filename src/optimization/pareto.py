"""
Funciones para manejo del Frente de Pareto.

El Frente de Pareto contiene las soluciones óptimas en optimización multiobjetivo,
donde ninguna solución es dominada por otra.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def is_dominated(
    solution1: NDArray[np.float64],
    solution2: NDArray[np.float64],
    minimize: bool = True
) -> bool:
    """
    Verifica si solution1 es dominada por solution2.
    
    En optimización multiobjetivo, una solución A domina a B si:
    - A es al menos tan buena como B en todos los objetivos
    - A es estrictamente mejor que B en al menos un objetivo
    
    Args:
        solution1: Array de valores objetivo de la primera solución.
        solution2: Array de valores objetivo de la segunda solución.
        minimize: Si True, asume minimización. Si False, maximización.
    
    Returns:
        True si solution1 es dominada por solution2.
    
    Examples:
        >>> sol1 = np.array([1.0, 5.0, 3.0])
        >>> sol2 = np.array([0.5, 4.0, 2.0])
        >>> is_dominated(sol1, sol2, minimize=True)
        True
    """
    if minimize:
        # Para minimización: solution2 debe ser <= en todo y < en algo
        better_or_equal_in_all = np.all(solution2 <= solution1)
        better_in_at_least_one = np.any(solution2 < solution1)
    else:
        # Para maximización: solution2 debe ser >= en todo y > en algo
        better_or_equal_in_all = np.all(solution2 >= solution1)
        better_in_at_least_one = np.any(solution2 > solution1)
    
    return better_or_equal_in_all and better_in_at_least_one


def build_pareto_front(
    solutions: List[Dict],
    minimize: bool = True
) -> List[Dict]:
    """
    Construye el Frente de Pareto a partir de un conjunto de soluciones.
    
    Args:
        solutions: Lista de diccionarios con 'position' y 'objectives'.
        minimize: Si True, asume minimización de objetivos.
    
    Returns:
        Lista con las soluciones no dominadas (Frente de Pareto).
    
    Examples:
        >>> solutions = [
        ...     {'position': [1, 2], 'objectives': np.array([1.0, 5.0])},
        ...     {'position': [2, 3], 'objectives': np.array([2.0, 3.0])},
        ...     {'position': [3, 4], 'objectives': np.array([3.0, 1.0])},
        ...     {'position': [1.5, 2.5], 'objectives': np.array([1.5, 4.0])},
        ... ]
        >>> pareto_front = build_pareto_front(solutions)
        >>> print(f"Frente de Pareto: {len(pareto_front)} soluciones")
    """
    if not solutions:
        return []
    
    pareto_front = []
    
    for candidate in solutions:
        is_dominated_by_any = False
        
        for other in solutions:
            if candidate is other:
                continue
            
            if is_dominated(
                candidate['objectives'],
                other['objectives'],
                minimize=minimize
            ):
                is_dominated_by_any = True
                break
        
        if not is_dominated_by_any:
            pareto_front.append(candidate)
    
    return pareto_front


def calculate_hypervolume(
    pareto_front: List[Dict],
    reference_point: NDArray[np.float64]
) -> float:
    """
    Calcula el hipervolumen del Frente de Pareto.
    
    El hipervolumen es una métrica de calidad que mide el volumen del
    espacio objetivo dominado por el Frente de Pareto.
    
    Args:
        pareto_front: Lista de soluciones del Frente de Pareto.
        reference_point: Punto de referencia (típicamente el peor valor posible).
    
    Returns:
        Valor del hipervolumen.
    
    Note:
        Implementación simplificada para 2D y 3D. Para dimensiones mayores,
        considerar usar bibliotecas especializadas.
    
    Examples:
        >>> pareto = [
        ...     {'objectives': np.array([1.0, 5.0])},
        ...     {'objectives': np.array([2.0, 3.0])},
        ...     {'objectives': np.array([3.0, 1.0])},
        ... ]
        >>> ref_point = np.array([10.0, 10.0])
        >>> hv = calculate_hypervolume(pareto, ref_point)
    """
    if not pareto_front:
        return 0.0
    
    n_objectives = len(pareto_front[0]['objectives'])
    
    if n_objectives == 2:
        return _hypervolume_2d(pareto_front, reference_point)
    elif n_objectives == 3:
        return _hypervolume_3d(pareto_front, reference_point)
    else:
        raise NotImplementedError(
            f"Cálculo de hipervolumen no implementado para {n_objectives} objetivos"
        )


def _hypervolume_2d(
    pareto_front: List[Dict],
    reference_point: NDArray[np.float64]
) -> float:
    """Calcula hipervolumen para 2 objetivos."""
    # Ordenar por primer objetivo
    sorted_front = sorted(pareto_front, key=lambda x: x['objectives'][0])
    
    hypervolume = 0.0
    prev_x = 0.0
    
    for solution in sorted_front:
        x, y = solution['objectives']
        
        if x >= reference_point[0] or y >= reference_point[1]:
            continue
        
        width = x - prev_x
        height = reference_point[1] - y
        
        hypervolume += width * height
        prev_x = x
    
    return hypervolume


def _hypervolume_3d(
    pareto_front: List[Dict],
    reference_point: NDArray[np.float64]
) -> float:
    """Calcula hipervolumen aproximado para 3 objetivos."""
    # Implementación simplificada usando suma de cuboides
    hypervolume = 0.0
    
    for solution in pareto_front:
        obj = solution['objectives']
        
        if np.any(obj >= reference_point):
            continue
        
        # Volumen del cuboide formado por la solución y el punto de referencia
        volume = np.prod(reference_point - obj)
        hypervolume += volume
    
    # Nota: Esta es una aproximación que puede contar overlaps
    # Para cálculo exacto, usar algoritmos especializados (WFG, HMS)
    return hypervolume * 0.5  # Factor de corrección aproximado


def calculate_spacing(pareto_front: List[Dict]) -> float:
    """
    Calcula la métrica de espaciado del Frente de Pareto.
    
    El espaciado mide la uniformidad de la distribución de soluciones
    en el Frente de Pareto. Un valor menor indica mejor distribución.
    
    Args:
        pareto_front: Lista de soluciones del Frente de Pareto.
    
    Returns:
        Valor de espaciado. 0 indica distribución perfectamente uniforme.
    
    Examples:
        >>> pareto = [
        ...     {'objectives': np.array([1.0, 5.0])},
        ...     {'objectives': np.array([2.0, 3.0])},
        ...     {'objectives': np.array([3.0, 1.0])},
        ... ]
        >>> spacing = calculate_spacing(pareto)
    """
    if len(pareto_front) < 2:
        return 0.0
    
    # Calcular distancia mínima a vecino más cercano para cada solución
    min_distances = []
    
    for i, sol1 in enumerate(pareto_front):
        min_dist = float('inf')
        
        for j, sol2 in enumerate(pareto_front):
            if i == j:
                continue
            
            dist = np.linalg.norm(sol1['objectives'] - sol2['objectives'])
            min_dist = min(min_dist, dist)
        
        min_distances.append(min_dist)
    
    # Calcular desviación estándar de las distancias mínimas
    mean_dist = np.mean(min_distances)
    variance = np.sum((min_distances - mean_dist) ** 2) / len(min_distances)
    spacing = np.sqrt(variance)
    
    return float(spacing)


def visualize_pareto_front_2d(
    pareto_front: List[Dict],
    all_solutions: Optional[List[Dict]] = None,
    objective_names: Optional[List[str]] = None,
    title: str = "Frente de Pareto 2D",
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza el Frente de Pareto en 2D.
    
    Args:
        pareto_front: Soluciones del Frente de Pareto.
        all_solutions: Todas las soluciones evaluadas (opcional).
        objective_names: Nombres de los objetivos para ejes.
        title: Título de la gráfica.
        save_path: Ruta para guardar la imagen (opcional).
    
    Examples:
        >>> pareto = [{'objectives': np.array([i, 5-i])} for i in range(6)]
        >>> visualize_pareto_front_2d(pareto, objective_names=['F1', 'F2'])
    """
    if not pareto_front:
        print("El Frente de Pareto está vacío")
        return
    
    if objective_names is None:
        objective_names = ['Objetivo 1', 'Objetivo 2']
    
    plt.figure(figsize=(10, 6))
    
    # Graficar todas las soluciones si se proporcionan
    if all_solutions:
        all_obj = np.array([s['objectives'] for s in all_solutions])
        plt.scatter(
            all_obj[:, 0], all_obj[:, 1],
            c='lightgray', alpha=0.5, s=30,
            label='Todas las soluciones'
        )
    
    # Graficar Frente de Pareto
    pareto_obj = np.array([s['objectives'] for s in pareto_front])
    plt.scatter(
        pareto_obj[:, 0], pareto_obj[:, 1],
        c='red', s=100, marker='*',
        label='Frente de Pareto', zorder=5
    )
    
    # Conectar puntos del Frente de Pareto
    sorted_indices = np.argsort(pareto_obj[:, 0])
    plt.plot(
        pareto_obj[sorted_indices, 0],
        pareto_obj[sorted_indices, 1],
        'r--', alpha=0.5, linewidth=2
    )
    
    plt.xlabel(objective_names[0], fontsize=12)
    plt.ylabel(objective_names[1], fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada en: {save_path}")
    
    plt.show()


def visualize_pareto_front_3d(
    pareto_front: List[Dict],
    all_solutions: Optional[List[Dict]] = None,
    objective_names: Optional[List[str]] = None,
    title: str = "Frente de Pareto 3D",
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza el Frente de Pareto en 3D.
    
    Args:
        pareto_front: Soluciones del Frente de Pareto.
        all_solutions: Todas las soluciones evaluadas (opcional).
        objective_names: Nombres de los objetivos para ejes.
        title: Título de la gráfica.
        save_path: Ruta para guardar la imagen (opcional).
    
    Examples:
        >>> pareto = [
        ...     {'objectives': np.array([i, 5-i, i**2])} for i in range(6)
        ... ]
        >>> visualize_pareto_front_3d(
        ...     pareto,
        ...     objective_names=['Entropía', 'SSIM', 'VQI']
        ... )
    """
    if not pareto_front:
        print("El Frente de Pareto está vacío")
        return
    
    if objective_names is None:
        objective_names = ['Objetivo 1', 'Objetivo 2', 'Objetivo 3']
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar todas las soluciones si se proporcionan
    if all_solutions:
        all_obj = np.array([s['objectives'] for s in all_solutions])
        ax.scatter(
            all_obj[:, 0], all_obj[:, 1], all_obj[:, 2],
            c='lightgray', alpha=0.3, s=20,
            label='Todas las soluciones'
        )
    
    # Graficar Frente de Pareto
    pareto_obj = np.array([s['objectives'] for s in pareto_front])
    ax.scatter(
        pareto_obj[:, 0], pareto_obj[:, 1], pareto_obj[:, 2],
        c='red', s=150, marker='*',
        label='Frente de Pareto', zorder=5
    )
    
    ax.set_xlabel(objective_names[0], fontsize=11)
    ax.set_ylabel(objective_names[1], fontsize=11)
    ax.set_zlabel(objective_names[2], fontsize=11)
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    # Rotar vista para mejor visualización
    ax.view_init(elev=20, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfica guardada en: {save_path}")
    
    plt.show()


def export_pareto_front(
    pareto_front: List[Dict],
    filename: str,
    include_positions: bool = True
) -> None:
    """
    Exporta el Frente de Pareto a un archivo CSV.
    
    Args:
        pareto_front: Soluciones del Frente de Pareto.
        filename: Nombre del archivo de salida.
        include_positions: Si True, incluye las posiciones (parámetros).
    
    Examples:
        >>> pareto = [{'position': [1, 2], 'objectives': np.array([1.0, 5.0])}]
        >>> export_pareto_front(pareto, 'pareto_front.csv')
    """
    import pandas as pd
    
    if not pareto_front:
        print("El Frente de Pareto está vacío")
        return
    
    data = []
    
    for i, solution in enumerate(pareto_front):
        row = {'solution_id': i}
        
        if include_positions and 'position' in solution:
            pos = solution['position']
            for j, val in enumerate(pos):
                row[f'param_{j}'] = val
        
        objectives = solution['objectives']
        for j, val in enumerate(objectives):
            row[f'objective_{j}'] = val
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Frente de Pareto exportado a: {filename}")
