"""
Funciones para manejo del Frente de Pareto.

El Frente de Pareto contiene las soluciones óptimas en optimización multiobjetivo,
donde ninguna solución es dominada por otra.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class Solution:
    """
    Representa una solución en el espacio de optimización.
    
    Attributes:
        parameters: Parámetros de CLAHE [R_x, R_y, clip_limit].
        objectives: Valores de las funciones objetivo [entropía, SSIM, VQI].
        crowding_distance: Distancia de apiñamiento para diversidad.
    """
    parameters: NDArray[np.float64]
    objectives: NDArray[np.float64]
    crowding_distance: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convierte la solución a diccionario."""
        return {
            'position': self.parameters.copy(),
            'objectives': self.objectives.copy(),
            'crowding_distance': self.crowding_distance
        }


class ParetoFront:
    """
    Clase para gestionar el Frente de Pareto en optimización multiobjetivo.
    
    El Frente de Pareto contiene soluciones no dominadas, donde ninguna
    solución es mejor que otra en todos los objetivos simultáneamente.
    
    Attributes:
        solutions: Lista de soluciones no dominadas.
        max_size: Tamaño máximo del frente (para podar por crowding distance).
        maximize: Si True, maximiza los objetivos; si False, minimiza.
    
    Examples:
        >>> front = ParetoFront(max_size=100, maximize=True)
        >>> solution = Solution(
        ...     parameters=np.array([8, 8, 2.0]),
        ...     objectives=np.array([7.5, 0.95, 85.0])
        ... )
        >>> front.add(solution)
        >>> print(f"Soluciones en el frente: {len(front)}")
    """
    
    def __init__(self, max_size: int = 100, maximize: bool = True):
        """
        Inicializa el Frente de Pareto.
        
        Args:
            max_size: Tamaño máximo del archivo externo.
            maximize: Si True, asume maximización de objetivos.
        """
        self.solutions: List[Solution] = []
        self.max_size = max_size
        self.maximize = maximize
    
    def __len__(self) -> int:
        """Retorna el número de soluciones en el frente."""
        return len(self.solutions)
    
    def __iter__(self):
        """Permite iterar sobre las soluciones."""
        return iter(self.solutions)
    
    def __getitem__(self, index: int) -> Solution:
        """Permite acceso por índice."""
        return self.solutions[index]
    
    def add(self, solution: Solution) -> bool:
        """
        Agrega una solución al frente si no es dominada.
        
        Args:
            solution: Solución a agregar.
        
        Returns:
            True si la solución fue agregada, False si fue dominada.
        """
        # Verificar si la solución es dominada por alguna existente
        dominated_indices = []
        
        for i, existing in enumerate(self.solutions):
            if self._dominates(existing.objectives, solution.objectives):
                return False  # La nueva solución es dominada
            
            if self._dominates(solution.objectives, existing.objectives):
                dominated_indices.append(i)
        
        # Eliminar soluciones dominadas por la nueva
        for i in reversed(dominated_indices):
            self.solutions.pop(i)
        
        # Agregar la nueva solución
        self.solutions.append(solution)
        
        # Podar si excede el tamaño máximo
        if len(self.solutions) > self.max_size:
            self._truncate()
        
        return True
    
    def _dominates(
        self,
        objectives1: NDArray[np.float64],
        objectives2: NDArray[np.float64]
    ) -> bool:
        """
        Verifica si objectives1 domina a objectives2.
        
        Args:
            objectives1: Valores objetivo de la primera solución.
            objectives2: Valores objetivo de la segunda solución.
        
        Returns:
            True si objectives1 domina a objectives2.
        """
        if self.maximize:
            # Para maximización: objectives1 >= objectives2 en todo y > en algo
            better_or_equal = np.all(objectives1 >= objectives2)
            strictly_better = np.any(objectives1 > objectives2)
        else:
            # Para minimización: objectives1 <= objectives2 en todo y < en algo
            better_or_equal = np.all(objectives1 <= objectives2)
            strictly_better = np.any(objectives1 < objectives2)
        
        return better_or_equal and strictly_better
    
    def update_crowding_distances(self) -> None:
        """
        Calcula la distancia de apiñamiento para cada solución.
        
        La distancia de apiñamiento mide qué tan aislada está una solución
        respecto a sus vecinos, favoreciendo la diversidad.
        """
        n = len(self.solutions)
        if n == 0:
            return
        
        n_objectives = len(self.solutions[0].objectives)
        
        # Reiniciar distancias
        for sol in self.solutions:
            sol.crowding_distance = 0.0
        
        for obj_idx in range(n_objectives):
            # Ordenar por este objetivo
            sorted_solutions = sorted(
                self.solutions,
                key=lambda s: s.objectives[obj_idx]
            )
            
            # Asignar distancia infinita a los extremos
            sorted_solutions[0].crowding_distance = float('inf')
            sorted_solutions[-1].crowding_distance = float('inf')
            
            # Calcular rango del objetivo
            obj_min = sorted_solutions[0].objectives[obj_idx]
            obj_max = sorted_solutions[-1].objectives[obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calcular distancia para soluciones intermedias
            for i in range(1, n - 1):
                distance = (
                    sorted_solutions[i + 1].objectives[obj_idx] -
                    sorted_solutions[i - 1].objectives[obj_idx]
                ) / obj_range
                sorted_solutions[i].crowding_distance += distance
    
    def _truncate(self) -> None:
        """Reduce el tamaño del frente usando crowding distance."""
        self.update_crowding_distances()
        
        # Ordenar por crowding distance (descendente)
        self.solutions.sort(key=lambda s: s.crowding_distance, reverse=True)
        
        # Mantener las mejores (más diversas)
        self.solutions = self.solutions[:self.max_size]
    
    def select_leader(self) -> Solution:
        """
        Selecciona un líder mediante torneo binario basado en crowding distance.
        
        Returns:
            Solución seleccionada como líder.
        """
        if len(self.solutions) == 0:
            raise ValueError("El frente de Pareto está vacío")
        
        if len(self.solutions) == 1:
            return self.solutions[0]
        
        # Actualizar crowding distances
        self.update_crowding_distances()
        
        # Torneo binario
        idx1, idx2 = np.random.choice(len(self.solutions), 2, replace=False)
        sol1, sol2 = self.solutions[idx1], self.solutions[idx2]
        
        # Seleccionar el más aislado (mayor crowding distance)
        return sol1 if sol1.crowding_distance >= sol2.crowding_distance else sol2
    
    def get_decision_matrix(self) -> NDArray[np.float64]:
        """
        Retorna la matriz de decisión para métodos MCDM.
        
        Returns:
            Matriz numpy de shape (n_soluciones, n_objetivos).
        """
        if not self.solutions:
            return np.array([])
        
        return np.array([sol.objectives for sol in self.solutions])
    
    def get_parameters_matrix(self) -> NDArray[np.float64]:
        """
        Retorna la matriz de parámetros de todas las soluciones.
        
        Returns:
            Matriz numpy de shape (n_soluciones, n_parámetros).
        """
        if not self.solutions:
            return np.array([])
        
        return np.array([sol.parameters for sol in self.solutions])
    
    def to_list(self) -> List[Dict]:
        """
        Convierte el frente a lista de diccionarios.
        
        Returns:
            Lista de diccionarios con 'position' y 'objectives'.
        """
        return [sol.to_dict() for sol in self.solutions]
    
    def get_best_by_objective(self, objective_index: int) -> Solution:
        """
        Obtiene la solución con mejor valor en un objetivo específico.
        
        Args:
            objective_index: Índice del objetivo (0=Entropía, 1=SSIM, 2=VQI).
        
        Returns:
            Solución con mejor valor en el objetivo especificado.
        """
        if not self.solutions:
            raise ValueError("El frente de Pareto está vacío")
        
        if self.maximize:
            return max(self.solutions, key=lambda s: s.objectives[objective_index])
        else:
            return min(self.solutions, key=lambda s: s.objectives[objective_index])
    
    def get_compromise_solution(self) -> Solution:
        """
        Obtiene la solución de compromiso (más cercana al punto ideal normalizado).
        
        Returns:
            Solución más balanceada entre todos los objetivos.
        """
        if not self.solutions:
            raise ValueError("El frente de Pareto está vacío")
        
        objectives_matrix = self.get_decision_matrix()
        
        # Normalizar objetivos
        obj_min = objectives_matrix.min(axis=0)
        obj_max = objectives_matrix.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range == 0] = 1  # Evitar división por cero
        
        normalized = (objectives_matrix - obj_min) / obj_range
        
        if self.maximize:
            # Punto ideal es (1, 1, 1) para maximización
            ideal_point = np.ones(normalized.shape[1])
        else:
            # Punto ideal es (0, 0, 0) para minimización
            ideal_point = np.zeros(normalized.shape[1])
        
        # Calcular distancia euclidiana al punto ideal
        distances = np.linalg.norm(normalized - ideal_point, axis=1)
        
        best_index = np.argmin(distances)
        return self.solutions[best_index]


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
    reference_point: NDArray[np.float64],
    maximize: bool = True
) -> float:
    """
    Calcula el hipervolumen exacto del Frente de Pareto (2 o 3 objetivos).

    El hipervolumen mide el volumen del espacio objetivo dominado por el
    frente respecto a un punto de referencia. Es la métrica estándar de
    calidad conjunta de convergencia y diversidad en optimización
    multiobjetivo.

    Args:
        pareto_front: Lista de soluciones del Frente de Pareto
                      (diccionarios con clave 'objectives').
        reference_point: Punto de referencia. Para maximización debe estar
                         dominado por todas las soluciones (peor que todas);
                         para minimización, dominado en sentido inverso.
        maximize: Si True (por defecto), los objetivos se maximizan.

    Returns:
        Valor exacto del hipervolumen (área en 2D, volumen en 3D).

    Examples:
        >>> pareto = [
        ...     {'objectives': np.array([1.0, 5.0])},
        ...     {'objectives': np.array([2.0, 3.0])},
        ...     {'objectives': np.array([3.0, 1.0])},
        ... ]
        >>> hv = calculate_hypervolume(pareto, np.array([0.0, 0.0]))
    """
    if not pareto_front:
        return 0.0

    points = np.array([s['objectives'] for s in pareto_front], dtype=np.float64)
    ref = np.asarray(reference_point, dtype=np.float64)

    # Convertir todo a maximización con referencia por debajo
    if not maximize:
        points = -points
        ref = -ref

    # Descartar puntos que no dominan al punto de referencia
    points = points[np.all(points > ref, axis=1)]
    if len(points) == 0:
        return 0.0

    n_objectives = points.shape[1]
    if n_objectives == 2:
        return _hypervolume_2d_max(points, ref)
    elif n_objectives == 3:
        return _hypervolume_3d_max(points, ref)
    else:
        raise NotImplementedError(
            f"Cálculo de hipervolumen no implementado para {n_objectives} objetivos"
        )


def _hypervolume_2d_max(
    points: NDArray[np.float64],
    ref: NDArray[np.float64]
) -> float:
    """
    Hipervolumen exacto 2D (maximización): área de la unión de rectángulos
    [ref_x, x_i] x [ref_y, y_i], calculada por barrido en franjas
    horizontales sobre los puntos ordenados por x descendente.
    """
    order = np.argsort(-points[:, 0])
    hv = 0.0
    y_cubierto = ref[1]
    for x, y in points[order]:
        if y > y_cubierto:
            hv += (x - ref[0]) * (y - y_cubierto)
            y_cubierto = y
    return float(hv)


def _hypervolume_3d_max(
    points: NDArray[np.float64],
    ref: NDArray[np.float64]
) -> float:
    """
    Hipervolumen exacto 3D (maximización) por barrido dimensional:
    se procesan los puntos por su tercera coordenada en orden descendente
    y se integra, para cada franja de altura entre niveles consecutivos de
    z, el área 2D (hipervolumen 2D) de los puntos ya procesados.
    """
    # Ordenar por z descendente
    order = np.argsort(-points[:, 2])
    pts = points[order]
    z_values = pts[:, 2]

    hv = 0.0
    for k in range(len(pts)):
        # Franja entre z_k y el siguiente nivel (o la referencia al final)
        z_top = z_values[k]
        z_bottom = z_values[k + 1] if k + 1 < len(pts) else ref[2]
        if z_top <= z_bottom:
            continue
        # Área 2D de todos los puntos con z >= z_top
        area = _hypervolume_2d_max(pts[:k + 1, :2], ref[:2])
        hv += area * (z_top - z_bottom)
    return float(hv)


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
