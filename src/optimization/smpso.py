"""
Implementación de SMPSO (Speed-constrained Multi-objective Particle Swarm Optimization).

SMPSO es un algoritmo de optimización multiobjetivo que utiliza enjambre de
partículas con restricción de velocidad para encontrar el Frente de Pareto.
"""

from typing import List, Callable, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class Particle:
    """
    Representa una partícula en el enjambre.
    
    Attributes:
        position: Posición actual en el espacio de búsqueda.
        velocity: Velocidad actual de la partícula.
        best_position: Mejor posición encontrada por esta partícula.
        objectives: Valores de las funciones objetivo en la posición actual.
        best_objectives: Valores de las funciones objetivo en best_position.
    """
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    best_position: NDArray[np.float64]
    objectives: Optional[NDArray[np.float64]] = None
    best_objectives: Optional[NDArray[np.float64]] = None


class SMPSO:
    """
    Optimizador SMPSO (Speed-constrained Multi-objective PSO).
    
    SMPSO es una variante de PSO multiobjetivo que incorpora:
    - Restricción de velocidad basada en límites del problema
    - Actualización de posición con distribución polinomial
    - Mantenimiento de archivo externo para soluciones no dominadas
    
    Examples:
        >>> def objective_function(x):
        ...     # Ejemplo: minimizar x^2 y (x-2)^2
        ...     return [x[0]**2, (x[0]-2)**2]
        >>> 
        >>> optimizer = SMPSO(
        ...     n_particles=30,
        ...     n_iterations=100,
        ...     bounds=[(-5, 5)]
        ... )
        >>> solutions = optimizer.optimize(objective_function)
        >>> print(f"Encontradas {len(solutions)} soluciones en el Frente de Pareto")
    """
    
    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 100,
        bounds: List[Tuple[float, float]] = None,
        c1: float = 1.5,
        c2: float = 1.5,
        mutation_probability: float = 0.1,
        archive_size: int = 100,
        verbose: bool = False
    ):
        """
        Inicializa el optimizador SMPSO.
        
        Args:
            n_particles: Número de partículas en el enjambre.
            n_iterations: Número de iteraciones del algoritmo.
            bounds: Lista de tuplas (min, max) para cada dimensión.
            c1: Coeficiente cognitivo (aprendizaje individual).
            c2: Coeficiente social (aprendizaje del enjambre).
            mutation_probability: Probabilidad de mutación polinomial.
            archive_size: Tamaño máximo del archivo de soluciones no dominadas.
            verbose: Si True, imprime información durante la optimización.
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.c1 = c1
        self.c2 = c2
        self.mutation_probability = mutation_probability
        self.archive_size = archive_size
        self.verbose = verbose
        
        self.swarm: List[Particle] = []
        self.archive: List[Particle] = []
        self.n_dimensions = len(bounds) if bounds else 0
        
    def optimize(
        self,
        objective_function: Callable[[NDArray[np.float64]], List[float]]
    ) -> List[dict]:
        """
        Ejecuta el algoritmo SMPSO para optimizar la función objetivo.
        
        Args:
            objective_function: Función que toma un array de parámetros
                              y retorna una lista de valores objetivo.
        
        Returns:
            Lista de diccionarios con las soluciones del Frente de Pareto.
            Cada diccionario contiene 'position' y 'objectives'.
        
        Examples:
            >>> def multi_objective(x):
            ...     return [x[0]**2, (x[0]-1)**2]
            >>> optimizer = SMPSO(n_particles=20, n_iterations=50, bounds=[(-5, 5)])
            >>> pareto_front = optimizer.optimize(multi_objective)
        """
        # Inicializar enjambre
        self._initialize_swarm(objective_function)
        
        # Bucle principal de optimización
        for iteration in range(self.n_iterations):
            if self.verbose and iteration % 10 == 0:
                print(f"Iteración {iteration}/{self.n_iterations}, "
                      f"Archivo: {len(self.archive)} soluciones")
            
            # Actualizar cada partícula
            for particle in self.swarm:
                # Seleccionar líder del archivo
                leader = self._select_leader()
                
                # Actualizar velocidad con restricción
                self._update_velocity(particle, leader)
                
                # Actualizar posición
                self._update_position(particle)
                
                # Aplicar mutación polinomial
                self._polynomial_mutation(particle)
                
                # Evaluar función objetivo
                particle.objectives = np.array(objective_function(particle.position))
                
                # Actualizar mejor posición personal
                self._update_personal_best(particle)
                
                # Actualizar archivo de soluciones no dominadas
                self._update_archive(particle)
        
        if self.verbose:
            print(f"Optimización completada. Frente de Pareto: {len(self.archive)} soluciones")
        
        # Convertir archivo a formato de salida
        return [
            {
                'position': p.position.copy(),
                'objectives': p.objectives.copy()
            }
            for p in self.archive
        ]
    
    def _initialize_swarm(
        self,
        objective_function: Callable
    ) -> None:
        """Inicializa el enjambre de partículas con posiciones aleatorias."""
        self.swarm = []
        
        for _ in range(self.n_particles):
            # Posición aleatoria dentro de los límites
            position = np.array([
                np.random.uniform(low, high)
                for low, high in self.bounds
            ])
            
            # Velocidad inicial aleatoria
            velocity = np.array([
                np.random.uniform(-abs(high - low) * 0.1, abs(high - low) * 0.1)
                for low, high in self.bounds
            ])
            
            # Crear partícula
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy()
            )
            
            # Evaluar función objetivo
            particle.objectives = np.array(objective_function(position))
            particle.best_objectives = particle.objectives.copy()
            
            self.swarm.append(particle)
            
            # Agregar al archivo si es no dominada
            self._update_archive(particle)
    
    def _update_velocity(self, particle: Particle, leader: Particle) -> None:
        """
        Actualiza la velocidad de una partícula con restricción de velocidad.
        
        La restricción de velocidad en SMPSO se basa en:
        v_max = (upper_bound - lower_bound) / 2
        """
        r1 = np.random.random(self.n_dimensions)
        r2 = np.random.random(self.n_dimensions)
        
        # Coeficiente de inercia adaptativo
        w = 0.9 - (0.5 * (len(self.archive) / self.archive_size))
        w = max(0.4, min(0.9, w))
        
        # Actualización de velocidad estándar de PSO
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        social = self.c2 * r2 * (leader.position - particle.position)
        
        particle.velocity = w * particle.velocity + cognitive + social
        
        # Aplicar restricción de velocidad (componente clave de SMPSO)
        for i, (low, high) in enumerate(self.bounds):
            delta = (high - low) / 2.0
            particle.velocity[i] = np.clip(particle.velocity[i], -delta, delta)
    
    def _update_position(self, particle: Particle) -> None:
        """Actualiza la posición de la partícula y verifica límites."""
        particle.position = particle.position + particle.velocity
        
        # Mantener dentro de los límites
        for i, (low, high) in enumerate(self.bounds):
            if particle.position[i] < low:
                particle.position[i] = low
                particle.velocity[i] = 0
            elif particle.position[i] > high:
                particle.position[i] = high
                particle.velocity[i] = 0
    
    def _polynomial_mutation(self, particle: Particle) -> None:
        """
        Aplica mutación polinomial a la partícula.
        
        La mutación polinomial es característica de SMPSO y ayuda a
        mantener la diversidad del enjambre.
        """
        if np.random.random() > self.mutation_probability:
            return
        
        distribution_index = 20.0
        
        for i, (low, high) in enumerate(self.bounds):
            if np.random.random() < 1.0 / self.n_dimensions:
                u = np.random.random()
                
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (distribution_index + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (distribution_index + 1.0))
                
                particle.position[i] += delta * (high - low)
                particle.position[i] = np.clip(particle.position[i], low, high)
    
    def _update_personal_best(self, particle: Particle) -> None:
        """Actualiza la mejor posición personal si la actual la domina."""
        if self._dominates(particle.objectives, particle.best_objectives):
            particle.best_position = particle.position.copy()
            particle.best_objectives = particle.objectives.copy()
    
    def _update_archive(self, particle: Particle) -> None:
        """Actualiza el archivo de soluciones no dominadas."""
        # Verificar si la partícula es dominada por alguna solución en el archivo
        is_dominated = False
        dominated_indices = []
        
        for i, archived in enumerate(self.archive):
            if self._dominates(archived.objectives, particle.objectives):
                is_dominated = True
                break
            elif self._dominates(particle.objectives, archived.objectives):
                dominated_indices.append(i)
        
        # Si no es dominada, agregar al archivo
        if not is_dominated:
            # Remover soluciones dominadas por la nueva partícula
            for i in reversed(dominated_indices):
                self.archive.pop(i)
            
            # Agregar nueva solución
            new_particle = Particle(
                position=particle.position.copy(),
                velocity=particle.velocity.copy(),
                best_position=particle.position.copy(),
                objectives=particle.objectives.copy(),
                best_objectives=particle.objectives.copy()
            )
            self.archive.append(new_particle)
            
            # Limitar tamaño del archivo
            if len(self.archive) > self.archive_size:
                self._truncate_archive()
    
    def _select_leader(self) -> Particle:
        """Selecciona un líder del archivo mediante selección por torneo binario."""
        if not self.archive:
            return self.swarm[0]
        
        # Selección por torneo binario
        idx1, idx2 = np.random.choice(len(self.archive), 2, replace=False)
        
        # Seleccionar el que tiene menor crowding distance (más aislado)
        # Por simplicidad, selección aleatoria
        return self.archive[idx1] if np.random.random() < 0.5 else self.archive[idx2]
    
    def _dominates(
        self,
        objectives1: NDArray[np.float64],
        objectives2: NDArray[np.float64]
    ) -> bool:
        """
        Verifica si objectives1 domina a objectives2 en el sentido de Pareto.
        
        Asume minimización de todos los objetivos.
        """
        better_in_all = np.all(objectives1 <= objectives2)
        better_in_some = np.any(objectives1 < objectives2)
        return better_in_all and better_in_some
    
    def _truncate_archive(self) -> None:
        """Reduce el tamaño del archivo usando crowding distance."""
        if len(self.archive) <= self.archive_size:
            return
        
        # Calcular crowding distance para cada solución
        crowding_distances = self._calculate_crowding_distances()
        
        # Ordenar por crowding distance (descendente)
        sorted_indices = np.argsort(crowding_distances)[::-1]
        
        # Mantener las soluciones con mayor crowding distance
        self.archive = [self.archive[i] for i in sorted_indices[:self.archive_size]]
    
    def _calculate_crowding_distances(self) -> NDArray[np.float64]:
        """Calcula la crowding distance para cada solución en el archivo."""
        n_solutions = len(self.archive)
        n_objectives = len(self.archive[0].objectives)
        
        distances = np.zeros(n_solutions)
        
        for obj_idx in range(n_objectives):
            # Extraer valores del objetivo
            obj_values = np.array([p.objectives[obj_idx] for p in self.archive])
            
            # Ordenar por este objetivo
            sorted_indices = np.argsort(obj_values)
            
            # Asignar distancia infinita a los extremos
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            # Calcular rango del objetivo
            obj_range = obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]]
            
            if obj_range == 0:
                continue
            
            # Calcular crowding distance para soluciones intermedias
            for i in range(1, n_solutions - 1):
                idx = sorted_indices[i]
                idx_prev = sorted_indices[i - 1]
                idx_next = sorted_indices[i + 1]
                
                distances[idx] += (
                    obj_values[idx_next] - obj_values[idx_prev]
                ) / obj_range
        
        return distances
