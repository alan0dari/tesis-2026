"""
Implementación de SMPSO (Speed-constrained Multi-objective Particle Swarm Optimization).

SMPSO es un algoritmo de optimización multiobjetivo que utiliza enjambre de
partículas con restricción de velocidad para encontrar el Frente de Pareto.

Este módulo incluye:
- Particle: Representación de una partícula en el enjambre
- SMPSO: Algoritmo genérico de SMPSO
- SMPSOImageOptimizer: SMPSO especializado para optimización de imágenes con CLAHE
"""

from typing import List, Callable, Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
import cv2

from src.optimization.pareto import ParetoFront, Solution
from src.clahe.processor import CLAHEProcessor
from src.metrics.entropy import calculate_entropy
from src.metrics.ssim import calculate_ssim
from src.metrics.vqi import calculate_vqi


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


class SMPSOImageOptimizer:
    """
    Optimizador SMPSO especializado para mejora de imágenes médicas con CLAHE.
    
    Este optimizador busca los mejores parámetros de CLAHE (R_x, R_y, clip_limit)
    que maximizan simultáneamente tres métricas de calidad de imagen:
    - Entropía: Cantidad de información/detalle en la imagen
    - SSIM: Similitud estructural con la imagen original
    - VQI: Índice de calidad visual percibida
    
    El resultado es un Frente de Pareto 3D con soluciones óptimas.
    
    Attributes:
        image: Imagen original en escala de grises.
        n_particles: Número de partículas en el enjambre.
        max_iterations: Número máximo de iteraciones.
        archive_size: Tamaño máximo del archivo externo (Frente de Pareto).
    
    Examples:
        >>> import cv2
        >>> image = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)
        >>> optimizer = SMPSOImageOptimizer(
        ...     image=image,
        ...     n_particles=50,
        ...     max_iterations=100
        ... )
        >>> pareto_front = optimizer.run()
        >>> print(f"Encontradas {len(pareto_front)} soluciones óptimas")
        >>> 
        >>> # Obtener matriz de decisión para MCDM
        >>> decision_matrix = pareto_front.get_decision_matrix()
    """
    
    # Límites de los parámetros de CLAHE
    BOUNDS = {
        'rx': (2, 64),           # Regiones en X
        'ry': (2, 64),           # Regiones en Y  
        'clip_limit': (1.0, 4.0) # Límite de contraste
    }
    
    def __init__(
        self,
        image: NDArray[np.uint8],
        n_particles: int = 50,
        max_iterations: int = 100,
        archive_size: int = 100,
        c1: float = 1.5,
        c2: float = 1.5,
        w_max: float = 0.9,
        w_min: float = 0.4,
        mutation_probability: float = 0.1,
        mutation_distribution_index: float = 20.0,
        verbose: bool = True,
        seed: Optional[int] = None,
        resize_for_optimization: Optional[int] = 512
    ):
        """
        Inicializa el optimizador SMPSO para imágenes.
        
        Args:
            image: Imagen en escala de grises (uint8) a optimizar.
            n_particles: Número de partículas en el enjambre.
            max_iterations: Número de iteraciones del algoritmo.
            archive_size: Tamaño máximo del Frente de Pareto.
            c1: Coeficiente cognitivo (aprendizaje personal).
            c2: Coeficiente social (aprendizaje del enjambre).
            w_max: Peso de inercia máximo (inicio).
            w_min: Peso de inercia mínimo (final).
            mutation_probability: Probabilidad de mutación polinomial.
            mutation_distribution_index: Índice de distribución para mutación.
            verbose: Si True, muestra progreso durante la optimización.
            seed: Semilla para reproducibilidad (opcional).
            resize_for_optimization: Tamaño máximo de la dimensión mayor para
                                     la imagen usada durante la búsqueda.
                                     None = usar imagen original (más lento).
                                     512 es un buen balance velocidad/precisión.
        
        Raises:
            ValueError: Si la imagen no es válida.
        """
        # Validar imagen
        if image is None or image.size == 0:
            raise ValueError("La imagen no puede estar vacía")
        
        if image.ndim != 2:
            raise ValueError(
                f"La imagen debe ser 2D (escala de grises), "
                f"pero tiene {image.ndim} dimensiones"
            )
        
        if image.dtype != np.uint8:
            raise ValueError(
                f"La imagen debe ser de tipo uint8, pero es {image.dtype}"
            )
        
        self.image = image
        self.image_original = image  # Guardar original para generar imágenes finales
        
        # Redimensionar para optimización si es necesario
        self.resize_for_optimization = resize_for_optimization
        if resize_for_optimization is not None:
            h, w = image.shape
            max_dim = max(h, w)
            if max_dim > resize_for_optimization:
                scale = resize_for_optimization / max_dim
                new_h = int(h * scale)
                new_w = int(w * scale)
                self.image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                if verbose:
                    print(f"Imagen redimensionada para optimización: {image.shape} -> {self.image.shape}")
        
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.archive_size = archive_size
        self.c1 = c1
        self.c2 = c2
        self.w_max = w_max
        self.w_min = w_min
        self.mutation_probability = mutation_probability
        self.mutation_distribution_index = mutation_distribution_index
        self.verbose = verbose
        
        # Configurar semilla si se proporciona
        if seed is not None:
            np.random.seed(seed)
        
        # Convertir límites a arrays para facilitar operaciones
        self.bounds_lower = np.array([
            self.BOUNDS['rx'][0],
            self.BOUNDS['ry'][0],
            self.BOUNDS['clip_limit'][0]
        ])
        self.bounds_upper = np.array([
            self.BOUNDS['rx'][1],
            self.BOUNDS['ry'][1],
            self.BOUNDS['clip_limit'][1]
        ])
        
        # Calcular delta para restricción de velocidad
        self.delta = (self.bounds_upper - self.bounds_lower) / 2.0
        
        # Inicializar enjambre y archivo
        self.swarm: List[Particle] = []
        self.pareto_front = ParetoFront(max_size=archive_size, maximize=True)
        
        # Contadores para estadísticas
        self.evaluations = 0
        self.history: List[dict] = []
    
    def run(self) -> ParetoFront:
        """
        Ejecuta el algoritmo SMPSO para encontrar el Frente de Pareto óptimo.
        
        Returns:
            ParetoFront: Objeto con las soluciones no dominadas.
            
        Examples:
            >>> optimizer = SMPSOImageOptimizer(image, n_particles=50)
            >>> pareto_front = optimizer.run()
            >>> 
            >>> # Acceder a las soluciones
            >>> for solution in pareto_front:
            ...     print(f"Params: {solution.parameters}")
            ...     print(f"Objetivos: {solution.objectives}")
        """
        if self.verbose:
            print("=" * 60)
            print("SMPSO - Optimización de CLAHE para Imágenes Médicas")
            print("=" * 60)
            print(f"Partículas: {self.n_particles}")
            print(f"Iteraciones: {self.max_iterations}")
            print(f"Tamaño máximo del archivo: {self.archive_size}")
            print(f"Tamaño de imagen: {self.image.shape}")
            print("=" * 60)
        
        # Paso 1: Inicializar enjambre
        self._initialize_swarm()
        
        if self.verbose:
            print(f"\nInicialización completa. "
                  f"Frente de Pareto: {len(self.pareto_front)} soluciones")
        
        # Paso 2: Bucle principal de optimización
        for iteration in range(self.max_iterations):
            # Calcular peso de inercia (decrece linealmente)
            w = self.w_max - (self.w_max - self.w_min) * (iteration / self.max_iterations)
            
            for particle in self.swarm:
                # Seleccionar líder del archivo
                if len(self.pareto_front) > 0:
                    leader = self.pareto_front.select_leader()
                    leader_position = leader.parameters
                else:
                    leader_position = particle.best_position
                
                # Actualizar velocidad con restricción
                self._update_velocity(particle, leader_position, w)
                
                # Actualizar posición
                self._update_position(particle)
                
                # Aplicar mutación polinomial
                self._polynomial_mutation(particle)
                
                # Evaluar función objetivo
                particle.objectives = self._evaluate(particle.position)
                self.evaluations += 1
                
                # Actualizar mejor posición personal
                self._update_personal_best(particle)
                
                # Intentar agregar al Frente de Pareto
                solution = Solution(
                    parameters=particle.position.copy(),
                    objectives=particle.objectives.copy()
                )
                self.pareto_front.add(solution)
            
            # Guardar estadísticas de la iteración
            self._record_iteration(iteration, w)
            
            # Mostrar progreso
            if self.verbose and (iteration + 1) % 10 == 0:
                self._print_progress(iteration + 1)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("OPTIMIZACIÓN COMPLETADA")
            print("=" * 60)
            print(f"Evaluaciones totales: {self.evaluations}")
            print(f"Soluciones en el Frente de Pareto: {len(self.pareto_front)}")
            self._print_pareto_summary()
        
        return self.pareto_front
    
    def _initialize_swarm(self) -> None:
        """
        Inicializa el enjambre con partículas aleatorias.
        
        Las primeras dos dimensiones (R_x, R_y) se redondean a enteros
        ya que representan el número de regiones de CLAHE.
        """
        self.swarm = []
        
        for _ in range(self.n_particles):
            # Generar posición aleatoria
            position = np.random.uniform(self.bounds_lower, self.bounds_upper)
            
            # Redondear R_x y R_y a enteros
            position[0] = round(position[0])
            position[1] = round(position[1])
            
            # Generar velocidad inicial pequeña
            velocity = np.random.uniform(-self.delta * 0.1, self.delta * 0.1)
            
            # Crear partícula
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy()
            )
            
            # Evaluar partícula
            particle.objectives = self._evaluate(position)
            particle.best_objectives = particle.objectives.copy()
            self.evaluations += 1
            
            self.swarm.append(particle)
            
            # Agregar al Frente de Pareto si no es dominada
            solution = Solution(
                parameters=position.copy(),
                objectives=particle.objectives.copy()
            )
            self.pareto_front.add(solution)
    
    def _evaluate(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Evalúa una partícula aplicando CLAHE y calculando métricas.
        
        Args:
            position: Parámetros [R_x, R_y, clip_limit].
        
        Returns:
            Array con [entropía, SSIM, VQI].
        """
        # Extraer y validar parámetros
        rx = int(round(np.clip(position[0], self.BOUNDS['rx'][0], self.BOUNDS['rx'][1])))
        ry = int(round(np.clip(position[1], self.BOUNDS['ry'][0], self.BOUNDS['ry'][1])))
        clip_limit = float(np.clip(
            position[2],
            self.BOUNDS['clip_limit'][0],
            self.BOUNDS['clip_limit'][1]
        ))
        
        try:
            # Aplicar CLAHE
            processor = CLAHEProcessor(rx=rx, ry=ry, clip_limit=clip_limit)
            enhanced_image = processor.process(self.image)
            
            # Calcular métricas (todas a maximizar)
            entropy = calculate_entropy(enhanced_image)
            ssim = calculate_ssim(self.image, enhanced_image)
            vqi = calculate_vqi(enhanced_image)
            
            return np.array([entropy, ssim, vqi])
            
        except Exception as e:
            # En caso de error, retornar valores muy bajos
            if self.verbose:
                print(f"  Warning: Error evaluando ({rx}, {ry}, {clip_limit:.2f}): {e}")
            return np.array([0.0, 0.0, 0.0])
    
    def _update_velocity(
        self,
        particle: Particle,
        leader_position: NDArray[np.float64],
        w: float
    ) -> None:
        """
        Actualiza la velocidad de la partícula con restricción SMPSO.
        
        La ecuación de velocidad es:
        v = w*v + c1*r1*(pbest - x) + c2*r2*(leader - x)
        
        Luego se aplica la restricción:
        v = clamp(v, -delta, +delta)
        
        Args:
            particle: Partícula a actualizar.
            leader_position: Posición del líder seleccionado.
            w: Peso de inercia actual.
        """
        r1 = np.random.random(3)
        r2 = np.random.random(3)
        
        # Componente cognitivo (hacia mejor posición personal)
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        
        # Componente social (hacia el líder)
        social = self.c2 * r2 * (leader_position - particle.position)
        
        # Actualizar velocidad
        particle.velocity = w * particle.velocity + cognitive + social
        
        # Aplicar restricción de velocidad (característica clave de SMPSO)
        particle.velocity = np.clip(particle.velocity, -self.delta, self.delta)
    
    def _update_position(self, particle: Particle) -> None:
        """
        Actualiza la posición de la partícula y verifica límites.
        
        Args:
            particle: Partícula a actualizar.
        """
        # Actualizar posición
        particle.position = particle.position + particle.velocity
        
        # Mantener dentro de límites
        for i in range(3):
            if particle.position[i] < self.bounds_lower[i]:
                particle.position[i] = self.bounds_lower[i]
                particle.velocity[i] = 0
            elif particle.position[i] > self.bounds_upper[i]:
                particle.position[i] = self.bounds_upper[i]
                particle.velocity[i] = 0
        
        # Redondear R_x y R_y a enteros
        particle.position[0] = round(particle.position[0])
        particle.position[1] = round(particle.position[1])
    
    def _polynomial_mutation(self, particle: Particle) -> None:
        """
        Aplica mutación polinomial para mantener diversidad.
        
        La mutación polinomial es característica de SMPSO y ayuda a
        explorar nuevas regiones del espacio de búsqueda.
        
        Args:
            particle: Partícula a mutar.
        """
        if np.random.random() > self.mutation_probability:
            return
        
        eta = self.mutation_distribution_index
        
        for i in range(3):
            if np.random.random() < 1.0 / 3.0:  # Probabilidad de mutar cada gen
                u = np.random.random()
                
                delta_lower = (particle.position[i] - self.bounds_lower[i]) / \
                             (self.bounds_upper[i] - self.bounds_lower[i])
                delta_upper = (self.bounds_upper[i] - particle.position[i]) / \
                             (self.bounds_upper[i] - self.bounds_lower[i])
                
                if u < 0.5:
                    xy = 1.0 - delta_lower
                    val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
                    delta_q = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta_upper
                    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - val ** (1.0 / (eta + 1.0))
                
                particle.position[i] += delta_q * (self.bounds_upper[i] - self.bounds_lower[i])
                particle.position[i] = np.clip(
                    particle.position[i],
                    self.bounds_lower[i],
                    self.bounds_upper[i]
                )
        
        # Redondear R_x y R_y después de mutación
        particle.position[0] = round(particle.position[0])
        particle.position[1] = round(particle.position[1])
    
    def _update_personal_best(self, particle: Particle) -> None:
        """
        Actualiza la mejor posición personal si la actual la domina.
        
        Args:
            particle: Partícula a evaluar.
        """
        # Verificar si la posición actual domina a la mejor personal
        current_dominates = (
            np.all(particle.objectives >= particle.best_objectives) and
            np.any(particle.objectives > particle.best_objectives)
        )
        
        if current_dominates:
            particle.best_position = particle.position.copy()
            particle.best_objectives = particle.objectives.copy()
        elif not np.all(particle.best_objectives >= particle.objectives):
            # Si ninguna domina, seleccionar aleatoriamente
            if np.random.random() < 0.5:
                particle.best_position = particle.position.copy()
                particle.best_objectives = particle.objectives.copy()
    
    def _record_iteration(self, iteration: int, w: float) -> None:
        """Registra estadísticas de la iteración."""
        objectives = self.pareto_front.get_decision_matrix()
        
        if len(objectives) > 0:
            self.history.append({
                'iteration': iteration,
                'w': w,
                'pareto_size': len(self.pareto_front),
                'entropy_mean': np.mean(objectives[:, 0]),
                'ssim_mean': np.mean(objectives[:, 1]),
                'vqi_mean': np.mean(objectives[:, 2]),
                'entropy_max': np.max(objectives[:, 0]),
                'ssim_max': np.max(objectives[:, 1]),
                'vqi_max': np.max(objectives[:, 2])
            })
    
    def _print_progress(self, iteration: int) -> None:
        """Imprime el progreso de la optimización."""
        if len(self.history) > 0:
            stats = self.history[-1]
            print(f"Iter {iteration:4d}/{self.max_iterations} | "
                  f"Pareto: {stats['pareto_size']:3d} | "
                  f"H: {stats['entropy_max']:.3f} | "
                  f"SSIM: {stats['ssim_max']:.4f} | "
                  f"VQI: {stats['vqi_max']:.2f}")
    
    def _print_pareto_summary(self) -> None:
        """Imprime resumen del Frente de Pareto final."""
        if len(self.pareto_front) == 0:
            print("No se encontraron soluciones")
            return
        
        objectives = self.pareto_front.get_decision_matrix()
        params = self.pareto_front.get_parameters_matrix()
        
        print(f"\nResumen del Frente de Pareto:")
        print(f"  Entropía:  min={objectives[:, 0].min():.4f}, "
              f"max={objectives[:, 0].max():.4f}, "
              f"mean={objectives[:, 0].mean():.4f}")
        print(f"  SSIM:      min={objectives[:, 1].min():.4f}, "
              f"max={objectives[:, 1].max():.4f}, "
              f"mean={objectives[:, 1].mean():.4f}")
        print(f"  VQI:       min={objectives[:, 2].min():.2f}, "
              f"max={objectives[:, 2].max():.2f}, "
              f"mean={objectives[:, 2].mean():.2f}")
        print(f"\nRango de parámetros:")
        print(f"  R_x:       {params[:, 0].min():.0f} - {params[:, 0].max():.0f}")
        print(f"  R_y:       {params[:, 1].min():.0f} - {params[:, 1].max():.0f}")
        print(f"  Clip:      {params[:, 2].min():.2f} - {params[:, 2].max():.2f}")
    
    def get_enhanced_image(
        self,
        solution: Solution,
        use_original: bool = True
    ) -> NDArray[np.uint8]:
        """
        Obtiene la imagen mejorada usando los parámetros de una solución.
        
        Args:
            solution: Solución del Frente de Pareto.
            use_original: Si True, aplica CLAHE a la imagen original (tamaño completo).
                          Si False, aplica a la imagen redimensionada usada en optimización.
        
        Returns:
            Imagen mejorada con CLAHE.
        
        Examples:
            >>> pareto_front = optimizer.run()
            >>> best_solution = pareto_front.get_compromise_solution()
            >>> enhanced = optimizer.get_enhanced_image(best_solution)
        """
        rx = int(round(solution.parameters[0]))
        ry = int(round(solution.parameters[1]))
        clip_limit = float(solution.parameters[2])
        
        processor = CLAHEProcessor(rx=rx, ry=ry, clip_limit=clip_limit)
        
        # Usar imagen original o redimensionada según parámetro
        image_to_process = self.image_original if use_original else self.image
        return processor.process(image_to_process)
    
    def get_all_enhanced_images(
        self,
        use_original: bool = True
    ) -> List[Tuple[Solution, NDArray[np.uint8]]]:
        """
        Genera todas las imágenes mejoradas del Frente de Pareto.
        
        Args:
            use_original: Si True, usa imagen original; si False, redimensionada.
        
        Returns:
            Lista de tuplas (solución, imagen_mejorada).
        """
        results = []
        for solution in self.pareto_front:
            enhanced = self.get_enhanced_image(solution, use_original=use_original)
            results.append((solution, enhanced))
        return results


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
