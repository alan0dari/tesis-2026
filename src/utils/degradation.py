"""
Módulo para simular problemas de visualización en radiografías.

Este módulo contiene funciones para degradar imágenes de manera controlada,
simulando problemas que ocurren naturalmente en la adquisición de radiografías
y que son aptos para ser corregidos con técnicas de mejora de contraste como CLAHE.

Tipos de degradación implementados:
- Bajo contraste: Compresión del rango dinámico
- Subexposición: Imagen oscurecida
- Sobreexposición: Imagen sobreiluminada
- Contraste local pobre: Pérdida de detalles finos
- Histograma sesgado: Distribución no uniforme de intensidades
"""

from typing import Tuple, Optional, Union, Dict, List
from enum import Enum
import numpy as np
from numpy.typing import NDArray
import cv2


class DegradationType(Enum):
    """Tipos de degradación disponibles."""
    LOW_CONTRAST = "low_contrast"
    UNDEREXPOSURE = "underexposure"
    OVEREXPOSURE = "overexposure"
    POOR_LOCAL_CONTRAST = "poor_local_contrast"
    SKEWED_HISTOGRAM = "skewed_histogram"


def apply_low_contrast(
    image: NDArray[np.uint8],
    factor: float = 0.5
) -> NDArray[np.uint8]:
    """
    Aplica degradación de bajo contraste comprimiendo el rango dinámico.
    
    Simula una imagen "lavada" o plana donde los detalles son difíciles
    de distinguir debido a la falta de diferencia entre tonos.
    
    Args:
        image: Imagen en escala de grises (uint8).
        factor: Factor de compresión (0.0-1.0). Menor = más degradación.
                0.5 significa que el rango se reduce a la mitad.
    
    Returns:
        Imagen degradada con bajo contraste.
    
    Examples:
        >>> img = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)
        >>> degraded = apply_low_contrast(img, factor=0.4)
    """
    if factor <= 0 or factor > 1:
        raise ValueError(f"factor debe estar en (0, 1], pero es {factor}")
    
    # Convertir a float para operaciones
    img_float = image.astype(np.float64)
    
    # Calcular el centro del rango (hacia donde comprimir)
    center = 128.0
    
    # Comprimir hacia el centro
    result = center + (img_float - center) * factor
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_underexposure(
    image: NDArray[np.uint8],
    gamma: float = 2.5,
    offset: int = -30
) -> NDArray[np.uint8]:
    """
    Aplica degradación de subexposición (imagen oscura).
    
    Simula una radiografía tomada con parámetros de exposición
    insuficientes, resultando en una imagen oscura con pérdida
    de detalles en las zonas de sombra.
    
    Args:
        image: Imagen en escala de grises (uint8).
        gamma: Factor gamma (>1 oscurece). Valores típicos: 1.5-3.0.
        offset: Desplazamiento de brillo (negativo = más oscuro).
    
    Returns:
        Imagen degradada subexpuesta.
    
    Examples:
        >>> img = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)
        >>> dark = apply_underexposure(img, gamma=2.0, offset=-20)
    """
    # Normalizar a [0, 1]
    img_norm = image.astype(np.float64) / 255.0
    
    # Aplicar corrección gamma inversa (oscurecer)
    img_gamma = np.power(img_norm, gamma)
    
    # Convertir de vuelta a [0, 255] y aplicar offset
    result = img_gamma * 255.0 + offset
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_overexposure(
    image: NDArray[np.uint8],
    gamma: float = 0.5,
    saturation_threshold: int = 240
) -> NDArray[np.uint8]:
    """
    Aplica degradación de sobreexposición (imagen clara/quemada).
    
    Simula una radiografía tomada con exceso de exposición,
    resultando en una imagen muy clara con pérdida de detalles
    en las zonas brillantes (saturación).
    
    Args:
        image: Imagen en escala de grises (uint8).
        gamma: Factor gamma (<1 aclara). Valores típicos: 0.3-0.7.
        saturation_threshold: Umbral sobre el cual saturar a blanco.
    
    Returns:
        Imagen degradada sobreexpuesta.
    
    Examples:
        >>> img = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)
        >>> bright = apply_overexposure(img, gamma=0.5)
    """
    # Normalizar a [0, 1]
    img_norm = image.astype(np.float64) / 255.0
    
    # Aplicar corrección gamma (aclarar)
    img_gamma = np.power(img_norm, gamma)
    
    # Convertir de vuelta a [0, 255]
    result = img_gamma * 255.0
    
    # Simular saturación (zonas quemadas)
    result[result > saturation_threshold] = 255
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_poor_local_contrast(
    image: NDArray[np.uint8],
    blur_kernel: int = 15,
    contrast_reduction: float = 0.6
) -> NDArray[np.uint8]:
    """
    Aplica degradación de contraste local pobre.
    
    Simula pérdida de detalles finos y bordes difusos que pueden
    ocurrir por movimiento del paciente o desenfoque del equipo.
    
    Args:
        image: Imagen en escala de grises (uint8).
        blur_kernel: Tamaño del kernel de suavizado (impar).
        contrast_reduction: Factor de reducción de contraste local.
    
    Returns:
        Imagen degradada con contraste local reducido.
    
    Examples:
        >>> img = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)
        >>> blurry = apply_poor_local_contrast(img, blur_kernel=11)
    """
    # Asegurar kernel impar
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    
    # Aplicar suavizado gaussiano
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    
    # Mezclar imagen original con suavizada (reduce contraste local)
    img_float = image.astype(np.float64)
    blur_float = blurred.astype(np.float64)
    
    result = img_float * contrast_reduction + blur_float * (1 - contrast_reduction)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_skewed_histogram(
    image: NDArray[np.uint8],
    skew_direction: str = "dark",
    intensity: float = 0.7
) -> NDArray[np.uint8]:
    """
    Aplica degradación de histograma sesgado.
    
    Simula una distribución no uniforme de intensidades donde
    la mayoría de los píxeles se concentran en un rango limitado.
    
    Args:
        image: Imagen en escala de grises (uint8).
        skew_direction: Dirección del sesgo ("dark" o "bright").
        intensity: Intensidad del sesgo (0.0-1.0).
    
    Returns:
        Imagen degradada con histograma sesgado.
    
    Examples:
        >>> img = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)
        >>> skewed = apply_skewed_histogram(img, skew_direction="dark", intensity=0.6)
    """
    img_float = image.astype(np.float64) / 255.0
    
    if skew_direction == "dark":
        # Sesgo hacia valores oscuros usando función logarítmica
        # Comprime los valores altos, expande los bajos
        result = np.power(img_float, 1 + intensity)
    elif skew_direction == "bright":
        # Sesgo hacia valores claros usando función exponencial
        result = np.power(img_float, 1 / (1 + intensity))
    else:
        raise ValueError(f"skew_direction debe ser 'dark' o 'bright', no '{skew_direction}'")
    
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def apply_degradation(
    image: NDArray[np.uint8],
    degradation_type: Union[DegradationType, str],
    **kwargs
) -> NDArray[np.uint8]:
    """
    Aplica un tipo específico de degradación a una imagen.
    
    Función de conveniencia que permite aplicar cualquier tipo
    de degradación usando una interfaz unificada.
    
    Args:
        image: Imagen en escala de grises (uint8).
        degradation_type: Tipo de degradación a aplicar.
        **kwargs: Parámetros específicos de cada tipo de degradación.
    
    Returns:
        Imagen degradada.
    
    Examples:
        >>> img = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)
        >>> degraded = apply_degradation(img, DegradationType.LOW_CONTRAST, factor=0.4)
        >>> degraded = apply_degradation(img, "underexposure", gamma=2.0)
    """
    if isinstance(degradation_type, str):
        degradation_type = DegradationType(degradation_type)
    
    degradation_functions = {
        DegradationType.LOW_CONTRAST: apply_low_contrast,
        DegradationType.UNDEREXPOSURE: apply_underexposure,
        DegradationType.OVEREXPOSURE: apply_overexposure,
        DegradationType.POOR_LOCAL_CONTRAST: apply_poor_local_contrast,
        DegradationType.SKEWED_HISTOGRAM: apply_skewed_histogram,
    }
    
    func = degradation_functions[degradation_type]
    return func(image, **kwargs)


def apply_random_degradation(
    image: NDArray[np.uint8],
    seed: Optional[int] = None
) -> Tuple[NDArray[np.uint8], DegradationType, Dict]:
    """
    Aplica una degradación aleatoria a la imagen.
    
    Selecciona aleatoriamente un tipo de degradación y parámetros
    dentro de rangos razonables.
    
    Args:
        image: Imagen en escala de grises (uint8).
        seed: Semilla para reproducibilidad.
    
    Returns:
        Tupla con (imagen_degradada, tipo_degradacion, parametros_usados).
    
    Examples:
        >>> img = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)
        >>> degraded, deg_type, params = apply_random_degradation(img, seed=42)
        >>> print(f"Aplicada degradacion: {deg_type.value} con params: {params}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Seleccionar tipo de degradación aleatoriamente
    deg_type = np.random.choice(list(DegradationType))
    
    # Definir rangos de parámetros para cada tipo
    if deg_type == DegradationType.LOW_CONTRAST:
        params = {'factor': np.random.uniform(0.3, 0.6)}
    elif deg_type == DegradationType.UNDEREXPOSURE:
        params = {
            'gamma': np.random.uniform(1.8, 2.8),
            'offset': np.random.randint(-40, -10)
        }
    elif deg_type == DegradationType.OVEREXPOSURE:
        params = {
            'gamma': np.random.uniform(0.4, 0.7),
            'saturation_threshold': np.random.randint(220, 250)
        }
    elif deg_type == DegradationType.POOR_LOCAL_CONTRAST:
        params = {
            'blur_kernel': np.random.choice([9, 11, 13, 15, 17]),
            'contrast_reduction': np.random.uniform(0.5, 0.7)
        }
    elif deg_type == DegradationType.SKEWED_HISTOGRAM:
        params = {
            'skew_direction': np.random.choice(['dark', 'bright']),
            'intensity': np.random.uniform(0.5, 0.8)
        }
    
    degraded = apply_degradation(image, deg_type, **params)
    
    return degraded, deg_type, params


def create_degradation_set(
    image: NDArray[np.uint8]
) -> Dict[str, Tuple[NDArray[np.uint8], Dict]]:
    """
    Crea un conjunto de imágenes con todas las degradaciones.
    
    Útil para comparar el efecto de diferentes tipos de degradación
    y cómo responde CLAHE a cada una.
    
    Args:
        image: Imagen original en escala de grises.
    
    Returns:
        Diccionario con nombre_degradacion: (imagen_degradada, parametros).
    
    Examples:
        >>> img = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)
        >>> degradations = create_degradation_set(img)
        >>> for name, (degraded, params) in degradations.items():
        ...     print(f"{name}: {params}")
    """
    # Parámetros predeterminados moderados para cada degradación
    presets = {
        'original': (image.copy(), {}),
        'low_contrast_mild': (apply_low_contrast(image, factor=0.6), {'factor': 0.6}),
        'low_contrast_severe': (apply_low_contrast(image, factor=0.35), {'factor': 0.35}),
        'underexposure_mild': (apply_underexposure(image, gamma=1.8, offset=-15), 
                               {'gamma': 1.8, 'offset': -15}),
        'underexposure_severe': (apply_underexposure(image, gamma=2.5, offset=-35),
                                 {'gamma': 2.5, 'offset': -35}),
        'overexposure_mild': (apply_overexposure(image, gamma=0.65, saturation_threshold=245),
                              {'gamma': 0.65, 'saturation_threshold': 245}),
        'overexposure_severe': (apply_overexposure(image, gamma=0.45, saturation_threshold=230),
                                {'gamma': 0.45, 'saturation_threshold': 230}),
        'poor_local_contrast': (apply_poor_local_contrast(image, blur_kernel=13, contrast_reduction=0.6),
                                {'blur_kernel': 13, 'contrast_reduction': 0.6}),
        'skewed_dark': (apply_skewed_histogram(image, skew_direction='dark', intensity=0.7),
                        {'skew_direction': 'dark', 'intensity': 0.7}),
        'skewed_bright': (apply_skewed_histogram(image, skew_direction='bright', intensity=0.7),
                          {'skew_direction': 'bright', 'intensity': 0.7}),
    }
    
    return presets


def get_image_quality_metrics(image: NDArray[np.uint8]) -> Dict[str, float]:
    """
    Calcula métricas básicas de calidad de imagen para diagnóstico.
    
    Útil para evaluar el nivel de degradación antes y después del procesamiento.
    
    Args:
        image: Imagen en escala de grises.
    
    Returns:
        Diccionario con métricas de calidad.
    """
    metrics = {}
    
    # Rango dinámico utilizado
    metrics['min_intensity'] = float(image.min())
    metrics['max_intensity'] = float(image.max())
    metrics['dynamic_range'] = metrics['max_intensity'] - metrics['min_intensity']
    
    # Estadísticas básicas
    metrics['mean'] = float(image.mean())
    metrics['std'] = float(image.std())
    
    # Contraste (desviación estándar normalizada)
    metrics['contrast'] = metrics['std'] / 128.0  # Normalizado a [0, ~2]
    
    # Histograma
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    hist_norm = hist / hist.sum()
    
    # Entropía del histograma
    hist_nonzero = hist_norm[hist_norm > 0]
    metrics['entropy'] = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    
    # Sesgo del histograma (skewness simplificado)
    # >0 = sesgado hacia oscuros, <0 = sesgado hacia claros
    metrics['histogram_skew'] = (metrics['mean'] - 128) / 128
    
    return metrics
