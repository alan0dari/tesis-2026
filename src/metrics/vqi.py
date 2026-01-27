"""
Cálculo del Índice de Calidad Visual (VQI - Visual Quality Index).

VQI evalúa la calidad visual de una imagen considerando características
perceptuales como contraste, nitidez y artefactos.
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


def calculate_vqi(
    image: NDArray[np.uint8],
    reference_image: Optional[NDArray[np.uint8]] = None,
    block_size: int = 8
) -> float:
    """
    Calcula el Índice de Calidad Visual (VQI) de una imagen.
    
    VQI considera múltiples factores perceptuales:
    - Contraste local
    - Nitidez (sharpness)
    - Ausencia de artefactos
    - Distribución de intensidades
    
    Args:
        image: Imagen a evaluar.
        reference_image: Imagen de referencia (opcional).
                        Si se proporciona, VQI evalúa calidad relativa.
        block_size: Tamaño de bloque para análisis local.
    
    Returns:
        Valor de VQI. Mayor valor indica mejor calidad visual.
        Rango típico: [0, 100].
    
    Examples:
        >>> import numpy as np
        >>> # Imagen con buen contraste
        >>> good_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> vqi_good = calculate_vqi(good_img)
        >>> print(f"VQI: {vqi_good:.2f}")
        
        >>> # Comparación con referencia
        >>> original = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
        >>> modified = original.copy()
        >>> vqi_relative = calculate_vqi(modified, reference_image=original)
        >>> print(f"VQI relativo: {vqi_relative:.2f}")
    """
    if image.ndim != 2:
        raise ValueError(f"La imagen debe ser 2D, pero tiene {image.ndim} dimensiones")
    
    # Convertir a float para cálculos
    img_float = image.astype(np.float64)
    
    # Componente 1: Contraste local
    contrast_score = _calculate_local_contrast(img_float, block_size)
    
    # Componente 2: Nitidez (sharpness)
    sharpness_score = _calculate_sharpness(img_float)
    
    # Componente 3: Distribución de intensidades (entropía)
    distribution_score = _calculate_intensity_distribution(image)
    
    # Si hay imagen de referencia, calcular similitud estructural
    if reference_image is not None:
        if image.shape != reference_image.shape:
            raise ValueError("La imagen y la referencia deben tener la misma forma")
        
        from src.metrics.ssim import calculate_ssim
        ssim_score = calculate_ssim(reference_image, image)
        
        # VQI con referencia: combinación ponderada
        vqi = (
            0.3 * contrast_score +
            0.2 * sharpness_score +
            0.2 * distribution_score +
            0.3 * (ssim_score * 100)
        )
    else:
        # VQI sin referencia: solo características intrínsecas
        vqi = (
            0.4 * contrast_score +
            0.3 * sharpness_score +
            0.3 * distribution_score
        )
    
    return float(vqi)


def _calculate_local_contrast(
    image: NDArray[np.float64],
    block_size: int
) -> float:
    """
    Calcula el contraste local promedio de la imagen.
    
    El contraste local se mide como la desviación estándar en bloques
    de la imagen. Utiliza operaciones vectorizadas para eficiencia.
    
    Args:
        image: Imagen en formato float.
        block_size: Tamaño de los bloques.
    
    Returns:
        Puntuación de contraste normalizada [0, 100].
    """
    from scipy import ndimage
    
    # Calcular media local usando filtro uniforme (más rápido)
    local_mean = ndimage.uniform_filter(image, size=block_size)
    
    # Calcular varianza local: E[X^2] - E[X]^2
    local_sqr_mean = ndimage.uniform_filter(image**2, size=block_size)
    local_variance = local_sqr_mean - local_mean**2
    
    # Evitar valores negativos por errores numéricos
    local_variance = np.maximum(local_variance, 0)
    
    # Desviación estándar local
    local_std = np.sqrt(local_variance)
    
    # Contraste promedio
    mean_contrast = np.mean(local_std)
    
    # Normalizar a escala 0-100 (asumiendo std máxima de ~75 para uint8)
    normalized_contrast = min(100, (mean_contrast / 75.0) * 100)
    
    return float(normalized_contrast)


def _calculate_sharpness(image: NDArray[np.float64]) -> float:
    """
    Calcula la nitidez de la imagen usando gradientes.
    
    La nitidez se mide analizando la magnitud de los gradientes
    en la imagen. Mayor gradiente indica bordes más definidos.
    
    Args:
        image: Imagen en formato float.
    
    Returns:
        Puntuación de nitidez normalizada [0, 100].
    """
    # Calcular gradientes usando filtro Sobel
    from scipy import ndimage
    
    # Gradientes en x e y
    grad_x = ndimage.sobel(image, axis=1)
    grad_y = ndimage.sobel(image, axis=0)
    
    # Magnitud del gradiente
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Nitidez como magnitud promedio del gradiente
    sharpness = np.mean(gradient_magnitude)
    
    # Normalizar a escala 0-100 (asumiendo magnitud máxima de ~100)
    normalized_sharpness = min(100, (sharpness / 50.0) * 100)
    
    return float(normalized_sharpness)


def _calculate_intensity_distribution(image: NDArray[np.uint8]) -> float:
    """
    Evalúa la distribución de intensidades usando entropía.
    
    Una buena distribución de intensidades mejora la calidad visual
    al utilizar todo el rango dinámico disponible.
    
    Args:
        image: Imagen en formato uint8.
    
    Returns:
        Puntuación de distribución normalizada [0, 100].
    """
    from src.metrics.entropy import calculate_entropy_normalized
    
    # Entropía normalizada como medida de distribución
    entropy_norm = calculate_entropy_normalized(image)
    
    # Convertir a escala 0-100
    distribution_score = entropy_norm * 100
    
    return float(distribution_score)


def calculate_vqi_components(
    image: NDArray[np.uint8],
    reference_image: Optional[NDArray[np.uint8]] = None,
    block_size: int = 8
) -> dict:
    """
    Calcula VQI y retorna también los componentes individuales.
    
    Útil para análisis detallado de qué aspectos contribuyen
    a la calidad visual.
    
    Args:
        image: Imagen a evaluar.
        reference_image: Imagen de referencia (opcional).
        block_size: Tamaño de bloque para análisis local.
    
    Returns:
        Diccionario con VQI total y componentes individuales.
    
    Examples:
        >>> import numpy as np
        >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> components = calculate_vqi_components(img)
        >>> print(f"VQI Total: {components['vqi_total']:.2f}")
        >>> print(f"Contraste: {components['contrast']:.2f}")
        >>> print(f"Nitidez: {components['sharpness']:.2f}")
    """
    img_float = image.astype(np.float64)
    
    contrast = _calculate_local_contrast(img_float, block_size)
    sharpness = _calculate_sharpness(img_float)
    distribution = _calculate_intensity_distribution(image)
    
    components = {
        'contrast': contrast,
        'sharpness': sharpness,
        'distribution': distribution,
    }
    
    if reference_image is not None:
        from src.metrics.ssim import calculate_ssim
        ssim = calculate_ssim(reference_image, image)
        components['ssim'] = ssim * 100
        
        vqi = (
            0.3 * contrast +
            0.2 * sharpness +
            0.2 * distribution +
            0.3 * (ssim * 100)
        )
    else:
        vqi = (
            0.4 * contrast +
            0.3 * sharpness +
            0.3 * distribution
        )
    
    components['vqi_total'] = vqi
    
    return components
