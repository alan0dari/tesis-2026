"""
Cálculo del Índice de Similitud Estructural (SSIM).

SSIM es una métrica de calidad de imagen que evalúa la similitud percibida
entre dos imágenes considerando luminancia, contraste y estructura.
"""

from typing import Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity


def calculate_ssim(
    reference_image: NDArray[np.uint8],
    test_image: NDArray[np.uint8],
    win_size: Optional[int] = None,
    data_range: Optional[int] = None,
    multichannel: bool = False,
    gaussian_weights: bool = False,
    full: bool = False
) -> Union[float, Tuple[float, NDArray[np.float64]]]:
    """
    Calcula el Índice de Similitud Estructural (SSIM) entre dos imágenes.
    
    SSIM se define como:
    SSIM(x,y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ
    
    donde:
    - l(x,y): luminancia
    - c(x,y): contraste
    - s(x,y): estructura
    
    Con α = β = γ = 1, la fórmula simplificada es:
    SSIM(x,y) = [(2μ_x·μ_y + C1)(2σ_xy + C2)] / [(μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)]
    
    Args:
        reference_image: Imagen de referencia (original).
        test_image: Imagen de prueba (mejorada o comprimida).
        win_size: Tamaño de la ventana deslizante. Si es None, se usa 7.
        data_range: Rango de datos de la imagen. Si es None, se usa 255 para uint8.
        multichannel: Si True, trata la última dimensión como canales.
        gaussian_weights: Si True, usa pesos gaussianos en lugar de uniformes.
        full: Si True, retorna también el mapa SSIM completo.
    
    Returns:
        SSIM medio entre las imágenes (rango: [-1, 1], típicamente [0, 1]).
        Si full=True, retorna tupla (ssim_medio, mapa_ssim).
    
    Raises:
        ValueError: Si las imágenes no tienen la misma forma.
    
    Examples:
        >>> import numpy as np
        >>> # Imagen original
        >>> original = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> # Imagen idéntica (SSIM = 1.0)
        >>> identical = original.copy()
        >>> ssim_perfect = calculate_ssim(original, identical)
        >>> print(f"SSIM idéntica: {ssim_perfect:.4f}")
        
        >>> # Imagen con ruido (SSIM < 1.0)
        >>> noisy = original + np.random.randint(-20, 20, (100, 100), dtype=np.int16)
        >>> noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        >>> ssim_noisy = calculate_ssim(original, noisy)
        >>> print(f"SSIM con ruido: {ssim_noisy:.4f}")
    """
    if reference_image.shape != test_image.shape:
        raise ValueError(
            f"Las imágenes deben tener la misma forma. "
            f"Referencia: {reference_image.shape}, Prueba: {test_image.shape}"
        )
    
    # Establecer valores predeterminados
    if data_range is None:
        if reference_image.dtype == np.uint8:
            data_range = 255
        elif reference_image.dtype == np.uint16:
            data_range = 65535
        else:
            data_range = np.max(reference_image) - np.min(reference_image)
    
    # Calcular SSIM usando scikit-image
    result = structural_similarity(
        reference_image,
        test_image,
        win_size=win_size,
        data_range=data_range,
        multichannel=multichannel,
        gaussian_weights=gaussian_weights,
        full=full
    )
    
    return result


def calculate_ssim_with_map(
    reference_image: NDArray[np.uint8],
    test_image: NDArray[np.uint8],
    win_size: Optional[int] = None
) -> Tuple[float, NDArray[np.float64]]:
    """
    Calcula SSIM y retorna también el mapa de similitud local.
    
    El mapa SSIM muestra la similitud local en cada región de la imagen,
    útil para identificar áreas con mayor o menor similitud.
    
    Args:
        reference_image: Imagen de referencia.
        test_image: Imagen de prueba.
        win_size: Tamaño de la ventana deslizante.
    
    Returns:
        Tupla (ssim_global, mapa_ssim_local).
    
    Examples:
        >>> import numpy as np
        >>> original = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> modified = original.copy()
        >>> ssim_value, ssim_map = calculate_ssim_with_map(original, modified)
        >>> print(f"SSIM: {ssim_value:.4f}")
        >>> print(f"Forma del mapa: {ssim_map.shape}")
    """
    return calculate_ssim(
        reference_image,
        test_image,
        win_size=win_size,
        full=True
    )


def calculate_mssim(
    reference_image: NDArray[np.uint8],
    test_image: NDArray[np.uint8],
    scales: int = 5
) -> float:
    """
    Calcula el SSIM Multi-Escala (MS-SSIM).
    
    MS-SSIM evalúa la similitud en múltiples escalas de la imagen,
    proporcionando una evaluación más robusta de la calidad visual.
    
    Args:
        reference_image: Imagen de referencia.
        test_image: Imagen de prueba.
        scales: Número de escalas a evaluar.
    
    Returns:
        Valor de MS-SSIM (rango: [0, 1]).
    
    Examples:
        >>> import numpy as np
        >>> original = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        >>> modified = original.copy()
        >>> mssim = calculate_mssim(original, modified)
        >>> print(f"MS-SSIM: {mssim:.4f}")
    """
    if reference_image.shape != test_image.shape:
        raise ValueError("Las imágenes deben tener la misma forma")
    
    # Verificar que la imagen es suficientemente grande para múltiples escalas
    min_dim = min(reference_image.shape)
    max_scales = int(np.floor(np.log2(min_dim))) - 1
    
    if scales > max_scales:
        scales = max_scales
        print(f"Advertencia: Reduciendo escalas a {scales} debido al tamaño de imagen")
    
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])[:scales]
    weights = weights / weights.sum()
    
    mssim_value = 1.0
    
    ref = reference_image.copy().astype(np.float64)
    test = test_image.copy().astype(np.float64)
    
    for i, weight in enumerate(weights):
        ssim_val = calculate_ssim(
            ref.astype(np.uint8),
            test.astype(np.uint8),
            gaussian_weights=True
        )
        
        mssim_value *= ssim_val ** weight
        
        # Reducir resolución para la siguiente escala
        if i < len(weights) - 1:
            from skimage.transform import rescale
            ref = rescale(ref, 0.5, anti_aliasing=True, channel_axis=None)
            test = rescale(test, 0.5, anti_aliasing=True, channel_axis=None)
    
    return float(mssim_value)
