"""
Cálculo de la entropía de Shannon para imágenes.

La entropía mide la cantidad de información contenida en una imagen.
Una mayor entropía indica mayor cantidad de información y detalles.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray


def calculate_entropy(image: NDArray[np.uint8]) -> float:
    """
    Calcula la entropía de Shannon de una imagen.
    
    La entropía se calcula usando la fórmula:
    H = -Σ(p_i × log2(p_i))
    
    donde p_i es la probabilidad del i-ésimo nivel de gris.
    
    Args:
        image: Imagen en escala de grises como array de NumPy.
               Debe ser de tipo uint8 con valores entre 0 y 255.
    
    Returns:
        Valor de entropía en bits. Un valor más alto indica mayor
        cantidad de información en la imagen.
    
    Raises:
        ValueError: Si la imagen no es 2D o no tiene el tipo correcto.
    
    Examples:
        >>> import numpy as np
        >>> # Imagen uniforme (baja entropía)
        >>> uniform_img = np.ones((100, 100), dtype=np.uint8) * 128
        >>> entropy_uniform = calculate_entropy(uniform_img)
        >>> print(f"Entropía uniforme: {entropy_uniform:.4f}")
        
        >>> # Imagen con ruido (alta entropía)
        >>> noise_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> entropy_noise = calculate_entropy(noise_img)
        >>> print(f"Entropía con ruido: {entropy_noise:.4f}")
    """
    if image.ndim != 2:
        raise ValueError(f"La imagen debe ser 2D, pero tiene {image.ndim} dimensiones")
    
    if image.dtype != np.uint8:
        raise ValueError(f"La imagen debe ser de tipo uint8, pero es {image.dtype}")
    
    # Calcular histograma normalizado (probabilidades)
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    
    # Normalizar el histograma para obtener probabilidades
    histogram = histogram.astype(np.float64)
    histogram = histogram / histogram.sum()
    
    # Eliminar bins con probabilidad cero (log de cero no está definido)
    histogram = histogram[histogram > 0]
    
    # Calcular entropía: H = -Σ(p_i × log2(p_i))
    entropy = -np.sum(histogram * np.log2(histogram))
    
    return float(entropy)


def calculate_entropy_normalized(image: NDArray[np.uint8]) -> float:
    """
    Calcula la entropía normalizada de una imagen.
    
    La entropía normalizada está en el rango [0, 1], donde 1 representa
    la máxima entropía posible (distribución uniforme de niveles de gris).
    
    Args:
        image: Imagen en escala de grises como array de NumPy.
    
    Returns:
        Entropía normalizada entre 0 y 1.
    
    Examples:
        >>> import numpy as np
        >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> entropy_norm = calculate_entropy_normalized(img)
        >>> print(f"Entropía normalizada: {entropy_norm:.4f}")
    """
    entropy = calculate_entropy(image)
    
    # La entropía máxima para una imagen de 8 bits es log2(256) = 8
    max_entropy = 8.0
    
    return entropy / max_entropy


def calculate_local_entropy(
    image: NDArray[np.uint8],
    window_size: int = 15
) -> NDArray[np.float64]:
    """
    Calcula la entropía local de una imagen usando ventanas deslizantes.
    
    La entropía local permite identificar regiones con diferente cantidad
    de información en la imagen.
    
    Args:
        image: Imagen en escala de grises como array de NumPy.
        window_size: Tamaño de la ventana para calcular entropía local.
                     Debe ser impar.
    
    Returns:
        Mapa de entropía local con el mismo tamaño que la imagen de entrada.
    
    Raises:
        ValueError: Si window_size no es impar o es demasiado grande.
    
    Examples:
        >>> import numpy as np
        >>> img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> local_entropy = calculate_local_entropy(img, window_size=15)
        >>> print(f"Forma: {local_entropy.shape}")
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size debe ser impar, pero es {window_size}")
    
    if window_size > min(image.shape):
        raise ValueError(
            f"window_size ({window_size}) es mayor que la dimensión "
            f"mínima de la imagen ({min(image.shape)})"
        )
    
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    
    # Calcular entropía local usando filtro de rango
    # disk crea un elemento estructurante circular
    radius = window_size // 2
    local_ent = entropy(image, disk(radius))
    
    return local_ent.astype(np.float64)
