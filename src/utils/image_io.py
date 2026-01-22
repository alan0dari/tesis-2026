"""
Utilidades para carga y guardado de imágenes.

Proporciona funciones para leer y escribir imágenes en diferentes formatos,
incluyendo soporte para imágenes médicas en formato DICOM.
"""

from typing import Optional, Union, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
from pathlib import Path


def load_image(
    filepath: Union[str, Path],
    as_gray: bool = True
) -> NDArray[np.uint8]:
    """
    Carga una imagen desde un archivo.
    
    Soporta formatos: PNG, JPEG, TIFF, BMP, DICOM.
    
    Args:
        filepath: Ruta al archivo de imagen.
        as_gray: Si True, convierte a escala de grises.
    
    Returns:
        Imagen como array de NumPy.
    
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato no es soportado o la imagen no se puede cargar.
    
    Examples:
        >>> image = load_image('data/ortopantomografia.png')
        >>> print(f"Forma: {image.shape}, Tipo: {image.dtype}")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"El archivo no existe: {filepath}")
    
    # Verificar si es DICOM
    if filepath.suffix.lower() in ['.dcm', '.dicom']:
        return load_dicom(filepath, as_gray=as_gray)
    
    # Cargar con OpenCV
    if as_gray:
        image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(filepath), cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {filepath}")
    
    return image


def save_image(
    image: NDArray[np.uint8],
    filepath: Union[str, Path],
    create_dirs: bool = True
) -> None:
    """
    Guarda una imagen en un archivo.
    
    Args:
        image: Imagen como array de NumPy.
        filepath: Ruta donde guardar la imagen.
        create_dirs: Si True, crea directorios si no existen.
    
    Raises:
        ValueError: Si la imagen no tiene el formato correcto.
    
    Examples:
        >>> import numpy as np
        >>> image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        >>> save_image(image, 'results/test.png')
    """
    filepath = Path(filepath)
    
    # Crear directorios si no existen
    if create_dirs:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Verificar que la imagen sea válida
    if not isinstance(image, np.ndarray):
        raise ValueError("La imagen debe ser un array de NumPy")
    
    if image.dtype != np.uint8:
        # Intentar convertir a uint8
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Guardar con OpenCV
    success = cv2.imwrite(str(filepath), image)
    
    if not success:
        raise ValueError(f"No se pudo guardar la imagen en: {filepath}")


def load_dicom(
    filepath: Union[str, Path],
    as_gray: bool = True
) -> NDArray[np.uint8]:
    """
    Carga una imagen DICOM.
    
    Args:
        filepath: Ruta al archivo DICOM.
        as_gray: Si True, asegura escala de grises.
    
    Returns:
        Imagen como array uint8.
    
    Examples:
        >>> image = load_dicom('data/scan.dcm')
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "pydicom no está instalado. Instale con: pip install pydicom"
        )
    
    filepath = Path(filepath)
    
    # Leer archivo DICOM
    dicom = pydicom.dcmread(str(filepath))
    
    # Obtener array de píxeles
    image = dicom.pixel_array
    
    # Normalizar a uint8
    image = normalize_to_uint8(image)
    
    # Convertir a escala de grises si es necesario
    if as_gray and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    return image


def normalize_to_uint8(
    image: NDArray
) -> NDArray[np.uint8]:
    """
    Normaliza una imagen a rango uint8 [0, 255].
    
    Args:
        image: Imagen a normalizar.
    
    Returns:
        Imagen normalizada como uint8.
    
    Examples:
        >>> img_float = np.random.random((100, 100))
        >>> img_uint8 = normalize_to_uint8(img_float)
    """
    # Si ya es uint8, retornar como está
    if image.dtype == np.uint8:
        return image
    
    # Normalizar a [0, 1]
    image_min = np.min(image)
    image_max = np.max(image)
    
    if image_max - image_min == 0:
        return np.zeros_like(image, dtype=np.uint8)
    
    normalized = (image - image_min) / (image_max - image_min)
    
    # Escalar a [0, 255]
    scaled = (normalized * 255).astype(np.uint8)
    
    return scaled


def load_image_batch(
    directory: Union[str, Path],
    pattern: str = "*.png",
    as_gray: bool = True,
    max_images: Optional[int] = None
) -> list:
    """
    Carga múltiples imágenes desde un directorio.
    
    Args:
        directory: Directorio con imágenes.
        pattern: Patrón glob para filtrar archivos.
        as_gray: Si True, convierte a escala de grises.
        max_images: Número máximo de imágenes a cargar.
    
    Returns:
        Lista de tuplas (filepath, image).
    
    Examples:
        >>> images = load_image_batch('data/original/', '*.png')
        >>> print(f"Cargadas {len(images)} imágenes")
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"El directorio no existe: {directory}")
    
    # Buscar archivos que coincidan con el patrón
    files = sorted(directory.glob(pattern))
    
    if max_images:
        files = files[:max_images]
    
    images = []
    
    for filepath in files:
        try:
            image = load_image(filepath, as_gray=as_gray)
            images.append((filepath, image))
        except Exception as e:
            print(f"Advertencia: No se pudo cargar {filepath}: {e}")
    
    return images


def save_image_comparison(
    original: NDArray[np.uint8],
    processed: NDArray[np.uint8],
    filepath: Union[str, Path],
    titles: Optional[Tuple[str, str]] = None
) -> None:
    """
    Guarda una comparación lado a lado de dos imágenes.
    
    Args:
        original: Imagen original.
        processed: Imagen procesada.
        filepath: Ruta donde guardar la comparación.
        titles: Tupla opcional con títulos (original, procesada).
    
    Examples:
        >>> original = load_image('original.png')
        >>> enhanced = apply_clahe_simple(original)
        >>> save_image_comparison(
        ...     original, enhanced,
        ...     'results/comparison.png',
        ...     titles=('Original', 'Mejorada')
        ... )
    """
    import matplotlib.pyplot as plt
    
    if titles is None:
        titles = ('Original', 'Procesada')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(titles[0], fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title(titles[1], fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()


def get_image_info(image: NDArray) -> dict:
    """
    Obtiene información sobre una imagen.
    
    Args:
        image: Imagen como array de NumPy.
    
    Returns:
        Diccionario con información de la imagen.
    
    Examples:
        >>> image = load_image('test.png')
        >>> info = get_image_info(image)
        >>> print(f"Tamaño: {info['shape']}, Tipo: {info['dtype']}")
    """
    return {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'size_mb': image.nbytes / (1024 ** 2)
    }
