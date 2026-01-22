"""
Procesador CLAHE (Contrast Limited Adaptive Histogram Equalization).

CLAHE es una técnica de mejora de contraste adaptativa que divide la imagen
en regiones y aplica ecualización de histograma con límite de contraste.
"""

from typing import Tuple, Optional, Union
import numpy as np
from numpy.typing import NDArray
import cv2


class CLAHEProcessor:
    """
    Procesador CLAHE para mejora de contraste en imágenes médicas.
    
    CLAHE mejora el contraste local de la imagen dividiendo en regiones
    (tiles) y aplicando ecualización de histograma con límite de contraste
    para evitar sobre-amplificación de ruido.
    
    Attributes:
        rx: Número de regiones en dirección horizontal.
        ry: Número de regiones en dirección vertical.
        clip_limit: Límite de contraste para recorte de histograma.
                   Valores típicos: 1.0 a 4.0.
    
    Examples:
        >>> import numpy as np
        >>> processor = CLAHEProcessor(rx=8, ry=8, clip_limit=2.0)
        >>> image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        >>> enhanced = processor.process(image)
        >>> print(f"Imagen original: {image.shape}, mejorada: {enhanced.shape}")
    """
    
    def __init__(
        self,
        rx: int = 8,
        ry: int = 8,
        clip_limit: float = 2.0
    ):
        """
        Inicializa el procesador CLAHE.
        
        Args:
            rx: Número de tiles en dirección horizontal (columnas).
                Valores típicos: 2 a 16.
            ry: Número de tiles en dirección vertical (filas).
                Valores típicos: 2 a 16.
            clip_limit: Límite de contraste normalizado.
                       1.0 = sin límite, valores más altos = mayor contraste.
                       Valores típicos: 1.0 a 4.0.
        
        Raises:
            ValueError: Si los parámetros están fuera de rango válido.
        """
        if rx < 2 or rx > 64:
            raise ValueError(f"rx debe estar entre 2 y 64, pero es {rx}")
        
        if ry < 2 or ry > 64:
            raise ValueError(f"ry debe estar entre 2 y 64, pero es {ry}")
        
        if clip_limit < 1.0 or clip_limit > 10.0:
            raise ValueError(
                f"clip_limit debe estar entre 1.0 y 10.0, pero es {clip_limit}"
            )
        
        self.rx = int(rx)
        self.ry = int(ry)
        self.clip_limit = float(clip_limit)
        
        # Crear objeto CLAHE de OpenCV
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.rx, self.ry)
        )
    
    def process(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Aplica CLAHE a una imagen en escala de grises.
        
        Args:
            image: Imagen en escala de grises (2D) de tipo uint8.
        
        Returns:
            Imagen mejorada con el mismo tamaño y tipo que la entrada.
        
        Raises:
            ValueError: Si la imagen no es 2D o no es uint8.
        
        Examples:
            >>> import numpy as np
            >>> processor = CLAHEProcessor(rx=8, ry=8, clip_limit=2.0)
            >>> image = np.random.randint(50, 150, (256, 256), dtype=np.uint8)
            >>> enhanced = processor.process(image)
            >>> # La imagen mejorada tiene mejor contraste local
        """
        if image.ndim != 2:
            raise ValueError(
                f"La imagen debe ser 2D (escala de grises), "
                f"pero tiene {image.ndim} dimensiones"
            )
        
        if image.dtype != np.uint8:
            raise ValueError(
                f"La imagen debe ser de tipo uint8, pero es {image.dtype}"
            )
        
        # Aplicar CLAHE
        enhanced = self._clahe.apply(image)
        
        return enhanced
    
    def process_with_mask(
        self,
        image: NDArray[np.uint8],
        mask: NDArray[np.bool_]
    ) -> NDArray[np.uint8]:
        """
        Aplica CLAHE solo en las regiones indicadas por la máscara.
        
        Útil para aplicar mejora solo en regiones de interés (ROI).
        
        Args:
            image: Imagen en escala de grises.
            mask: Máscara booleana del mismo tamaño que la imagen.
                 True indica regiones donde aplicar CLAHE.
        
        Returns:
            Imagen mejorada solo en regiones enmascaradas.
        
        Examples:
            >>> import numpy as np
            >>> processor = CLAHEProcessor()
            >>> image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
            >>> mask = np.zeros((256, 256), dtype=bool)
            >>> mask[50:200, 50:200] = True  # ROI en el centro
            >>> enhanced = processor.process_with_mask(image, mask)
        """
        if image.shape != mask.shape:
            raise ValueError(
                f"La imagen y la máscara deben tener la misma forma. "
                f"Imagen: {image.shape}, Máscara: {mask.shape}"
            )
        
        # Crear copia de la imagen
        result = image.copy()
        
        # Aplicar CLAHE a toda la imagen
        enhanced = self.process(image)
        
        # Copiar solo las regiones enmascaradas
        result[mask] = enhanced[mask]
        
        return result
    
    def update_parameters(
        self,
        rx: Optional[int] = None,
        ry: Optional[int] = None,
        clip_limit: Optional[float] = None
    ) -> None:
        """
        Actualiza los parámetros del procesador CLAHE.
        
        Args:
            rx: Nuevo valor para rx (opcional).
            ry: Nuevo valor para ry (opcional).
            clip_limit: Nuevo valor para clip_limit (opcional).
        
        Examples:
            >>> processor = CLAHEProcessor(rx=8, ry=8, clip_limit=2.0)
            >>> processor.update_parameters(clip_limit=3.0)
            >>> # Ahora el procesador usa clip_limit=3.0
        """
        if rx is not None:
            if rx < 2 or rx > 64:
                raise ValueError(f"rx debe estar entre 2 y 64, pero es {rx}")
            self.rx = int(rx)
        
        if ry is not None:
            if ry < 2 or ry > 64:
                raise ValueError(f"ry debe estar entre 2 y 64, pero es {ry}")
            self.ry = int(ry)
        
        if clip_limit is not None:
            if clip_limit < 1.0 or clip_limit > 10.0:
                raise ValueError(
                    f"clip_limit debe estar entre 1.0 y 10.0, pero es {clip_limit}"
                )
            self.clip_limit = float(clip_limit)
        
        # Recrear objeto CLAHE con nuevos parámetros
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.rx, self.ry)
        )
    
    def get_parameters(self) -> dict:
        """
        Obtiene los parámetros actuales del procesador.
        
        Returns:
            Diccionario con los parámetros rx, ry y clip_limit.
        
        Examples:
            >>> processor = CLAHEProcessor(rx=8, ry=8, clip_limit=2.0)
            >>> params = processor.get_parameters()
            >>> print(f"Parámetros: {params}")
        """
        return {
            'rx': self.rx,
            'ry': self.ry,
            'clip_limit': self.clip_limit
        }
    
    @staticmethod
    def get_parameter_ranges() -> dict:
        """
        Obtiene los rangos válidos para los parámetros de CLAHE.
        
        Útil para definir el espacio de búsqueda en optimización.
        
        Returns:
            Diccionario con rangos (min, max) para cada parámetro.
        
        Examples:
            >>> ranges = CLAHEProcessor.get_parameter_ranges()
            >>> print(f"Rango de rx: {ranges['rx']}")
            >>> print(f"Rango de ry: {ranges['ry']}")
            >>> print(f"Rango de clip_limit: {ranges['clip_limit']}")
        """
        return {
            'rx': (2, 16),
            'ry': (2, 16),
            'clip_limit': (1.0, 4.0)
        }
    
    @staticmethod
    def get_default_parameters() -> dict:
        """
        Obtiene los parámetros predeterminados recomendados.
        
        Returns:
            Diccionario con valores predeterminados.
        
        Examples:
            >>> defaults = CLAHEProcessor.get_default_parameters()
            >>> processor = CLAHEProcessor(**defaults)
        """
        return {
            'rx': 8,
            'ry': 8,
            'clip_limit': 2.0
        }
    
    def __repr__(self) -> str:
        """Representación en string del procesador."""
        return (
            f"CLAHEProcessor(rx={self.rx}, ry={self.ry}, "
            f"clip_limit={self.clip_limit})"
        )
    
    def __str__(self) -> str:
        """String descriptivo del procesador."""
        return (
            f"Procesador CLAHE:\n"
            f"  - Regiones horizontales (rx): {self.rx}\n"
            f"  - Regiones verticales (ry): {self.ry}\n"
            f"  - Límite de contraste: {self.clip_limit}"
        )


def apply_clahe_simple(
    image: NDArray[np.uint8],
    rx: int = 8,
    ry: int = 8,
    clip_limit: float = 2.0
) -> NDArray[np.uint8]:
    """
    Función auxiliar para aplicar CLAHE de forma simple.
    
    Crea un procesador CLAHE temporal y aplica la mejora.
    
    Args:
        image: Imagen en escala de grises.
        rx: Número de regiones horizontales.
        ry: Número de regiones verticales.
        clip_limit: Límite de contraste.
    
    Returns:
        Imagen mejorada.
    
    Examples:
        >>> import numpy as np
        >>> image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        >>> enhanced = apply_clahe_simple(image, rx=8, ry=8, clip_limit=2.0)
    """
    processor = CLAHEProcessor(rx=rx, ry=ry, clip_limit=clip_limit)
    return processor.process(image)
