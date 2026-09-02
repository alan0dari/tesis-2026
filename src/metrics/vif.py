"""
Cálculo del Índice de Fidelidad de Información Visual (VIF - Visual Information Fidelity).

VIF (Sheikh & Bovik, 2006) cuantifica la fidelidad de información visual entre
una imagen de referencia y una imagen distorsionada/procesada, modelando:
- La imagen de referencia con estadísticas de escenas naturales (GSM).
- El canal de distorsión como ganancia + ruido aditivo gaussiano.
- El sistema visual humano (HVS) como ruido aditivo gaussiano.

VIF = I(C; F | s) / I(C; E | s), la razón entre la información mutua que el
HVS puede extraer de la imagen procesada y la que puede extraer de la referencia.

Esta implementación corresponde a la versión en dominio de píxeles (VIFP,
multi-escala con 4 niveles), equivalente al vifp_mscale.m de los autores.

Propiedad relevante para mejora de contraste: a diferencia de SSIM, VIF puede
superar 1.0 cuando la imagen procesada contiene MÁS información visual que la
referencia (p. ej., tras un realce de contraste efectivo), lo que lo hace
especialmente adecuado como función objetivo en este framework.

Referencia:
    H. R. Sheikh and A. C. Bovik, "Image information and visual quality,"
    IEEE Transactions on Image Processing, vol. 15, no. 2, pp. 430-444, 2006.
    doi: 10.1109/TIP.2005.859378
"""

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


def calculate_vif(
    reference: NDArray,
    image: NDArray,
    sigma_nsq: float = 2.0
) -> float:
    """
    Calcula el VIF (versión en dominio de píxeles, multi-escala) entre
    una imagen de referencia y una imagen procesada.

    Args:
        reference: Imagen de referencia en escala de grises (uint8 o float).
        image: Imagen procesada/distorsionada, misma forma que la referencia.
        sigma_nsq: Varianza del ruido del HVS (valor estándar: 2.0).

    Returns:
        Valor de VIF. 1.0 = misma información que la referencia;
        < 1.0 = pérdida de información; > 1.0 = ganancia de información
        visual (típico en realce de contraste efectivo).

    Raises:
        ValueError: Si las imágenes no son 2D o difieren en forma.

    Examples:
        >>> import numpy as np
        >>> img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        >>> vif_identical = calculate_vif(img, img)
        >>> round(vif_identical, 2)
        1.0
    """
    if reference.ndim != 2 or image.ndim != 2:
        raise ValueError("Las imágenes deben ser 2D (escala de grises)")
    if reference.shape != image.shape:
        raise ValueError(
            f"Las imágenes deben tener la misma forma: "
            f"{reference.shape} vs {image.shape}"
        )

    ref = reference.astype(np.float64)
    dist = image.astype(np.float64)

    eps = 1e-10
    num = 0.0
    den = 0.0

    for scale in range(1, 5):
        # Tamaño de ventana gaussiana según la escala: 17, 9, 5, 3
        n = 2 ** (4 - scale + 1) + 1
        sd = n / 5.0

        if scale > 1:
            # Filtrar y submuestrear por 2 antes de cada escala superior
            ref = _gaussian_filter(ref, sd, n)[::2, ::2]
            dist = _gaussian_filter(dist, sd, n)[::2, ::2]

        # Estadísticos locales
        mu1 = _gaussian_filter(ref, sd, n)
        mu2 = _gaussian_filter(dist, sd, n)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = _gaussian_filter(ref * ref, sd, n) - mu1_sq
        sigma2_sq = _gaussian_filter(dist * dist, sd, n) - mu2_sq
        sigma12 = _gaussian_filter(ref * dist, sd, n) - mu1_mu2

        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)

        # Ganancia del canal de distorsión y varianza del ruido
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        # Casos degenerados (regiones sin varianza)
        g = np.where(sigma1_sq < eps, 0.0, g)
        sv_sq = np.where(sigma1_sq < eps, sigma2_sq, sv_sq)
        sigma1_sq = np.where(sigma1_sq < eps, 0.0, sigma1_sq)

        sv_sq = np.where(sigma2_sq < eps, 0.0, sv_sq)
        g = np.where(sigma2_sq < eps, 0.0, g)

        sv_sq = np.where(g < 0.0, sigma2_sq, sv_sq)
        g = np.maximum(g, 0.0)
        sv_sq = np.maximum(sv_sq, eps)

        # Información mutua acumulada por escala
        num += np.sum(np.log10(1.0 + (g * g) * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq))

    if den == 0.0:
        # Referencia sin información (imagen constante): definir VIF neutro
        return 1.0

    return float(num / den)


def _gaussian_filter(
    image: NDArray[np.float64],
    sd: float,
    size: int
) -> NDArray[np.float64]:
    """
    Filtro gaussiano con ventana truncada de tamaño fijo, equivalente a
    fspecial('gaussian', size, sd) + filter2 de MATLAB.
    """
    # truncate tal que radius = (size - 1) / 2
    radius = (size - 1) / 2.0
    truncate = radius / sd
    return ndimage.gaussian_filter(image, sigma=sd, truncate=truncate, mode='nearest')
