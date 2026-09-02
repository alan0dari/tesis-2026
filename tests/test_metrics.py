"""
Tests para las métricas de evaluación de imágenes.
"""

import pytest
import numpy as np
from metrics.entropy import calculate_entropy, calculate_entropy_normalized
from metrics.ssim import calculate_ssim
from metrics.vif import calculate_vif
from metrics.vqi import calculate_vqi


class TestEntropy:
    """Tests para cálculo de entropía."""
    
    def test_entropy_uniform_image(self):
        """Una imagen uniforme debe tener entropía cercana a 0."""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        entropy = calculate_entropy(img)
        assert entropy == 0.0
    
    def test_entropy_random_image(self):
        """Una imagen aleatoria debe tener entropía alta."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        entropy = calculate_entropy(img)
        assert entropy > 5.0  # Entropía típica de ruido
    
    def test_entropy_normalized_range(self):
        """La entropía normalizada debe estar en [0, 1]."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        entropy_norm = calculate_entropy_normalized(img)
        assert 0.0 <= entropy_norm <= 1.0
    
    def test_entropy_invalid_input(self):
        """Debe fallar con entrada no 2D."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            calculate_entropy(img)


class TestSSIM:
    """Tests para cálculo de SSIM."""
    
    def test_ssim_identical_images(self):
        """SSIM de imágenes idénticas debe ser 1.0."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        ssim = calculate_ssim(img, img)
        assert ssim == 1.0
    
    def test_ssim_different_images(self):
        """SSIM de imágenes diferentes debe ser < 1.0."""
        img1 = np.random.randint(0, 128, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(128, 256, (100, 100), dtype=np.uint8)
        ssim = calculate_ssim(img1, img2)
        assert ssim < 1.0
    
    def test_ssim_range(self):
        """SSIM debe estar en rango válido."""
        img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        ssim = calculate_ssim(img1, img2)
        assert -1.0 <= ssim <= 1.0
    
    def test_ssim_shape_mismatch(self):
        """Debe fallar con imágenes de diferentes tamaños."""
        img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        with pytest.raises(ValueError):
            calculate_ssim(img1, img2)


class TestVIF:
    """Tests para el Índice de Fidelidad de Información Visual (VIF)."""

    def test_vif_identical_images(self):
        """VIF de imágenes idénticas debe ser 1.0."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        vif = calculate_vif(img, img)
        assert abs(vif - 1.0) < 1e-6

    def test_vif_degraded_image(self):
        """Una imagen degradada (contraste reducido + ruido) debe tener VIF < 1."""
        np.random.seed(42)
        ref = np.random.randint(30, 220, (128, 128), dtype=np.uint8)
        # Reducción de contraste y ruido: pérdida de información
        degraded = (ref.astype(np.float64) * 0.4 + 60)
        degraded += np.random.normal(0, 5, ref.shape)
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)
        vif = calculate_vif(ref, degraded)
        assert vif < 1.0

    def test_vif_positive(self):
        """VIF debe ser no negativo para imágenes con contenido."""
        np.random.seed(0)
        ref = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        vif = calculate_vif(ref, img)
        assert vif >= 0.0

    def test_vif_shape_mismatch(self):
        """Debe fallar con imágenes de diferentes tamaños."""
        img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        with pytest.raises(ValueError):
            calculate_vif(img1, img2)

    def test_vif_invalid_dimensions(self):
        """Debe fallar con entrada 3D."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            calculate_vif(img, img)

    def test_vif_constant_reference(self):
        """Referencia constante (sin información) debe dar VIF neutro sin dividir por cero."""
        ref = np.full((64, 64), 128, dtype=np.uint8)
        img = np.full((64, 64), 128, dtype=np.uint8)
        vif = calculate_vif(ref, img)
        assert vif == 1.0


class TestVQI:
    """Tests para cálculo de VQI (métrica histórica, ya no usada en el pipeline)."""
    
    def test_vqi_basic(self):
        """VQI debe retornar un valor numérico."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        vqi = calculate_vqi(img)
        assert isinstance(vqi, (int, float))
        assert vqi >= 0
    
    def test_vqi_with_reference(self):
        """VQI con referencia debe funcionar."""
        img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        vqi = calculate_vqi(img1, reference_image=img2)
        assert isinstance(vqi, (int, float))
    
    def test_vqi_monotonic_with_contrast(self):
        """A mayor contraste local, mayor VQI (comparación relativa)."""
        np.random.seed(42)
        base = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        # Versión de bajo contraste de la misma imagen
        low_contrast = (base.astype(np.float64) * 0.3 + 90).astype(np.uint8)
        assert calculate_vqi(base) > calculate_vqi(low_contrast)
    
    def test_vqi_invalid_dimensions(self):
        """Debe fallar con entrada 3D."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            calculate_vqi(img)


def test_metrics_consistency():
    """Test de consistencia entre métricas."""
    # Crear imagen base y versiones mejoradas
    np.random.seed(42)
    original = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    
    # Estiramiento lineal SIN saturación: mapea [50, 200] a [0, 255].
    # Un mapeo biyectivo de intensidades preserva el histograma (y por
    # tanto la entropía); con saturación (clip) se perdería información.
    enhanced = ((original.astype(np.float32) - 50) * (255.0 / 150.0))
    enhanced = np.round(enhanced).astype(np.uint8)

    # Calcular métricas
    entropy_orig = calculate_entropy(original)
    entropy_enh = calculate_entropy(enhanced)

    # La entropía debe preservarse (mapeo biyectivo) o variar mínimamente
    assert abs(entropy_enh - entropy_orig) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
