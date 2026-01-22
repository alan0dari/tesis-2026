"""
Tests para las métricas de evaluación de imágenes.
"""

import pytest
import numpy as np
from metrics.entropy import calculate_entropy, calculate_entropy_normalized
from metrics.ssim import calculate_ssim
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


class TestVQI:
    """Tests para cálculo de VQI."""
    
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
    
    def test_vqi_high_contrast(self):
        """Imagen con buen contraste debe tener VQI alto."""
        # Crear imagen con buen contraste
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 255
        vqi = calculate_vqi(img)
        assert vqi > 50  # Umbral arbitrario
    
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
    
    # Versión con más contraste
    enhanced = np.clip(original.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    
    # Calcular métricas
    entropy_orig = calculate_entropy(original)
    entropy_enh = calculate_entropy(enhanced)
    
    # La imagen mejorada debería tener mayor entropía
    assert entropy_enh >= entropy_orig or abs(entropy_enh - entropy_orig) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
