"""
Tests para el procesador CLAHE.
"""

import pytest
import numpy as np
from clahe.processor import CLAHEProcessor, apply_clahe_simple


class TestCLAHEProcessor:
    """Tests para la clase CLAHEProcessor."""
    
    def test_initialization_default(self):
        """Inicialización con parámetros por defecto."""
        processor = CLAHEProcessor()
        assert processor.rx == 8
        assert processor.ry == 8
        assert processor.clip_limit == 2.0
    
    def test_initialization_custom(self):
        """Inicialización con parámetros personalizados."""
        processor = CLAHEProcessor(rx=4, ry=4, clip_limit=3.0)
        assert processor.rx == 4
        assert processor.ry == 4
        assert processor.clip_limit == 3.0
    
    def test_invalid_rx(self):
        """Debe fallar con rx fuera de rango."""
        with pytest.raises(ValueError):
            CLAHEProcessor(rx=1)
        
        with pytest.raises(ValueError):
            CLAHEProcessor(rx=100)
    
    def test_invalid_ry(self):
        """Debe fallar con ry fuera de rango."""
        with pytest.raises(ValueError):
            CLAHEProcessor(ry=1)
        
        with pytest.raises(ValueError):
            CLAHEProcessor(ry=100)
    
    def test_invalid_clip_limit(self):
        """Debe fallar con clip_limit fuera de rango."""
        with pytest.raises(ValueError):
            CLAHEProcessor(clip_limit=0.5)
        
        with pytest.raises(ValueError):
            CLAHEProcessor(clip_limit=15.0)
    
    def test_process_basic(self):
        """Procesamiento básico de imagen."""
        processor = CLAHEProcessor()
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        
        result = processor.process(img)
        
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert np.min(result) >= 0
        assert np.max(result) <= 255
    
    def test_process_low_contrast(self):
        """Procesamiento de imagen con bajo contraste."""
        processor = CLAHEProcessor()
        # Imagen con bajo contraste
        img = np.ones((256, 256), dtype=np.uint8) * 128
        img[100:150, 100:150] = 140
        
        result = processor.process(img)
        
        # La imagen procesada debe tener mayor rango
        assert (np.max(result) - np.min(result)) >= (np.max(img) - np.min(img))
    
    def test_process_invalid_shape(self):
        """Debe fallar con imagen 3D."""
        processor = CLAHEProcessor()
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            processor.process(img)
    
    def test_process_invalid_dtype(self):
        """Debe fallar con dtype incorrecto."""
        processor = CLAHEProcessor()
        img = np.random.random((256, 256))  # float64
        
        with pytest.raises(ValueError):
            processor.process(img)
    
    def test_update_parameters(self):
        """Actualización de parámetros."""
        processor = CLAHEProcessor(rx=8, ry=8, clip_limit=2.0)
        
        processor.update_parameters(rx=4, clip_limit=3.0)
        
        assert processor.rx == 4
        assert processor.ry == 8  # No cambió
        assert processor.clip_limit == 3.0
    
    def test_get_parameters(self):
        """Obtener parámetros actuales."""
        processor = CLAHEProcessor(rx=4, ry=6, clip_limit=2.5)
        params = processor.get_parameters()
        
        assert params['rx'] == 4
        assert params['ry'] == 6
        assert params['clip_limit'] == 2.5
    
    def test_get_parameter_ranges(self):
        """Obtener rangos de parámetros."""
        ranges = CLAHEProcessor.get_parameter_ranges()
        
        assert 'rx' in ranges
        assert 'ry' in ranges
        assert 'clip_limit' in ranges
        assert ranges['rx'] == (2, 16)
    
    def test_get_default_parameters(self):
        """Obtener parámetros por defecto."""
        defaults = CLAHEProcessor.get_default_parameters()
        
        assert defaults['rx'] == 8
        assert defaults['ry'] == 8
        assert defaults['clip_limit'] == 2.0
    
    def test_process_with_mask(self):
        """Procesamiento con máscara."""
        processor = CLAHEProcessor()
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=bool)
        mask[50:200, 50:200] = True
        
        result = processor.process_with_mask(img, mask)
        
        assert result.shape == img.shape
        # Las regiones fuera de la máscara deben permanecer iguales
        assert np.array_equal(result[0:49, :], img[0:49, :])


class TestCLAHESimple:
    """Tests para la función apply_clahe_simple."""
    
    def test_simple_function(self):
        """Función auxiliar debe funcionar."""
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        result = apply_clahe_simple(img)
        
        assert result.shape == img.shape
        assert result.dtype == np.uint8
    
    def test_simple_with_params(self):
        """Función auxiliar con parámetros personalizados."""
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        result = apply_clahe_simple(img, rx=4, ry=4, clip_limit=3.0)
        
        assert result.shape == img.shape


def test_clahe_improves_contrast():
    """Test de que CLAHE mejora el contraste."""
    # Crear imagen con bajo contraste
    img = np.ones((256, 256), dtype=np.uint8) * 100
    img[50:150, 50:150] = 120
    
    processor = CLAHEProcessor()
    enhanced = processor.process(img)
    
    # El rango debe aumentar
    original_range = np.max(img) - np.min(img)
    enhanced_range = np.max(enhanced) - np.min(enhanced)
    
    assert enhanced_range > original_range


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
