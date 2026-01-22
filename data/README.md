# Dataset de Ortopantomografías

## Descripción

Este directorio debe contener las ortopantomografías (radiografías dentales panorámicas) utilizadas en los experimentos.

## Formato de Imágenes

- **Formato**: PNG, JPEG, TIFF, o DICOM
- **Resolución recomendada**: Al menos 1000x500 píxeles
- **Escala de grises**: 8 o 16 bits por píxel

## Organización

Se recomienda organizar las imágenes de la siguiente manera:

```
data/
├── original/          # Imágenes originales sin procesar
│   ├── paciente_001.png
│   ├── paciente_002.png
│   └── ...
├── test/              # Conjunto de prueba
│   ├── test_001.png
│   └── ...
└── validation/        # Conjunto de validación
    ├── val_001.png
    └── ...
```

## Fuentes de Datos

### Datasets Públicos

1. **Tufts Dental Database**
   - URL: http://tdd.ece.tufts.edu/
   - Contiene radiografías dentales anotadas

2. **Open Dental**
   - URL: https://www.opendental.com/
   - Software de gestión dental con imágenes de ejemplo

3. **Kaggle Dental Datasets**
   - URL: https://www.kaggle.com/
   - Buscar: "panoramic dental x-ray" o "orthopantomogram"

### Consideraciones Éticas

- Todas las imágenes deben estar anonimizadas (sin información del paciente)
- Obtener consentimiento informado si se utilizan imágenes clínicas
- Cumplir con las regulaciones de privacidad de datos médicos (HIPAA, GDPR)

## Preprocesamiento

Antes de usar las imágenes con el framework:

1. **Conversión a escala de grises** (si están en color)
2. **Normalización de tamaño** (opcional)
3. **Eliminación de metadatos** del paciente
4. **Verificación de calidad** de la imagen

## Ejemplo de Carga

```python
from utils.image_io import load_image

# Cargar imagen
image = load_image('data/original/paciente_001.png')
print(f"Forma de la imagen: {image.shape}")
print(f"Tipo de datos: {image.dtype}")
```

## Nota Importante

**No incluir imágenes médicas reales en el repositorio de Git.** El archivo `.gitignore` está configurado para excluir archivos de imagen de este directorio.

Para compartir resultados, utilizar:
- Imágenes sintéticas o de ejemplo
- Enlaces a datasets públicos
- Estadísticas y métricas agregadas
