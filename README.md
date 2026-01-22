# Framework de Mejora de Imágenes Radiográficas con Optimización Multiobjetivo y Métodos de Decisión Multicriterio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Descripción

Este repositorio contiene la implementación completa de una tesis de grado sobre:

**"Framework basado en aplicación de métodos de decisión multicriterio para selección de imagen radiográfica mejorada con optimización multiobjetivo"**

El framework permite mejorar el contraste de ortopantomografías (radiografías dentales panorámicas) usando CLAHE (Contrast Limited Adaptive Histogram Equalization), optimizar los parámetros mediante SMPSO (Speed-constrained Multi-objective Particle Swarm Optimization), y seleccionar la mejor imagen mejorada usando 8 diferentes métodos de decisión multicriterio.

## Características Principales

- **Mejora de Contraste**: Implementación de CLAHE con parámetros ajustables (Rx, Ry, C)
- **Optimización Multiobjetivo**: SMPSO para encontrar parámetros óptimos de CLAHE
- **Métricas de Evaluación**: 
  - Entropía de Shannon
  - SSIM (Structural Similarity Index)
  - VQI (Visual Quality Index)
- **Frente de Pareto 3D**: Visualización de soluciones óptimas en espacio 3D
- **8 Métodos MCDM**: Selección de la mejor solución usando múltiples criterios

## Estructura del Proyecto

```
tesis-2026/
├── src/                        # Código fuente principal
│   ├── clahe/                  # Módulo de procesamiento CLAHE
│   │   └── processor.py
│   ├── metrics/                # Métricas de evaluación
│   │   ├── entropy.py
│   │   ├── ssim.py
│   │   └── vqi.py
│   ├── optimization/           # Optimización multiobjetivo
│   │   ├── smpso.py
│   │   └── pareto.py
│   ├── mcdm/                   # Métodos de decisión multicriterio
│   │   ├── base.py
│   │   ├── smarter.py
│   │   ├── topsis.py
│   │   ├── bellman_zadeh.py
│   │   ├── promethee_ii.py
│   │   ├── gra.py
│   │   ├── vikor.py
│   │   ├── codas.py
│   │   └── mabac.py
│   └── utils/                  # Utilidades
│       ├── image_io.py
│       ├── normalization.py
│       └── visualization.py
├── docs/                       # Documentación
│   ├── libro/                  # Libro de tesis en LaTeX
│   └── articulo/               # Artículo académico
├── data/                       # Directorio para datasets
├── experiments/                # Notebooks de experimentación
├── results/                    # Resultados de experimentos
└── tests/                      # Tests unitarios
```

## Requisitos

### Software
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Librerías
Las dependencias se especifican en `requirements.txt`:
- numpy>=1.21.0
- scipy>=1.7.0
- scikit-image>=0.18.0
- opencv-python>=4.5.0
- matplotlib>=3.4.0
- pandas>=1.3.0
- jupyter>=1.0.0
- pytest>=6.2.0

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/alan0dari/tesis-2026.git
cd tesis-2026
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Instalar el paquete en modo desarrollo
```bash
pip install -e .
```

## Uso Básico

### 1. Procesamiento CLAHE
```python
from clahe.processor import CLAHEProcessor
from utils.image_io import load_image, save_image

# Cargar imagen
image = load_image('data/ortopantomografia.png')

# Crear procesador CLAHE
processor = CLAHEProcessor(rx=8, ry=8, clip_limit=2.0)

# Aplicar CLAHE
enhanced_image = processor.process(image)

# Guardar resultado
save_image(enhanced_image, 'results/enhanced.png')
```

### 2. Calcular Métricas
```python
from metrics.entropy import calculate_entropy
from metrics.ssim import calculate_ssim
from metrics.vqi import calculate_vqi

# Calcular métricas
entropy_val = calculate_entropy(enhanced_image)
ssim_val = calculate_ssim(image, enhanced_image)
vqi_val = calculate_vqi(enhanced_image)

print(f"Entropía: {entropy_val:.4f}")
print(f"SSIM: {ssim_val:.4f}")
print(f"VQI: {vqi_val:.4f}")
```

### 3. Optimización con SMPSO
```python
from optimization.smpso import SMPSO
from optimization.pareto import build_pareto_front

# Definir función objetivo
def objective_function(params):
    rx, ry, clip_limit = params
    processor = CLAHEProcessor(rx, ry, clip_limit)
    enhanced = processor.process(image)
    
    entropy = calculate_entropy(enhanced)
    ssim = calculate_ssim(image, enhanced)
    vqi = calculate_vqi(enhanced)
    
    return [entropy, ssim, vqi]

# Ejecutar SMPSO
optimizer = SMPSO(
    n_particles=30,
    n_iterations=100,
    bounds=[(2, 16), (2, 16), (1.0, 4.0)]
)

solutions = optimizer.optimize(objective_function)
pareto_front = build_pareto_front(solutions)
```

### 4. Aplicar Métodos MCDM
```python
from mcdm.topsis import TOPSIS
from mcdm.smarter import SMARTER
from mcdm.vikor import VIKOR

# Preparar matriz de decisión
decision_matrix = [[sol.objectives for sol in pareto_front]]

# Aplicar TOPSIS
topsis = TOPSIS()
best_solution_topsis = topsis.select(decision_matrix, weights=[0.33, 0.33, 0.34])

# Aplicar SMARTER
smarter = SMARTER()
best_solution_smarter = smarter.select(decision_matrix)

# Aplicar VIKOR
vikor = VIKOR()
best_solution_vikor = vikor.select(decision_matrix, weights=[0.33, 0.33, 0.34])
```

## Componentes del Framework

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
Técnica de mejora de contraste adaptativa que divide la imagen en regiones y aplica ecualización de histograma con límite de contraste para evitar sobre-amplificación de ruido.

**Parámetros:**
- `Rx, Ry`: Tamaño de la región contextual
- `C`: Límite de contraste (clip limit)

### Métricas de Evaluación

#### 1. Entropía de Shannon
Mide la cantidad de información en la imagen:
```
H = -Σ(p_i × log2(p_i))
```
Donde p_i es la probabilidad del i-ésimo nivel de gris.

#### 2. SSIM (Structural Similarity Index)
Evalúa la similitud estructural entre imagen original y mejorada, considerando luminancia, contraste y estructura.

#### 3. VQI (Visual Quality Index)
Índice de calidad visual que evalúa la percepción humana de la imagen.

### SMPSO (Speed-constrained Multi-objective PSO)
Algoritmo de optimización multiobjetivo basado en enjambre de partículas con restricción de velocidad, diseñado para encontrar el Frente de Pareto de soluciones óptimas.

### Métodos de Decisión Multicriterio (MCDM)

El framework implementa 8 métodos MCDM para seleccionar la mejor solución del Frente de Pareto:

1. **SMARTER**: Simple Multi-Attribute Rating Technique using Exploiting Ranks
   - Utiliza pesos automáticos basados en rankings
   - Función de utilidad aditiva

2. **TOPSIS**: Technique for Order Preference by Similarity to Ideal Solution
   - Selecciona la alternativa más cercana a la solución ideal positiva
   - Y más lejana de la solución ideal negativa

3. **Bellman-Zadeh**: Método de decisión difusa
   - Basado en intersección de conjuntos difusos
   - Utiliza funciones de pertenencia

4. **PROMETHEE II**: Preference Ranking Organization Method for Enrichment Evaluations
   - Flujos de preferencia netos
   - Función de preferencia gaussiana

5. **GRA**: Grey Relational Analysis
   - Análisis relacional de sistemas grises
   - Coeficiente de relación gris

6. **VIKOR**: VIseKriterijumska Optimizacija I Kompromisno Resenje
   - Método de compromiso multicriterio
   - Ranking basado en cercanía a solución ideal

7. **CODAS**: COmbinative Distance-based ASsessment
   - Combina distancia Euclidiana y Taxicab
   - Evaluación basada en índice de evaluación

8. **MABAC**: Multi-Attributive Border Approximation area Comparison
   - Aproximación al área de borde
   - Distancia a funciones de aproximación de borde

## Ejecutar Experimentos

### Notebook de Ejemplo
```bash
jupyter notebook experiments/ejemplo_uso.ipynb
```

### Tests
```bash
pytest tests/
```

## Estructura de Documentación

El directorio `docs/` contiene:
- **libro/**: Tesis completa en formato LaTeX
  - Capítulo 1: Introducción
  - Capítulo 2: Marco Teórico - Imágenes Médicas
  - Capítulo 3: Marco Teórico - Técnicas
  - Capítulo 4: Metodología Propuesta
  - Capítulo 5: Experimentación y Resultados
  - Capítulo 6: Conclusiones
- **articulo/**: Artículo académico

## Dataset

Las ortopantomografías utilizadas en este proyecto deben colocarse en el directorio `data/`. Consulte `data/README.md` para instrucciones sobre cómo obtener y preparar el dataset.

## Resultados

Los resultados de los experimentos se guardan en el directorio `results/`:
- Imágenes mejoradas
- Frentes de Pareto
- Selecciones de métodos MCDM
- Métricas calculadas

## Referencias

### Mejora de Imágenes
- Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. Graphics Gems IV.

### Optimización Multiobjetivo
- Nebro, A. J., et al. (2009). SMPSO: A new PSO-based metaheuristic for multi-objective optimization.

### Métodos MCDM
- Hwang, C. L., & Yoon, K. (1981). Multiple Attribute Decision Making: Methods and Applications.
- Brans, J. P., & Vincke, P. (1985). PROMETHEE method.
- Opricovic, S., & Tzeng, G. H. (2004). VIKOR method.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

## Autor

**alan0dari**

## Contribuciones

Este es un proyecto de tesis. Para consultas o sugerencias, por favor abra un issue en el repositorio.

## Citar este trabajo

Si utiliza este framework en su investigación, por favor cite:

```bibtex
@mastersthesis{tesis2026,
  author = {alan0dari},
  title = {Framework basado en aplicación de métodos de decisión multicriterio para selección de imagen radiográfica mejorada con optimización multiobjetivo},
  school = {[Universidad]},
  year = {2026},
  type = {Tesis de Maestría}
}
```
