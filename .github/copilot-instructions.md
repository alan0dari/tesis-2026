# Instrucciones para GitHub Copilot - Proyecto de Tesis

## Contexto del Proyecto

Este repositorio contiene la implementación de una tesis de grado titulada:

**"Framework basado en aplicación de métodos de decisión multicriterio para selección de imagen radiográfica mejorada con optimización multiobjetivo"**

### Dominio
- Procesamiento de imágenes médicas (ortopantomografías/radiografías dentales)
- Optimización multiobjetivo (SMPSO - Speed-constrained Multi-objective PSO)
- Métodos de decisión multicriterio (MCDM)
- Mejora de contraste con CLAHE

### Objetivos del Framework
1. Mejorar contraste de radiografías dentales usando CLAHE con parámetros optimizados
2. Evaluar calidad con 3 métricas: Entropía, SSIM y VQI
3. Generar Frente de Pareto 3D con soluciones óptimas
4. Aplicar 8 métodos MCDM para seleccionar la mejor imagen mejorada

---

## Convenciones de Código

### Idioma
- **Código**: Variables, funciones y clases en inglés
- **Documentación**: Docstrings y comentarios en español
- **Commits**: Mensajes en español
- **README y docs**: En español

### Estilo Python
```python
# Usar type hints siempre
def calcular_entropia(imagen: np.ndarray) -> float:
    """
    Calcula la entropía de Shannon de una imagen en escala de grises.
    
    La entropía mide el nivel de incertidumbre/información de la imagen.
    Valores altos indican mayor detalle, pero también posible ruido.
    
    Args:
        imagen: Matriz numpy de la imagen en escala de grises (0-255)
    
    Returns:
        Valor de entropía en bits, rango [0, log2(256)]
    
    Ejemplo:
        >>> img = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)
        >>> h = calcular_entropia(img)
        >>> print(f"Entropía: {h:.4f} bits")
    """
    pass
```

### Nomenclatura
| Tipo | Convención | Ejemplo |
|------|------------|---------|
| Clases | PascalCase | `CLAHEProcessor`, `ParetoFront` |
| Funciones | snake_case | `calcular_ssim`, `aplicar_clahe` |
| Constantes | UPPER_SNAKE | `MAX_ITERACIONES`, `NIVELES_GRIS` |
| Variables | snake_case | `imagen_mejorada`, `frente_pareto` |
| Archivos | snake_case | `bellman_zadeh.py`, `promethee_ii.py` |

---

## Estructura del Proyecto

```
src/
├── clahe/          # Mejora de contraste CLAHE
├── metrics/        # Entropía, SSIM, VQI
├── optimization/   # SMPSO y Frente de Pareto
├── mcdm/           # 8 métodos de decisión multicriterio
└── utils/          # Normalización, visualización, I/O
```

---

## Ecuaciones Clave

### Entropía (Shannon)
```
H = -Σ(p_i × log₂(p_i))  para i = 0 hasta L-1

donde:
- p_i = probabilidad del nivel de gris i
- L = 256 (niveles de gris en imagen de 8 bits)
- H ∈ [0, log₂(L)] = [0, 8] bits
```

### SSIM (Structural Similarity Index)
```
SSIM(A,B) = [(2μ_A μ_B + C₁)(2σ_AB + C₂)] / [(μ_A² + μ_B² + C₁)(σ_A² + σ_B² + C₂)]

donde:
- μ = media de intensidades
- σ² = varianza
- σ_AB = covarianza
- C₁, C₂ = constantes de estabilización
- SSIM ∈ [0, 1], donde 1 = imágenes idénticas
```

### Dominancia de Pareto
```
x_a ≻ x_b  sí y solo sí:
- f_i(x_a) ≥ f_i(x_b)  ∀i  (mejor o igual en todos los objetivos)
- f_i(x_a) > f_i(x_b)  ∃i  (estrictamente mejor en al menos uno)
```

### Velocidad SMPSO
```
v_i = ω × v_{i-1} + c₁ × r₁ × (x_p - x_i) + c₂ × r₂ × (x_g - x_i)

Restricción:
v_i = clamp(v_i, -delta, +delta)
delta = (param_max - param_min) / 2
```

---

## Métodos MCDM Implementados

| Método | Archivo | Tipo | Ecuación de Selección |
|--------|---------|------|----------------------|
| SMARTER | `smarter.py` | Utilidad | `x_s = argmax(Σ w_i × r_ji)` |
| TOPSIS | `topsis.py` | Distancia | `x_s = argmax(ED⁻ / (ED⁺ + ED⁻))` |
| Bellman-Zadeh | `bellman_zadeh.py` | Difuso | `x_s = argmax(min(μ_ji))` |
| PROMETHEE II | `promethee_ii.py` | Flujo | `x_s = argmax(φ_saliente - φ_entrante)` |
| GRA | `gra.py` | Relacional | `x_s = argmax(Σ w_i × ξ_ji)` |
| VIKOR | `vikor.py` | Compromiso | `x_s = argmin(Q)` |
| CODAS | `codas.py` | Distancia combinada | `x_s = argmax(Σ h_jk)` |
| MABAC | `mabac.py` | Aproximación al borde | `x_s = argmax(Σ q_ji)` |

---

## Parámetros de CLAHE

```python
# Parámetros a optimizar
R_x: int      # Filas de la región contextual (2 a 64)
R_y: int      # Columnas de la región contextual (2 a 64)
C: float      # Clip limit para contraste (0.0 a 1.0)

# Partícula en SMPSO
particula = (R_x, R_y, C)
```

---

## Criterios para MCDM

| Criterio | Tipo | Peso por defecto | Descripción |
|----------|------|------------------|-------------|
| Entropía | Beneficio (maximizar) | 0.40 | Mayor detalle en la imagen |
| SSIM | Beneficio (maximizar) | 0.35 | Preservación de estructura |
| VQI | Beneficio (maximizar) | 0.25 | Calidad visual percibida |

**Nota**: Los pesos pueden ajustarse según preferencias del profesional médico.

---

## Preferencias de Copilot

### Al generar código
1. Incluir docstrings completos en español
2. Usar type hints en todas las funciones
3. Agregar ejemplos de uso en docstrings
4. Incluir validación de parámetros de entrada
5. Manejar excepciones con mensajes descriptivos

### Al generar tests
1. Usar pytest como framework
2. Incluir casos límite y casos normales
3. Usar fixtures para imágenes de prueba
4. Documentar qué se está probando

### Al generar documentación LaTeX
1. Usar español formal académico
2. Numerar ecuaciones importantes
3. Incluir referencias bibliográficas
4. Usar paquetes: amsmath, amssymb, graphicx, hyperref

---

## Librerías Principales

```python
import numpy as np              # Operaciones matriciales
import cv2                      # Procesamiento de imágenes, CLAHE
from skimage.metrics import structural_similarity  # SSIM
import matplotlib.pyplot as plt # Visualización
from mpl_toolkits.mplot3d import Axes3D  # Pareto 3D
import pandas as pd             # Manejo de datos tabulares
```

---

## Ejemplos de Uso Esperado

### Flujo completo del framework
```python
from src.clahe import CLAHEProcessor
from src.metrics import calcular_entropia, calcular_ssim, calcular_vqi
from src.optimization import SMPSO, ParetoFront
from src.mcdm import TOPSIS, SMARTER, VIKOR

# 1. Cargar imagen
imagen_original = cv2.imread('radiografia.png', cv2.IMREAD_GRAYSCALE)

# 2. Configurar y ejecutar SMPSO
smpso = SMPSO(
    imagen=imagen_original,
    n_particulas=100,
    max_iteraciones=100,
    funciones_objetivo=[calcular_entropia, calcular_ssim, calcular_vqi]
)
frente_pareto = smpso.ejecutar()

# 3. Aplicar método MCDM
topsis = TOPSIS(
    matriz_decision=frente_pareto.matriz_decision,
    pesos=[0.4, 0.35, 0.25],
    tipos_criterio=['beneficio', 'beneficio', 'beneficio']
)
mejor_solucion = topsis.seleccionar()

# 4. Obtener imagen mejorada
clahe = CLAHEProcessor(*mejor_solucion.parametros)
imagen_mejorada = clahe.aplicar(imagen_original)
```

---

## Notas Adicionales

- El dataset de ortopantomografías es público y se encuentra referenciado en el README
- Las imágenes son en escala de grises de 8 bits (0-255)
- El Frente de Pareto es 3D (Entropía vs SSIM vs VQI)
- Los métodos MCDM reciben una matriz de decisión normalizada
- La validación final la realiza un profesional médico
