"""
Utilidades para visualización de resultados.

Proporciona funciones para visualizar imágenes, métricas, Frentes de Pareto,
y selecciones de métodos MCDM.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches


def visualize_clahe_comparison(
    original: NDArray[np.uint8],
    enhanced: NDArray[np.uint8],
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Comparación CLAHE",
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza comparación entre imagen original y mejorada con CLAHE.
    
    Args:
        original: Imagen original.
        enhanced: Imagen mejorada con CLAHE.
        metrics: Diccionario opcional con métricas calculadas.
        title: Título de la figura.
        save_path: Ruta para guardar la imagen (opcional).
    
    Examples:
        >>> original = load_image('original.png')
        >>> enhanced = apply_clahe_simple(original)
        >>> metrics = {'Entropía': 7.5, 'SSIM': 0.95, 'VQI': 85.3}
        >>> visualize_clahe_comparison(original, enhanced, metrics)
    """
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(2, 3, figure=fig)
    
    # Imagen original
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Imagen Original', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Imagen mejorada
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.imshow(enhanced, cmap='gray')
    ax2.set_title('Imagen Mejorada (CLAHE)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Histogramas
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(original.ravel(), bins=256, range=(0, 256), 
             alpha=0.7, color='blue', label='Original')
    ax3.set_title('Histograma Original', fontsize=12)
    ax3.set_xlabel('Intensidad')
    ax3.set_ylabel('Frecuencia')
    ax3.legend()
    
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(enhanced.ravel(), bins=256, range=(0, 256), 
             alpha=0.7, color='red', label='Mejorada')
    ax4.set_title('Histograma Mejorado', fontsize=12)
    ax4.set_xlabel('Intensidad')
    ax4.set_ylabel('Frecuencia')
    ax4.legend()
    
    # Mostrar métricas si están disponibles
    if metrics:
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        fig.text(0.72, 0.5, metrics_text, fontsize=11, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()


def visualize_pareto_solutions(
    pareto_front: List[Dict],
    all_solutions: Optional[List[Dict]] = None,
    selected_solutions: Optional[Dict[str, int]] = None,
    objective_names: Optional[List[str]] = None,
    title: str = "Frente de Pareto 3D",
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza el Frente de Pareto en 3D con soluciones seleccionadas por MCDM.
    
    Args:
        pareto_front: Lista de soluciones del Frente de Pareto.
        all_solutions: Todas las soluciones evaluadas (opcional).
        selected_solutions: Dict con nombre_método: índice_solución.
        objective_names: Nombres de los 3 objetivos.
        title: Título de la figura.
        save_path: Ruta para guardar la imagen.
    
    Examples:
        >>> pareto = [{'objectives': np.array([i, 5-i, i**2])} for i in range(6)]
        >>> selected = {'TOPSIS': 2, 'VIKOR': 3}
        >>> visualize_pareto_solutions(pareto, selected_solutions=selected)
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    if not pareto_front:
        print("El Frente de Pareto está vacío")
        return
    
    if objective_names is None:
        objective_names = ['Entropía', 'SSIM', 'VQI']
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Graficar todas las soluciones si se proporcionan
    if all_solutions:
        all_obj = np.array([s['objectives'] for s in all_solutions])
        ax.scatter(
            all_obj[:, 0], all_obj[:, 1], all_obj[:, 2],
            c='lightgray', alpha=0.3, s=20,
            label='Todas las soluciones'
        )
    
    # Graficar Frente de Pareto
    pareto_obj = np.array([s['objectives'] for s in pareto_front])
    ax.scatter(
        pareto_obj[:, 0], pareto_obj[:, 1], pareto_obj[:, 2],
        c='blue', s=100, marker='o', alpha=0.6,
        label='Frente de Pareto', zorder=5
    )
    
    # Graficar soluciones seleccionadas por MCDM
    if selected_solutions:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_solutions)))
        
        for (method_name, sol_idx), color in zip(selected_solutions.items(), colors):
            if sol_idx < len(pareto_front):
                obj = pareto_front[sol_idx]['objectives']
                ax.scatter(
                    obj[0], obj[1], obj[2],
                    c=[color], s=300, marker='*',
                    edgecolors='black', linewidths=2,
                    label=f'{method_name}', zorder=10
                )
    
    ax.set_xlabel(objective_names[0], fontsize=12)
    ax.set_ylabel(objective_names[1], fontsize=12)
    ax.set_zlabel(objective_names[2], fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    
    # Rotar vista para mejor visualización
    ax.view_init(elev=20, azim=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()


def visualize_mcdm_rankings(
    rankings_dict: Dict[str, NDArray[np.float64]],
    pareto_front: List[Dict],
    title: str = "Rankings de Métodos MCDM",
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza los rankings de múltiples métodos MCDM.
    
    Args:
        rankings_dict: Dict con nombre_método: array_rankings.
        pareto_front: Lista de soluciones del Frente de Pareto.
        title: Título de la figura.
        save_path: Ruta para guardar la imagen.
    
    Examples:
        >>> rankings = {
        ...     'TOPSIS': np.array([0.8, 0.6, 0.9, 0.7]),
        ...     'VIKOR': np.array([0.7, 0.85, 0.6, 0.75])
        ... }
        >>> visualize_mcdm_rankings(rankings, pareto_front)
    """
    n_methods = len(rankings_dict)
    n_solutions = len(pareto_front)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (method_name, rankings) in zip(axes, rankings_dict.items()):
        # Normalizar rankings a [0, 1] para visualización
        rankings_norm = (rankings - np.min(rankings)) / (np.max(rankings) - np.min(rankings) + 1e-10)
        
        # Crear gráfico de barras
        x = np.arange(n_solutions)
        bars = ax.bar(x, rankings_norm, color='steelblue', alpha=0.7)
        
        # Resaltar la mejor solución
        best_idx = np.argmax(rankings)
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(1.0)
        
        ax.set_xlabel('Solución del Frente de Pareto', fontsize=11)
        ax.set_ylabel('Ranking Normalizado', fontsize=11)
        ax.set_title(method_name, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{i}' for i in range(n_solutions)])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()


def visualize_metrics_evolution(
    iterations: List[int],
    metrics_history: Dict[str, List[float]],
    title: str = "Evolución de Métricas durante Optimización",
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza la evolución de métricas durante la optimización.
    
    Args:
        iterations: Lista de números de iteración.
        metrics_history: Dict con nombre_métrica: lista_valores.
        title: Título de la figura.
        save_path: Ruta para guardar la imagen.
    
    Examples:
        >>> iterations = list(range(100))
        >>> history = {
        ...     'Entropía': [7.0 + i*0.01 for i in iterations],
        ...     'SSIM': [0.85 + i*0.001 for i in iterations]
        ... }
        >>> visualize_metrics_evolution(iterations, history)
    """
    fig, axes = plt.subplots(len(metrics_history), 1, 
                            figsize=(10, 4*len(metrics_history)))
    
    if len(metrics_history) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_history)))
    
    for ax, ((metric_name, values), color) in zip(axes, zip(metrics_history.items(), colors)):
        ax.plot(iterations, values, color=color, linewidth=2, label=metric_name)
        ax.set_xlabel('Iteración', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'Evolución de {metric_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()


def visualize_parameter_space(
    solutions: List[Dict],
    parameter_names: Optional[List[str]] = None,
    title: str = "Espacio de Parámetros",
    save_path: Optional[str] = None
) -> None:
    """
    Visualiza el espacio de parámetros explorado durante la optimización.
    
    Args:
        solutions: Lista de soluciones con 'position' y 'objectives'.
        parameter_names: Nombres de los parámetros.
        title: Título de la figura.
        save_path: Ruta para guardar la imagen.
    
    Examples:
        >>> solutions = [
        ...     {'position': np.array([8, 8, 2.0]), 'objectives': np.array([7.5, 0.9, 85])}
        ... ]
        >>> visualize_parameter_space(solutions, ['Rx', 'Ry', 'ClipLimit'])
    """
    if not solutions:
        print("No hay soluciones para visualizar")
        return
    
    if parameter_names is None:
        n_params = len(solutions[0]['position'])
        parameter_names = [f'Param {i+1}' for i in range(n_params)]
    
    positions = np.array([s['position'] for s in solutions])
    objectives = np.array([s['objectives'] for s in solutions])
    
    # Usar primera métrica objetivo para colorear
    colors = objectives[:, 0]
    
    n_params = positions.shape[1]
    
    # Crear scatter matrix
    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))
    
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histograma
                ax.hist(positions[:, i], bins=20, alpha=0.7, color='steelblue')
                ax.set_ylabel('Frecuencia', fontsize=9)
            else:
                # Fuera de diagonal: scatter plot
                scatter = ax.scatter(positions[:, j], positions[:, i], 
                                   c=colors, cmap='viridis', alpha=0.6, s=30)
            
            if i == n_params - 1:
                ax.set_xlabel(parameter_names[j], fontsize=10)
            else:
                ax.set_xticks([])
            
            if j == 0 and i != j:
                ax.set_ylabel(parameter_names[i], fontsize=10)
            elif j != 0:
                ax.set_yticks([])
    
    # Colorbar
    fig.colorbar(scatter, ax=axes.ravel().tolist(), label='Objetivo 1')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()


def create_results_summary_figure(
    original: NDArray[np.uint8],
    enhanced: NDArray[np.uint8],
    pareto_front: List[Dict],
    mcdm_selections: Dict[str, int],
    metrics: Dict[str, float],
    save_path: Optional[str] = None
) -> None:
    """
    Crea una figura resumen con todos los resultados principales.
    
    Args:
        original: Imagen original.
        enhanced: Imagen mejorada seleccionada.
        pareto_front: Frente de Pareto.
        mcdm_selections: Selecciones de métodos MCDM.
        metrics: Métricas de la imagen seleccionada.
        save_path: Ruta para guardar.
    
    Examples:
        >>> create_results_summary_figure(
        ...     original, enhanced, pareto_front,
        ...     {'TOPSIS': 2, 'VIKOR': 3},
        ...     {'Entropía': 7.5, 'SSIM': 0.95, 'VQI': 85}
        ... )
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Imágenes
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original', fontsize=13, fontweight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(enhanced, cmap='gray')
    ax2.set_title('Mejorada (Selección MCDM)', fontsize=13, fontweight='bold')
    ax2.axis('off')
    
    # Histogramas
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(original.ravel(), bins=256, alpha=0.5, color='blue', label='Original')
    ax3.hist(enhanced.ravel(), bins=256, alpha=0.5, color='red', label='Mejorada')
    ax3.set_title('Histogramas', fontsize=12)
    ax3.legend()
    
    # Frente de Pareto 3D
    ax4 = fig.add_subplot(gs[:, 2], projection='3d')
    pareto_obj = np.array([s['objectives'] for s in pareto_front])
    ax4.scatter(pareto_obj[:, 0], pareto_obj[:, 1], pareto_obj[:, 2],
               c='blue', s=80, alpha=0.6)
    ax4.set_xlabel('Entropía', fontsize=10)
    ax4.set_ylabel('SSIM', fontsize=10)
    ax4.set_zlabel('VQI', fontsize=10)
    ax4.set_title('Frente de Pareto 3D', fontsize=12, fontweight='bold')
    
    # Métricas y selecciones
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    info_text = "MÉTRICAS:\n"
    for k, v in metrics.items():
        info_text += f"  {k}: {v:.4f}\n"
    
    info_text += "\nSELECCIONES MCDM:\n"
    for method, idx in mcdm_selections.items():
        info_text += f"  {method}: Solución {idx}\n"
    
    ax5.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Resumen de Resultados', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Resumen guardado en: {save_path}")
    
    plt.show()
