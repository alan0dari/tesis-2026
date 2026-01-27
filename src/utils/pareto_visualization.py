"""
Módulo de visualización avanzada para Frentes de Pareto 3D.

Proporciona visualizaciones estáticas (matplotlib) e interactivas (Plotly):
- Vista multi-ángulo (4 perspectivas)
- Proyecciones 2D con imágenes de entrada y salida
- Partículas interactivas 3D
- Superficie triangulada interactiva con partículas
- Codificación por parámetros CLAHE

Autor: Framework MCDM para Imágenes Médicas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List, Union
import warnings

# Intentar importar plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly no está instalado. Visualización interactiva no disponible.")

# Intentar importar scipy
try:
    from scipy.spatial import Delaunay
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy no está instalado. Triangulación no disponible.")


# =============================================================================
# VISUALIZACIONES ESTÁTICAS (Matplotlib)
# =============================================================================

def create_multi_angle_pareto(
    pareto_front,
    title: str = "Frente de Pareto - Múltiples Perspectivas",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Crea una visualización del Frente de Pareto desde 4 ángulos diferentes.
    
    Args:
        pareto_front: Objeto ParetoFront con las soluciones.
        title: Título principal de la figura.
        save_path: Ruta para guardar la imagen (opcional).
        figsize: Tamaño de la figura en pulgadas.
    """
    objectives = pareto_front.get_decision_matrix()
    compromise = pareto_front.get_compromise_solution()
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Ángulos de visualización: (elevación, azimut, título)
    angles = [
        (25, 45, "Vista Principal"),
        (25, 135, "Vista Lateral Derecha"),
        (25, 225, "Vista Posterior"),
        (90, 0, "Vista Superior (H vs SSIM)")
    ]
    
    for i, (elev, azim, subtitle) in enumerate(angles, 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        
        scatter = ax.scatter(
            objectives[:, 0],
            objectives[:, 1],
            objectives[:, 2],
            c=objectives[:, 2],
            cmap='viridis',
            s=40,
            alpha=0.8,
            edgecolors='black',
            linewidth=0.3
        )
        
        # Solución de compromiso
        ax.scatter(
            [compromise.objectives[0]],
            [compromise.objectives[1]],
            [compromise.objectives[2]],
            c='red',
            s=120,
            marker='*',
            edgecolors='black',
            linewidth=1,
            zorder=5
        )
        
        ax.set_xlabel('Entropía (H)', fontsize=9, labelpad=5)
        ax.set_ylabel('SSIM', fontsize=9, labelpad=5)
        ax.set_zlabel('VQI', fontsize=9, labelpad=5)
        ax.set_title(subtitle, fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Vista multi-ángulo guardada en: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def create_pareto_with_projections_and_images(
    pareto_front,
    degraded_image: np.ndarray,
    enhanced_image: np.ndarray,
    title: str = "Frente de Pareto 3D con Proyecciones e Imágenes",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 14)
) -> None:
    """
    Crea visualización 3D del Frente de Pareto con proyecciones 2D
    e imágenes de entrada (degradada) y salida (solución compromiso).
    
    Args:
        pareto_front: Objeto ParetoFront con las soluciones.
        degraded_image: Imagen degradada de entrada (numpy array).
        enhanced_image: Imagen mejorada con la solución de compromiso.
        title: Título principal de la figura.
        save_path: Ruta para guardar la imagen (opcional).
        figsize: Tamaño de la figura en pulgadas.
    """
    objectives = pareto_front.get_decision_matrix()
    compromise = pareto_front.get_compromise_solution()
    
    entropy = objectives[:, 0]
    ssim = objectives[:, 1]
    vqi = objectives[:, 2]
    
    # Crear figura con GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.2, 1, 1], 
                  width_ratios=[1.5, 1, 1, 1], hspace=0.3, wspace=0.3)
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # ===== FILA 1: Imágenes y Plot 3D =====
    
    # Imagen degradada (entrada)
    ax_deg = fig.add_subplot(gs[0, 0])
    ax_deg.imshow(degraded_image, cmap='gray')
    ax_deg.set_title('Imagen de Entrada\n(Degradada)', fontsize=12, fontweight='bold')
    ax_deg.axis('off')
    
    # Plot 3D principal
    ax3d = fig.add_subplot(gs[0:2, 1:3], projection='3d')
    
    scatter3d = ax3d.scatter(
        entropy, ssim, vqi,
        c=vqi, cmap='viridis',
        s=50, alpha=0.8,
        edgecolors='black', linewidth=0.3
    )
    
    ax3d.scatter(
        [compromise.objectives[0]],
        [compromise.objectives[1]],
        [compromise.objectives[2]],
        c='red', s=180, marker='*',
        label='Compromiso',
        edgecolors='black', linewidth=1.5, zorder=5
    )
    
    ax3d.set_xlabel('Entropía (H)', fontsize=11, labelpad=8)
    ax3d.set_ylabel('SSIM', fontsize=11, labelpad=8)
    ax3d.set_zlabel('VQI', fontsize=11, labelpad=8)
    ax3d.set_title('Frente de Pareto 3D', fontsize=12, fontweight='bold')
    ax3d.view_init(elev=25, azim=45)
    ax3d.legend(loc='upper left', fontsize=9)
    
    cbar = plt.colorbar(scatter3d, ax=ax3d, shrink=0.5, aspect=12, pad=0.1)
    cbar.set_label('VQI', fontsize=10)
    
    # Imagen mejorada (salida - compromiso)
    ax_enh = fig.add_subplot(gs[0, 3])
    ax_enh.imshow(enhanced_image, cmap='gray')
    ax_enh.set_title('Imagen de Salida\n(Solución Compromiso)', fontsize=12, fontweight='bold')
    ax_enh.axis('off')
    
    # Parámetros de la solución compromiso debajo de la imagen
    params_text = (f"Rx={compromise.parameters[0]:.0f}, "
                   f"Ry={compromise.parameters[1]:.0f}, "
                   f"Clip={compromise.parameters[2]:.2f}")
    ax_enh.text(0.5, -0.08, params_text, transform=ax_enh.transAxes,
                fontsize=10, ha='center', style='italic')
    
    # ===== FILA 2: Proyecciones 2D =====
    
    # Proyección 1: Entropía vs SSIM
    ax1 = fig.add_subplot(gs[1, 3])
    scatter1 = ax1.scatter(entropy, ssim, c=vqi, cmap='viridis',
                           s=35, alpha=0.7, edgecolors='black', linewidth=0.2)
    ax1.scatter(compromise.objectives[0], compromise.objectives[1],
                c='red', s=120, marker='*', edgecolors='black', linewidth=1, zorder=5)
    ax1.set_xlabel('Entropía (H)', fontsize=10)
    ax1.set_ylabel('SSIM', fontsize=10)
    ax1.set_title('Entropía vs SSIM', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ===== FILA 3: Más proyecciones y estadísticas =====
    
    # Proyección 2: Entropía vs VQI
    ax2 = fig.add_subplot(gs[2, 0])
    scatter2 = ax2.scatter(entropy, vqi, c=ssim, cmap='RdYlGn',
                           s=35, alpha=0.7, edgecolors='black', linewidth=0.2)
    ax2.scatter(compromise.objectives[0], compromise.objectives[2],
                c='red', s=120, marker='*', edgecolors='black', linewidth=1, zorder=5)
    ax2.set_xlabel('Entropía (H)', fontsize=10)
    ax2.set_ylabel('VQI', fontsize=10)
    ax2.set_title('Entropía vs VQI', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('SSIM', fontsize=9)
    
    # Proyección 3: SSIM vs VQI
    ax3 = fig.add_subplot(gs[2, 1])
    scatter3 = ax3.scatter(ssim, vqi, c=entropy, cmap='plasma',
                           s=35, alpha=0.7, edgecolors='black', linewidth=0.2)
    ax3.scatter(compromise.objectives[1], compromise.objectives[2],
                c='red', s=120, marker='*', edgecolors='black', linewidth=1, zorder=5)
    ax3.set_xlabel('SSIM', fontsize=10)
    ax3.set_ylabel('VQI', fontsize=10)
    ax3.set_title('SSIM vs VQI', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
    cbar3.set_label('Entropía', fontsize=9)
    
    # Panel de estadísticas
    ax_stats = fig.add_subplot(gs[2, 2:4])
    ax_stats.axis('off')
    
    stats_text = (
        f"{'═' * 45}\n"
        f"       ESTADÍSTICAS DEL FRENTE DE PARETO\n"
        f"{'═' * 45}\n\n"
        f"  Soluciones totales: {len(objectives)}\n\n"
        f"  Entropía (H):\n"
        f"    • Rango: [{entropy.min():.4f}, {entropy.max():.4f}]\n"
        f"    • Media: {entropy.mean():.4f}\n\n"
        f"  SSIM:\n"
        f"    • Rango: [{ssim.min():.4f}, {ssim.max():.4f}]\n"
        f"    • Media: {ssim.mean():.4f}\n\n"
        f"  VQI:\n"
        f"    • Rango: [{vqi.min():.2f}, {vqi.max():.2f}]\n"
        f"    • Media: {vqi.mean():.2f}\n\n"
        f"{'─' * 45}\n"
        f"  SOLUCIÓN DE COMPROMISO\n"
        f"{'─' * 45}\n"
        f"  Parámetros CLAHE:\n"
        f"    Rx={compromise.parameters[0]:.0f}, "
        f"Ry={compromise.parameters[1]:.0f}, "
        f"Clip={compromise.parameters[2]:.3f}\n\n"
        f"  Objetivos:\n"
        f"    H={compromise.objectives[0]:.4f}, "
        f"SSIM={compromise.objectives[1]:.4f}, "
        f"VQI={compromise.objectives[2]:.2f}"
    )
    
    ax_stats.text(
        0.05, 0.95, stats_text,
        transform=ax_stats.transAxes,
        fontsize=10, fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3)
    )
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Pareto con proyecciones e imágenes guardado en: {save_path}")
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# VISUALIZACIONES INTERACTIVAS (Plotly)
# =============================================================================

def create_interactive_pareto_particles(
    pareto_front,
    title: str = "Frente de Pareto 3D Interactivo",
    save_path: Optional[str] = None,
    color_by: str = 'vqi',
    marker_size: int = 5,
    compromise_size: int = 8
) -> Optional[object]:
    """
    Crea una visualización 3D interactiva del Frente de Pareto (solo partículas).
    
    Args:
        pareto_front: Objeto ParetoFront con las soluciones.
        title: Título de la visualización.
        save_path: Ruta para guardar el HTML interactivo.
        color_by: Variable para colorear ('vqi', 'ssim', 'entropy', 
                  'clip_limit', 'rx', 'ry', 'region_area').
        marker_size: Tamaño de los marcadores de partículas.
        compromise_size: Tamaño del marcador de la solución compromiso.
        
    Returns:
        Objeto Figure de Plotly.
    """
    if not PLOTLY_AVAILABLE:
        print("ERROR: Plotly no está instalado.")
        return None
    
    objectives = pareto_front.get_decision_matrix()
    solutions = list(pareto_front)
    compromise = pareto_front.get_compromise_solution()
    
    # Extraer parámetros
    rx = np.array([s.parameters[0] for s in solutions])
    ry = np.array([s.parameters[1] for s in solutions])
    clip = np.array([s.parameters[2] for s in solutions])
    
    # Opciones de color
    color_options = {
        'vqi': (objectives[:, 2], 'VQI', 'Viridis'),
        'ssim': (objectives[:, 1], 'SSIM', 'RdYlGn'),
        'entropy': (objectives[:, 0], 'Entropía', 'Plasma'),
        'clip_limit': (clip, 'Clip Limit', 'Inferno'),
        'rx': (rx, 'Rx', 'Blues'),
        'ry': (ry, 'Ry', 'Greens'),
        'region_area': (rx * ry, 'Área Región', 'Cividis')
    }
    
    if color_by not in color_options:
        color_by = 'vqi'
    
    color_data, color_label, colorscale = color_options[color_by]
    
    # Tooltips
    hover_texts = []
    for i, sol in enumerate(solutions):
        hover_texts.append(
            f"<b>Partícula {i+1}</b><br>"
            f"<b>━━ Parámetros CLAHE ━━</b><br>"
            f"  Rx: {sol.parameters[0]:.0f}<br>"
            f"  Ry: {sol.parameters[1]:.0f}<br>"
            f"  Clip: {sol.parameters[2]:.3f}<br>"
            f"<b>━━ Objetivos ━━</b><br>"
            f"  Entropía: {sol.objectives[0]:.4f}<br>"
            f"  SSIM: {sol.objectives[1]:.4f}<br>"
            f"  VQI: {sol.objectives[2]:.2f}"
        )
    
    fig = go.Figure()
    
    # Partículas
    fig.add_trace(go.Scatter3d(
        x=objectives[:, 0],
        y=objectives[:, 1],
        z=objectives[:, 2],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color_data,
            colorscale=colorscale,
            opacity=0.85,
            line=dict(width=0.5, color='black'),
            colorbar=dict(
                title=color_label,
                thickness=15,
                len=0.6
            )
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Soluciones Pareto'
    ))
    
    # Solución de compromiso
    fig.add_trace(go.Scatter3d(
        x=[compromise.objectives[0]],
        y=[compromise.objectives[1]],
        z=[compromise.objectives[2]],
        mode='markers',
        marker=dict(
            size=compromise_size,
            color='red',
            symbol='diamond',
            line=dict(width=1, color='black')
        ),
        text=[
            f"<b>★ COMPROMISO ★</b><br>"
            f"Rx={compromise.parameters[0]:.0f}, "
            f"Ry={compromise.parameters[1]:.0f}, "
            f"Clip={compromise.parameters[2]:.2f}<br>"
            f"H={compromise.objectives[0]:.4f}<br>"
            f"SSIM={compromise.objectives[1]:.4f}<br>"
            f"VQI={compromise.objectives[2]:.2f}"
        ],
        hoverinfo='text',
        name='Solución de Compromiso'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sub>{len(objectives)} soluciones | Color: {color_label}</sub>",
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(title=dict(text='Entropía (H)', font=dict(size=13)),
                       backgroundcolor='rgb(245, 245, 245)'),
            yaxis=dict(title=dict(text='SSIM', font=dict(size=13)),
                       backgroundcolor='rgb(245, 245, 245)'),
            zaxis=dict(title=dict(text='VQI', font=dict(size=13)),
                       backgroundcolor='rgb(245, 245, 245)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    if save_path:
        fig.write_html(save_path, include_plotlyjs=True, full_html=True)
        print(f"Partículas interactivas guardadas en: {save_path}")
    
    return fig


def create_interactive_pareto_triangulated(
    pareto_front,
    title: str = "Superficie Triangulada del Frente de Pareto",
    save_path: Optional[str] = None,
    surface_opacity: float = 0.6,
    color_by: str = 'clip_limit',
    marker_size: int = 5,
    compromise_size: int = 8
) -> Optional[object]:
    """
    Crea superficie triangulada (mesh) que conecta las partículas del Frente de Pareto.
    
    La superficie pasa exactamente por cada partícula usando triangulación de Delaunay.
    
    Args:
        pareto_front: Objeto ParetoFront con las soluciones.
        title: Título de la visualización.
        save_path: Ruta para guardar el HTML.
        surface_opacity: Opacidad de la superficie.
        color_by: Variable para colorear superficie y partículas.
        marker_size: Tamaño de los marcadores de partículas.
        compromise_size: Tamaño del marcador de compromiso.
        
    Returns:
        Objeto Figure de Plotly.
    """
    if not PLOTLY_AVAILABLE:
        print("ERROR: Plotly no está instalado.")
        return None
    
    if not SCIPY_AVAILABLE:
        print("ERROR: SciPy no está instalado.")
        return None
    
    objectives = pareto_front.get_decision_matrix()
    solutions = list(pareto_front)
    compromise = pareto_front.get_compromise_solution()
    
    entropy = objectives[:, 0]
    ssim = objectives[:, 1]
    vqi = objectives[:, 2]
    
    # Parámetros CLAHE
    rx = np.array([s.parameters[0] for s in solutions])
    ry = np.array([s.parameters[1] for s in solutions])
    clip = np.array([s.parameters[2] for s in solutions])
    
    # Opciones de color
    color_options = {
        'vqi': (vqi, 'VQI', 'Viridis'),
        'ssim': (ssim, 'SSIM', 'RdYlGn'),
        'entropy': (entropy, 'Entropía', 'Plasma'),
        'clip_limit': (clip, 'Clip Limit', 'Inferno'),
        'rx': (rx, 'Rx', 'Blues'),
        'ry': (ry, 'Ry', 'Greens'),
        'region_area': (rx * ry, 'Área Región', 'Cividis')
    }
    
    if color_by not in color_options:
        color_by = 'clip_limit'
    
    color_data, color_label, colorscale = color_options[color_by]
    
    # Triangulación de Delaunay en 2D (H, SSIM)
    points_2d = np.column_stack((entropy, ssim))
    
    try:
        tri = Delaunay(points_2d)
        simplices = tri.simplices
    except Exception as e:
        print(f"Error en triangulación: {e}")
        return create_interactive_pareto_particles(pareto_front, title, save_path)
    
    fig = go.Figure()
    
    # Superficie triangulada (Mesh3D)
    fig.add_trace(go.Mesh3d(
        x=entropy,
        y=ssim,
        z=vqi,
        i=simplices[:, 0],
        j=simplices[:, 1],
        k=simplices[:, 2],
        intensity=color_data,
        colorscale=colorscale,
        opacity=surface_opacity,
        showscale=True,
        colorbar=dict(
            title=dict(text=color_label, font=dict(size=11)),
            thickness=12,
            len=0.5,
            x=1.02
        ),
        name='Superficie',
        hoverinfo='skip'
    ))
    
    # Aristas de los triángulos
    edge_x, edge_y, edge_z = [], [], []
    for simplex in simplices:
        for i in range(3):
            p1, p2 = simplex[i], simplex[(i + 1) % 3]
            edge_x.extend([entropy[p1], entropy[p2], None])
            edge_y.extend([ssim[p1], ssim[p2], None])
            edge_z.extend([vqi[p1], vqi[p2], None])
    
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(50, 50, 50, 0.3)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Partículas como vértices
    hover_texts = []
    for i, sol in enumerate(solutions):
        hover_texts.append(
            f"<b>Partícula {i+1}</b><br>"
            f"<b>━━ Parámetros CLAHE ━━</b><br>"
            f"  Rx: {sol.parameters[0]:.0f}<br>"
            f"  Ry: {sol.parameters[1]:.0f}<br>"
            f"  Clip: {sol.parameters[2]:.3f}<br>"
            f"<b>━━ Objetivos ━━</b><br>"
            f"  H: {sol.objectives[0]:.4f}<br>"
            f"  SSIM: {sol.objectives[1]:.4f}<br>"
            f"  VQI: {sol.objectives[2]:.2f}"
        )
    
    fig.add_trace(go.Scatter3d(
        x=entropy, y=ssim, z=vqi,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=color_data,
            colorscale=colorscale,
            line=dict(width=1, color='white'),
            showscale=False
        ),
        text=hover_texts,
        hoverinfo='text',
        name=f'Partículas ({len(solutions)})'
    ))
    
    # Solución de compromiso
    fig.add_trace(go.Scatter3d(
        x=[compromise.objectives[0]],
        y=[compromise.objectives[1]],
        z=[compromise.objectives[2]],
        mode='markers',
        marker=dict(
            size=compromise_size,
            color='red',
            symbol='diamond',
            line=dict(width=1.5, color='gold')
        ),
        text=[
            f"<b>⭐ COMPROMISO ⭐</b><br>"
            f"Rx={compromise.parameters[0]:.0f}, "
            f"Ry={compromise.parameters[1]:.0f}, "
            f"Clip={compromise.parameters[2]:.2f}<br>"
            f"H={compromise.objectives[0]:.4f}<br>"
            f"SSIM={compromise.objectives[1]:.4f}<br>"
            f"VQI={compromise.objectives[2]:.2f}"
        ],
        hoverinfo='text',
        name='★ Compromiso'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sub>{len(solutions)} partículas | Color: {color_label}</sub>",
            x=0.5,
            font=dict(size=15)
        ),
        scene=dict(
            xaxis=dict(title=dict(text='Entropía (H)', font=dict(size=12)),
                       backgroundcolor='rgb(248, 248, 250)'),
            yaxis=dict(title=dict(text='SSIM', font=dict(size=12)),
                       backgroundcolor='rgb(248, 250, 248)'),
            zaxis=dict(title=dict(text='VQI', font=dict(size=12)),
                       backgroundcolor='rgb(250, 248, 248)'),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8)
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="gray", borderwidth=1),
        margin=dict(l=0, r=50, t=80, b=0)
    )
    
    if save_path:
        fig.write_html(save_path, include_plotlyjs=True, full_html=True)
        print(f"Superficie triangulada guardada en: {save_path}")
    
    return fig


def create_pareto_by_clahe_params(
    pareto_front,
    title: str = "Frente de Pareto - Codificado por Parámetros CLAHE",
    save_path: Optional[str] = None,
    marker_size_range: Tuple[int, int] = (3, 10)
) -> Optional[object]:
    """
    Crea visualización con codificación dual:
    - Color según Clip Limit
    - Tamaño de marcador según área de región (Rx × Ry)
    
    Args:
        pareto_front: Objeto ParetoFront.
        title: Título de la visualización.
        save_path: Ruta para guardar el HTML.
        marker_size_range: Tupla (min, max) para tamaño de marcadores.
    """
    if not PLOTLY_AVAILABLE:
        print("ERROR: Plotly no está instalado.")
        return None
    
    objectives = pareto_front.get_decision_matrix()
    solutions = list(pareto_front)
    compromise = pareto_front.get_compromise_solution()
    
    entropy = objectives[:, 0]
    ssim = objectives[:, 1]
    vqi = objectives[:, 2]
    
    rx = np.array([s.parameters[0] for s in solutions])
    ry = np.array([s.parameters[1] for s in solutions])
    clip = np.array([s.parameters[2] for s in solutions])
    area = rx * ry
    
    # Normalizar área para tamaño de marcador
    area_norm = (area - area.min()) / (area.max() - area.min() + 1e-6)
    min_size, max_size = marker_size_range
    marker_sizes = min_size + area_norm * (max_size - min_size)
    
    hover_texts = []
    for i, sol in enumerate(solutions):
        hover_texts.append(
            f"<b>Partícula {i+1}</b><br>"
            f"Rx: {sol.parameters[0]:.0f}, "
            f"Ry: {sol.parameters[1]:.0f}<br>"
            f"Área: {sol.parameters[0] * sol.parameters[1]:.0f}<br>"
            f"Clip: {sol.parameters[2]:.3f}<br>"
            f"H: {sol.objectives[0]:.4f}<br>"
            f"SSIM: {sol.objectives[1]:.4f}<br>"
            f"VQI: {sol.objectives[2]:.2f}"
        )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=entropy, y=ssim, z=vqi,
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=clip,
            colorscale='Inferno',
            opacity=0.85,
            line=dict(width=0.8, color='white'),
            colorbar=dict(title='Clip Limit', thickness=12, len=0.5)
        ),
        text=hover_texts,
        hoverinfo='text',
        name='Partículas'
    ))
    
    # Compromiso
    comp_area = compromise.parameters[0] * compromise.parameters[1]
    fig.add_trace(go.Scatter3d(
        x=[compromise.objectives[0]],
        y=[compromise.objectives[1]],
        z=[compromise.objectives[2]],
        mode='markers',
        marker=dict(
            size=10,
            color='lime',
            symbol='diamond',
            line=dict(width=1.5, color='black')
        ),
        text=[f"★ Compromiso<br>Clip={compromise.parameters[2]:.2f}<br>Área={comp_area:.0f}"],
        hoverinfo='text',
        name='★ Compromiso'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><sub>Color: Clip Limit | Tamaño: Área (Rx×Ry)</sub>",
            x=0.5, font=dict(size=14)
        ),
        scene=dict(
            xaxis=dict(title=dict(text='Entropía (H)', font=dict(size=12))),
            yaxis=dict(title=dict(text='SSIM', font=dict(size=12))),
            zaxis=dict(title=dict(text='VQI', font=dict(size=12))),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(l=0, r=40, t=80, b=0)
    )
    
    if save_path:
        fig.write_html(save_path, include_plotlyjs=True, full_html=True)
        print(f"Pareto por parámetros guardado en: {save_path}")
    
    return fig


def create_complete_pareto_report(
    pareto_front,
    output_dir: str,
    base_name: str = "pareto",
    degraded_image: Optional[np.ndarray] = None,
    enhanced_image: Optional[np.ndarray] = None
) -> List[str]:
    """
    Genera reporte completo de visualizaciones del Frente de Pareto.
    
    Args:
        pareto_front: Objeto ParetoFront.
        output_dir: Directorio de salida.
        base_name: Prefijo para archivos.
        degraded_image: Imagen degradada (para proyecciones).
        enhanced_image: Imagen mejorada (para proyecciones).
        
    Returns:
        Lista de rutas de archivos generados.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    # 1. Vista multi-ángulo (estática)
    path1 = os.path.join(output_dir, f"{base_name}_multiangulo.png")
    create_multi_angle_pareto(pareto_front, save_path=path1)
    generated_files.append(path1)
    
    # 2. Proyecciones con imágenes (si se proporcionan)
    if degraded_image is not None and enhanced_image is not None:
        path2 = os.path.join(output_dir, f"{base_name}_proyecciones.png")
        create_pareto_with_projections_and_images(
            pareto_front, degraded_image, enhanced_image, save_path=path2
        )
        generated_files.append(path2)
    
    # 3. Partículas interactivas
    if PLOTLY_AVAILABLE:
        path3 = os.path.join(output_dir, f"{base_name}_interactivo.html")
        create_interactive_pareto_particles(pareto_front, save_path=path3)
        generated_files.append(path3)
        
        # 4. Superficie triangulada
        if SCIPY_AVAILABLE:
            path4 = os.path.join(output_dir, f"{base_name}_triangulado.html")
            create_interactive_pareto_triangulated(pareto_front, save_path=path4)
            generated_files.append(path4)
        
        # 5. Por parámetros CLAHE
        path5 = os.path.join(output_dir, f"{base_name}_por_parametros.html")
        create_pareto_by_clahe_params(pareto_front, save_path=path5)
        generated_files.append(path5)
    
    print(f"\n{'=' * 50}")
    print("REPORTE DE PARETO GENERADO")
    print(f"{'=' * 50}")
    print(f"Archivos en: {output_dir}")
    for f in generated_files:
        print(f"  • {os.path.basename(f)}")
    
    return generated_files
