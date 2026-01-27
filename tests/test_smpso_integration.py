"""
Test de integración para SMPSOImageOptimizer.

Este test verifica que el flujo completo de optimización funciona correctamente:
1. Crear imagen sintética
2. Ejecutar SMPSO con pocos parámetros (test rápido)
3. Verificar que el Frente de Pareto tiene soluciones válidas
"""

import sys
import numpy as np

# Agregar el directorio raíz al path
sys.path.insert(0, '.')

from src.optimization.smpso import SMPSOImageOptimizer
from src.optimization.pareto import ParetoFront, Solution


def test_smpso_basic():
    """Test básico de SMPSOImageOptimizer con imagen sintética."""
    print("\n" + "=" * 60)
    print("TEST: SMPSOImageOptimizer - Ejecucion basica")
    print("=" * 60)
    
    # Crear imagen sintética con gradiente (simula radiografía)
    np.random.seed(42)
    base = np.linspace(50, 200, 128).reshape(1, -1)
    image = np.tile(base, (128, 1)).astype(np.uint8)
    noise = np.random.randint(-20, 20, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    print(f"Imagen de prueba: {image.shape}, dtype={image.dtype}")
    print(f"Rango de valores: [{image.min()}, {image.max()}]")
    
    # Crear optimizador con parametros reducidos para test rapido
    optimizer = SMPSOImageOptimizer(
        image=image,
        n_particles=10,      # Pocas particulas para test rapido
        max_iterations=5,    # Pocas iteraciones para test rapido
        archive_size=20,
        verbose=True,
        seed=42
    )
    
    # Ejecutar optimizacion
    print("\nIniciando optimizacion...")
    pareto_front = optimizer.run()
    
    # Verificaciones
    print("\n" + "-" * 40)
    print("VERIFICACIONES:")
    print("-" * 40)
    
    # 1. El frente debe tener soluciones
    assert len(pareto_front) > 0, "El Frente de Pareto esta vacio"
    print(f"[OK] Frente de Pareto tiene {len(pareto_front)} soluciones")
    
    # 2. Cada solucion debe tener 3 parametros
    for sol in pareto_front:
        assert len(sol.parameters) == 3, "Los parametros deben ser 3 (rx, ry, clip)"
        assert len(sol.objectives) == 3, "Los objetivos deben ser 3 (H, SSIM, VQI)"
    print("[OK] Todas las soluciones tienen 3 parametros y 3 objetivos")
    
    # 3. Los parametros deben estar en rango valido
    params = pareto_front.get_parameters_matrix()
    assert np.all(params[:, 0] >= 2) and np.all(params[:, 0] <= 64), "rx fuera de rango"
    assert np.all(params[:, 1] >= 2) and np.all(params[:, 1] <= 64), "ry fuera de rango"
    assert np.all(params[:, 2] >= 1.0) and np.all(params[:, 2] <= 4.0), "clip fuera de rango"
    print("[OK] Todos los parametros estan en rango valido")
    
    # 4. Los objetivos deben ser valores razonables
    objectives = pareto_front.get_decision_matrix()
    assert np.all(objectives[:, 0] > 0), "Entropia debe ser positiva"
    assert np.all(objectives[:, 1] >= 0) and np.all(objectives[:, 1] <= 1), "SSIM en [0,1]"
    assert np.all(objectives[:, 2] > 0), "VQI debe ser positivo"
    print("[OK] Todos los objetivos tienen valores razonables")
    
    # 5. Probar obtener solucion de compromiso
    compromise = pareto_front.get_compromise_solution()
    assert compromise is not None, "Debe existir solucion de compromiso"
    print(f"[OK] Solucion de compromiso: params={compromise.parameters}")
    
    # 6. Probar obtener imagen mejorada
    enhanced = optimizer.get_enhanced_image(compromise)
    assert enhanced.shape == image.shape, "La imagen mejorada debe tener el mismo tamano"
    assert enhanced.dtype == np.uint8, "La imagen mejorada debe ser uint8"
    print(f"[OK] Imagen mejorada generada: {enhanced.shape}")
    
    print("\n" + "=" * 60)
    print("TODOS LOS TESTS PASARON CORRECTAMENTE")
    print("=" * 60)
    
    return pareto_front


def test_pareto_front_methods():
    """Test de los metodos de ParetoFront."""
    print("\n" + "=" * 60)
    print("TEST: ParetoFront - Metodos auxiliares")
    print("=" * 60)
    
    # Crear frente de prueba
    front = ParetoFront(max_size=10, maximize=True)
    
    # Agregar soluciones no dominadas
    solutions_data = [
        ([8, 8, 2.0], [7.5, 0.95, 85.0]),   # Mejor SSIM
        ([16, 16, 3.0], [7.8, 0.90, 90.0]), # Mejor VQI
        ([12, 12, 2.5], [7.9, 0.88, 82.0]), # Mejor entropia
        ([10, 10, 2.2], [7.6, 0.92, 87.0]), # Balanceada
    ]
    
    for params, objs in solutions_data:
        sol = Solution(
            parameters=np.array(params),
            objectives=np.array(objs)
        )
        front.add(sol)
    
    print(f"Soluciones agregadas: {len(front)}")
    
    # Test get_decision_matrix
    dm = front.get_decision_matrix()
    assert dm.shape[1] == 3, "Matriz de decision debe tener 3 columnas"
    print(f"[OK] get_decision_matrix: shape={dm.shape}")
    
    # Test get_parameters_matrix
    pm = front.get_parameters_matrix()
    assert pm.shape[1] == 3, "Matriz de parametros debe tener 3 columnas"
    print(f"[OK] get_parameters_matrix: shape={pm.shape}")
    
    # Test get_best_by_objective
    best_entropy = front.get_best_by_objective(0)
    best_ssim = front.get_best_by_objective(1)
    best_vqi = front.get_best_by_objective(2)
    print(f"[OK] Mejor entropia: {best_entropy.objectives[0]:.4f}")
    print(f"[OK] Mejor SSIM: {best_ssim.objectives[1]:.4f}")
    print(f"[OK] Mejor VQI: {best_vqi.objectives[2]:.2f}")
    
    # Test get_compromise_solution
    compromise = front.get_compromise_solution()
    print(f"[OK] Solucion de compromiso: {compromise.objectives}")
    
    # Test select_leader
    leader = front.select_leader()
    assert leader is not None, "Debe seleccionar un lider"
    print(f"[OK] Lider seleccionado: crowding_distance={leader.crowding_distance:.4f}")
    
    # Test to_list
    list_repr = front.to_list()
    assert len(list_repr) == len(front), "to_list debe retornar todas las soluciones"
    print(f"[OK] to_list: {len(list_repr)} soluciones")
    
    print("\n" + "=" * 60)
    print("TODOS LOS TESTS DE PARETO FRONT PASARON")
    print("=" * 60)


if __name__ == "__main__":
    # Ejecutar tests
    test_pareto_front_methods()
    test_smpso_basic()
