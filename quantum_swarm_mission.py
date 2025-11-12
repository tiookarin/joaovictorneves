# quantum_swarm_mission.py
"""
Facade de alto nível para configurar e executar uma missão de formação do enxame.

Uso:
    from quantum_swarm_mission import FormationMission
    mission = FormationMission(n_robots=3, target_pos=np.array([10.0, 10.0]))
    results = mission.execute_full_cycle(backend='simulator')
"""
from typing import Any, Dict, Optional
import numpy as np
import logging
import time

from core.hamiltonian_builder import build_hamiltonian
from core.vqe_formation_optimizer import VQEFormationOptimizer
from core.qaoa_assignment_solver import QAOAAssignmentSolver
from core.swarm_quantum_simulator import SwarmQuantumSimulator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FormationMission:
    def __init__(
        self,
        n_robots: int = 3,
        target_pos: Optional[np.ndarray] = None,
        backend: str = "simulator",
    ) -> None:
        self.n_robots = int(n_robots)
        self.target_pos = (
            np.array(target_pos, dtype=float) if target_pos is not None else np.array([10.0, 10.0])
        )
        self.backend = backend

    def execute_full_cycle(self, backend: Optional[str] = None) -> Dict[str, Any]:
        start_total = time.time()
        backend = backend or self.backend
        logger.info("Starting FormationMission: n_robots=%d, target=%s, backend=%s", self.n_robots, self.target_pos, backend)

        # 1) Construir Hamiltoniano (mock clássico)
        H = build_hamiltonian(n_qubits=self.n_robots, alpha=1.0, beta=0.8, gamma=0.5, delta=0.2)
        logger.info("Hamiltoniano construído (shape=%s)", H.shape)

        # 2) Rodar VQE (aqui: diagonalização simples para obter auto-valor mínimo)
        vqe = VQEFormationOptimizer(backend=backend)
        energy_vqe, vqe_params, vqe_time = vqe.run_vqe(H)
        logger.info("VQE retornou energy=%.6f em %.3fs", energy_vqe, vqe_time)

        # 3) Montar matriz de custo para QAOA (exemplo sintético: custos de distância a slots)
        # slots: formação geométrica simples (ex.: V-shape)
        slots = self._generate_formation_slots(self.n_robots, center=self.target_pos)
        cost_matrix = self._build_cost_matrix(initial_positions=self._random_initial_positions(), slots=slots)
        qaoa = QAOAAssignmentSolver(backend=backend)
        assignment, assignment_cost, qaoa_time = qaoa.solve_assignment(cost_matrix)
        logger.info("QAOA-assignment done cost=%.3f in %.3fs assignment=%s", assignment_cost, qaoa_time, assignment)

        # 4) Simular dinâmica unicycle com controle derivado (mock) e ruído
        simulator = SwarmQuantumSimulator(n_robots=self.n_robots)
        sim_results = simulator.simulate(initial_positions=self._random_initial_positions(), target=self.target_pos, assignment=assignment)
        logger.info("Simulação completa. Final cohesion=%.3f", sim_results["metrics"]["final_cohesion"])

        total_time = time.time() - start_total

        results = {
            "vqe": {"energy": float(energy_vqe), "params": vqe_params, "time": float(vqe_time)},
            "qaoa": {"assignment": assignment, "cost": float(assignment_cost), "time": float(qaoa_time)},
            "simulation": sim_results,
            "total_time": float(total_time),
        }
        return results

    def _generate_formation_slots(self, n: int, center: np.ndarray) -> np.ndarray:
        # Gera slots simples em V-shape em torno do centro
        angles = np.linspace(-0.4, 0.4, n)
        radius = 1.5 * max(1.0, n / 3.0)
        slots = np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], axis=1)
        return slots

    def _random_initial_positions(self) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        pos = rng.normal(loc=0.0, scale=3.0, size=(self.n_robots, 2))
        return pos

    def _build_cost_matrix(self, initial_positions: np.ndarray, slots: np.ndarray) -> np.ndarray:
        # custo por distância euclidiana
        n = initial_positions.shape[0]
        cost = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                cost[i, j] = np.linalg.norm(initial_positions[i] - slots[j])
        return cost


if __name__ == "__main__":
    # Demo rápido quando executado diretamente
    mission = FormationMission(n_robots=3, target_pos=np.array([10.0, 10.0]))
    res = mission.execute_full_cycle()
    print(res)