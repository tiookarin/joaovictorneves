# core/swarm_quantum_simulator.py
"""
Simulador de dinâmica unicycle com ruído browniano (simplificado).

Fornece:
 - simulate(initial_positions, target, assignment) -> dict com trajetórias e métricas
 - compute_cohesion(positions) -> float
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import math
import logging
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compute_cohesion(positions: np.ndarray) -> float:
    """
    Coesão definida de forma simples:
      cohesion = 1 - (std(distances_to_centroid) / mean_pairwise_distance)
    Retorna valor no intervalo [0, 1+] (não estritamente limitado), tipicamente próximo de 1 para alta coesão.
    """
    if positions.size == 0:
        return 0.0
    centroid = positions.mean(axis=0)
    d_to_centroid = np.linalg.norm(positions - centroid, axis=1)
    std = float(d_to_centroid.std())
    pairwise = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    mean_pairwise = float(pairwise[np.triu_indices(positions.shape[0], k=1)].mean()) if positions.shape[0] > 1 else 1.0
    if mean_pairwise == 0:
        return 1.0
    cohesion = max(0.0, 1.0 - std / mean_pairwise)
    return cohesion


class SwarmQuantumSimulator:
    def __init__(self, n_robots: int = 3, dt: float = 0.1, steps: int = 200, noise_sigma: float = 0.05) -> None:
        self.n_robots = int(n_robots)
        self.dt = float(dt)
        self.steps = int(steps)
        self.noise_sigma = float(noise_sigma)

    def simulate(self, initial_positions: np.ndarray, target: np.ndarray, assignment: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Simula uma trajetória simples onde cada robô é atraído ao slot atribuído (ou ao alvo)
        com dinâmicas velocity-first order e ruído Browniano.
        """
        t0 = time.time()
        rng = np.random.default_rng(seed=123)
        positions = initial_positions.copy().astype(float)
        velocities = np.zeros_like(positions)
        traj = [positions.copy()]

        # Se tiver assignment, construir destino por robô, senão usar target
        n = positions.shape[0]
        if assignment is not None:
            # assignment é lista de pares (robot_idx, slot_idx); se slots não fornecido, usa target
            # aqui assumimos que o usuário passou slots via closure (não ideal), mas simplificamos:
            dests = np.tile(target.reshape(1, 2), (n, 1))
        else:
            dests = np.tile(target.reshape(1, 2), (n, 1))

        for step in range(self.steps):
            # controle P simples: u = k_p * (dest - pos)
            u = 0.5 * (dests - positions)
            # clamp speed
            speeds = np.linalg.norm(u, axis=1)
            maxspeed = 0.5
            scale = np.minimum(1.0, maxspeed / (speeds + 1e-8))
            u = u * scale[:, None]

            # atualizar posições com ruído
            noise = rng.normal(scale=self.noise_sigma, size=positions.shape)
            positions = positions + u * self.dt + noise * math.sqrt(self.dt)
            traj.append(positions.copy())

        traj = np.stack(traj, axis=0)  # shape (T, n, 2)
        metrics = {
            "final_cohesion": float(compute_cohesion(traj[-1])),
            "initial_cohesion": float(compute_cohesion(traj[0])),
            "steps": self.steps,
        }
        runtime = time.time() - t0
        return {"trajectories": traj, "metrics": metrics, "runtime": runtime}