# core/swarm_quantum_simulator.py
"""
Simulador de dinâmica unicycle (simplificado) com ruído browniano.

Fornece:
 - SwarmQuantumSimulator.simulate(initial_positions, target, assignment, slots) -> dict com trajetórias e métricas
 - compute_cohesion(positions) -> float

Melhorias:
 - seed configurável para reprodutibilidade
 - kp, maxspeed configuráveis
 - tratamento correto de 'assignment' + 'slots'
 - validações de shapes e logs opcionais
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

    Retorna valor no intervalo [0, +inf) na formulação atual; tipicamente próximo de 1 para alta coesão.
    """
    if positions is None or positions.size == 0:
        return 0.0
    if positions.ndim != 2 or positions.shape[1] < 1:
        raise ValueError("positions deve ser array 2D de forma (n, dim).")
    n = positions.shape[0]
    centroid = positions.mean(axis=0)
    d_to_centroid = np.linalg.norm(positions - centroid, axis=1)
    std = float(d_to_centroid.std())
    if n <= 1:
        mean_pairwise = 1.0
    else:
        pairwise = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        # usar somente elementos acima da diagonal (cada par contado uma vez)
        triu = pairwise[np.triu_indices(n, k=1)]
        mean_pairwise = float(triu.mean()) if triu.size > 0 else 1.0
    if mean_pairwise == 0:
        return 1.0
    cohesion = max(0.0, 1.0 - std / mean_pairwise)
    return cohesion


class SwarmQuantumSimulator:
    """
    Simulador simples de controle de enxame com ruído Browniano.

    Parâmetros principais:
      n_robots: número esperado de robôs (opcional — se initial_positions tiver shape, é usado)
      dt: passo de tempo
      steps: número de passos de simulação
      noise_sigma: desvio padrão do ruído Browniano adicionado a cada passo
      kp: ganho proporcional (controle P)
      maxspeed: velocidade máxima (clamp)
      seed: seed do RNG para reprodutibilidade (None => não fixa seed)
      verbose: se True, loga informações de runtime
    """
    def __init__(
        self,
        n_robots: int = 3,
        dt: float = 0.1,
        steps: int = 200,
        noise_sigma: float = 0.05,
        kp: float = 0.5,
        maxspeed: float = 0.5,
        seed: Optional[int] = 123,
        verbose: bool = False,
    ) -> None:
        self.n_robots = int(n_robots)
        self.dt = float(dt)
        self.steps = int(steps)
        self.noise_sigma = float(noise_sigma)
        self.kp = float(kp)
        self.maxspeed = float(maxspeed)
        self.seed = seed
        self.verbose = bool(verbose)

    def simulate(
        self,
        initial_positions: np.ndarray,
        target: np.ndarray,
        assignment: Optional[List[Tuple[int, int]]] = None,
        slots: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Simula trajetórias onde cada robô é atraído a um destino (slot atribuído ou target)
        com dinâmicas de primeira ordem (velocity-first order) e ruído Browniano.

        Parameters:
          initial_positions: array (n, dim) com posições iniciais (dim normalmente 2)
          target: array (dim,) ou (1,dim) usado como fallback quando slots/assignment não ocupados
          assignment: opcional lista de pares (robot_idx, slot_idx)
          slots: opcional array (m, dim) contendo posições de slots; usado quando assignment é passado

        Returns:
          dict com campos:
            - trajectories: array (T, n, dim)
            - metrics: dicionário com metrics (initial_cohesion, final_cohesion, final_distances, mean_speed, steps)
            - runtime: tempo de execução em segundos
        """
        t0 = time.time()
        rng = np.random.default_rng(seed=self.seed)

        if initial_positions is None:
            raise ValueError("initial_positions não pode ser None")
        positions = np.array(initial_positions, dtype=float)
        if positions.ndim != 2:
            raise ValueError("initial_positions deve ter shape (n, dim).")
        n, dim = positions.shape
        if n == 0:
            raise ValueError("initial_positions deve conter ao menos um robô.")

        # usar n inferido se diferente do n_robots fornecido
        if self.n_robots != n:
            if self.verbose:
                logger.info("n_robots fornecido (%d) diferente de initial_positions (%d). Usando %d.", self.n_robots, n, n)
            self.n_robots = n

        target = np.asarray(target, dtype=float).reshape(-1)
        if target.size != dim:
            raise ValueError(f"target deve ter dimensão {dim}, mas tem {target.size}.")

        # construir destinos (dests) por robô
        if assignment is not None and len(assignment) > 0:
            if slots is None:
                # fallback: todos apontam para target
                if self.verbose:
                    logger.info("assignment fornecido sem 'slots' — usando target como destino para todos.")
                dests = np.tile(target.reshape(1, dim), (n, 1))
            else:
                slots_arr = np.asarray(slots, dtype=float)
                if slots_arr.ndim != 2 or slots_arr.shape[1] != dim:
                    raise ValueError("slots deve ser array 2D com a mesma dimensão espacial que initial_positions.")
                dests = np.tile(target.reshape(1, dim), (n, 1))
                # preencher destinos conforme assignment — assignment lista pares (robot_idx, slot_idx)
                for robot_idx, slot_idx in assignment:
                    if not (0 <= robot_idx < n):
                        raise IndexError(f"robot_idx fora de range: {robot_idx}")
                    if not (0 <= slot_idx < slots_arr.shape[0]):
                        raise IndexError(f"slot_idx fora de range: {slot_idx}")
                    dests[robot_idx] = slots_arr[slot_idx]
        else:
            dests = np.tile(target.reshape(1, dim), (n, 1))

        velocities = np.zeros_like(positions)
        traj = [positions.copy()]

        # simulação principal
        for step in range(self.steps):
            # controle P simples: u = kp * (dest - pos)
            u = self.kp * (dests - positions)

            # clamp speed por robô
            speeds = np.linalg.norm(u, axis=1)
            # evitar divisão por zero
            scale = np.minimum(1.0, self.maxspeed / (speeds + 1e-8))
            u = u * scale[:, None]

            # atualizar posições com ruído Browniano
            noise = rng.normal(scale=self.noise_sigma, size=positions.shape)
            positions = positions + u * self.dt + noise * math.sqrt(self.dt)

            # armazenar velocidade aproximada (para métricas)
            velocities = u
            traj.append(positions.copy())

        traj = np.stack(traj, axis=0)  # shape (T, n, dim)

        # métricas
        final_pos = traj[-1]
        final_distances = np.linalg.norm(final_pos - dests, axis=1)  # por robô
        metrics = {
            "initial_cohesion": float(compute_cohesion(traj[0])),
            "final_cohesion": float(compute_cohesion(final_pos)),
            "final_distances": final_distances.tolist(),
            "final_mean_distance": float(np.mean(final_distances)),
            "mean_speed": float(np.mean(np.linalg.norm(velocities, axis=1))),
            "steps": self.steps,
        }
        runtime = time.time() - t0
        if self.verbose:
            logger.info("Simulação concluída em %.3fs — steps=%d, final_mean_distance=%.4f", runtime, self.steps, metrics["final_mean_distance"])
        return {"trajectories": traj, "metrics": metrics, "runtime": runtime}
