# core/qaoa_assignment_solver.py
"""
Solver QAOA (mock) para atribuição robô->slot.

Implementação atual:
  - Constrói problema de atribuição a partir da matriz de custo e resolve via Hungarian (scipy)
  - Retorna uma tupla (assignment, cost, runtime)
assignment: lista de pares (robot_idx -> slot_idx)
"""
from typing import Any, List, Optional, Tuple
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    logger.warning("scipy não disponível; QAOA-assignment usará heurística gulosa.")


class QAOAAssignmentSolver:
    def __init__(self, backend: Optional[str] = "simulator") -> None:
        self.backend = backend

    def solve_assignment(self, cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], float, float]:
        t0 = time.time()
        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignment = list(zip(row_ind.tolist(), col_ind.tolist()))
            total_cost = float(cost_matrix[row_ind, col_ind].sum())
        else:
            # Heurística gulosa fallback
            n = cost_matrix.shape[0]
            assigned_slots = set()
            assignment = []
            total_cost = 0.0
            for i in range(n):
                j = int(np.argmin([cost_matrix[i, k] if k not in assigned_slots else np.inf for k in range(n)]))
                assignment.append((i, j))
                assigned_slots.add(j)
                total_cost += float(cost_matrix[i, j])
        runtime = time.time() - t0
        return assignment, total_cost, runtime