# core/vqe_formation_optimizer.py
"""
Otimizador VQE (mock/classical) para desenvolvimento.

Implementação atual: diagonaliza o Hamiltoniano e retorna o menor autovalor
(como se fosse o resultado ótimo de VQE) e o autovetor correspondente como 'params'.
Isso evita dependências pesadas de runtime e é reproduceável para testes.
"""
from typing import Any, Dict, Tuple, Optional
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VQEFormationOptimizer:
    def __init__(self, backend: Optional[str] = "simulator") -> None:
        self.backend = backend

    def run_vqe(self, hamiltonian: np.ndarray, maxiter: int = 200) -> Tuple[float, Dict[str, Any], float]:
        """
        "Executa" VQE retornando (energy, params, runtime)
        - energy: menor autovalor do Hamiltoniano
        - params: dicionário contendo 'statevector' aproximado (autovetor)
        - runtime: tempo em segundos
        """
        t0 = time.time()
        # diagonalização direta para obter o menor autovalor (sólido para n_qubits <= 6)
        try:
            w, v = np.linalg.eigh(hamiltonian)
            idx = int(np.argmin(w))
            energy = float(w[idx])
            state = v[:, idx]
            params = {"statevector": state.tolist(), "method": "exact_diagonalization"}
        except Exception as e:
            logger.exception("Falha na diagonalização: %s", e)
            # fallback: trace da matriz
            energy = float(np.trace(hamiltonian)) / hamiltonian.shape[0]
            params = {"statevector": [], "method": "trace_fallback"}

        runtime = time.time() - t0
        return energy, params, runtime