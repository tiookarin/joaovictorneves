"""
core/qiskit_nature_swarm.py

Ferramentas para modelar partículas fermionicas, mapear para qubits (Jordan-Wigner)
e construir modelos de Ising para formação de enxames.

Funcionalidades:
 - build_fermionic_hamiltonian: criar Hamiltoniano de segunda quantização (mock)
 - jordan_wigner_transform: mapear operador fermiónico -> matriz qubit (quando qiskit_nature disponível)
 - build_ising_formation: construir Ising-like Hamiltoniano que incentiva formação desejada
 - ground_state_solver: VQE / QAOA / exact diagonalization fallback
"""
from typing import Any, Dict, Optional, Tuple
import logging
import numpy as np
import time
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from qiskit_nature.second_q.hamiltonians import HermitianSecondQuantizedOp
    from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
    from qiskit_nature.mappers.second_quantization import JordanWignerMapper
    from qiskit.opflow import PauliSumOp
    from qiskit import Aer, transpile
    _HAS_QISKIT_NATURE = True
except Exception:
    _HAS_QISKIT_NATURE = False
    logger.info("qiskit-nature not available -- using numpy fallbacks for fermionic models.")


def build_fermionic_hamiltonian(n_modes: int, coupling_scale: float = 1.0) -> Any:
    """
    Construct a toy fermionic Hamiltonian in second quantization.
    If qiskit_nature available, returns a SecondQuantizedOp; else returns a dense numpy matrix.
    """
    if n_modes <= 0:
        raise ValueError("n_modes must be >=1")
    if _HAS_QISKIT_NATURE:
        # Minimal illustrative Hermitian matrix to wrap into SecondQuantizedOp is nontrivial;
        # here we provide a placeholder and let ground_state_solver decide how to handle it.
        logger.debug("qiskit-nature path selected for fermionic Hamiltonian (placeholder).")
        # For now, return a simple identity-like placeholder object
        return {"type": "fermionic_placeholder", "n_modes": n_modes, "scale": coupling_scale}
    else:
        # Build a toy sparse Hamiltonian in qubit space dimension 2^n_modes
        dim = 2 ** n_modes
        H = np.zeros((dim, dim), dtype=float)
        # simple nearest-neighbor hopping-like term (mock)
        for i in range(dim):
            H[i, i] += coupling_scale * (bin(i).count("1") % 3 - 1)
        # small off-diagonals
        for i in range(dim - 1):
            H[i, i + 1] = -0.5 * coupling_scale
            H[i + 1, i] = -0.5 * coupling_scale
        return H


def jordan_wigner_transform(fermionic_op: Any) -> Any:
    """
    Map fermionic operator to qubit operator. If qiskit-nature available, use JordanWignerMapper.
    Otherwise provide identity mapping (placeholder).
    """
    if _HAS_QISKIT_NATURE:
        logger.debug("Applying Jordan-Wigner mapper (qiskit-nature).")
        # In real implementation we'd convert SecondQuantizedOp -> PauliSumOp
        return {"mapped": True, "original": fermionic_op}
    else:
        logger.debug("Fallback jordan_wigner: returning original numpy matrix (assumed qubit H).")
        return fermionic_op


def build_ising_formation(n_spins: int, coupling: float = 1.0, local_fields: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build a simple Ising Hamiltonian in numpy (dense) that encourages certain spin alignments
    representing formation objectives. Returns a (2^n, 2^n) Hermitian matrix.
    """
    if n_spins <= 0:
        raise ValueError("n_spins must be >=1")
    dim = 2 ** n_spins
    H = np.zeros((dim, dim), dtype=float)
    # Z single-qubit matrix
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    I = np.eye(2)
    # sum h_i Z_i
    if local_fields is None:
        local_fields = np.zeros(n_spins)
    # build via kronecker
    for i in range(n_spins):
        ops = [I] * n_spins
        ops[i] = Z
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H += local_fields[i] * term
    # pairwise coupling Z_i Z_j
    for i in range(n_spins - 1):
        ops = [I] * n_spins
        ops[i] = Z
        ops[i + 1] = Z
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H += coupling * term
    # symmetrize
    H = 0.5 * (H + H.T.conj())
    return H


def ground_state_solver(operator: Any, method: str = "exact", backend: str = "statevector", maxiter: int = 200) -> Dict[str, Any]:
    """
    Solve for ground state energy and state.
    - If operator is numpy ndarray and method == 'exact' -> diagonalize
    - If qiskit available and method == 'vqe' -> run a small VQE (best-effort)
    - method: 'exact'|'vqe'|'qaoa'
    Returns dict with keys: energy, statevector (or None), method, runtime
    """
    t0 = time.time()
    if isinstance(operator, np.ndarray) and method == "exact":
        w, v = np.linalg.eigh(operator)
        idx = int(np.argmin(w))
        energy = float(w[idx])
        state = v[:, idx]
        return {"energy": energy, "state": state, "method": "exact", "runtime": time.time() - t0}
    if _HAS_QISKIT_NATURE and method in ("vqe", "qaoa"):
        # Try a minimal VQE using qiskit primitives
        try:
            from qiskit import Aer
            from qiskit.algorithms import VQE
            from qiskit.circuit.library import RealAmplitudes
            from qiskit.algorithms.optimizers import COBYLA
            from qiskit.opflow import PauliSumOp

            if isinstance(operator, np.ndarray):
                Pauli = PauliSumOp.from_list([("I" * int(np.log2(operator.shape[0])), float(np.trace(operator) / operator.shape[0]))])
                # NOTE: converting dense operator to PauliSumOp is nontrivial; here we fallback
                H = Pauli
            else:
                H = operator  # assume already proper op
            ansatz = RealAmplitudes(int(np.log2(operator.shape[0])), reps=1)
            optimizer = COBYLA(maxiter=200)
            vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=Aer.get_backend("statevector_simulator"))
            res = vqe.compute_minimum_eigenvalue(operator=H)
            return {"energy": float(res.eigenvalue.real), "state": None, "method": "vqe", "runtime": time.time() - t0}
        except Exception as e:
            logger.exception("VQE fallback failed: %s", e)
    # If we reach here, fallback to diagonalization if operator is matrix-like
    if isinstance(operator, np.ndarray):
        w, v = np.linalg.eigh(operator)
        idx = int(np.argmin(w))
        energy = float(w[idx])
        state = v[:, idx]
        return {"energy": energy, "state": state, "method": "fallback_exact", "runtime": time.time() - t0}
    raise RuntimeError("Unsupported operator type for ground_state_solver")