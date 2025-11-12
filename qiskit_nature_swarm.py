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
from typing import Any, Dict, Optional
import logging
import numpy as np
import time
import math

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_HAS_QISKIT_NATURE = False
try:
    # Optional Qiskit imports (may be absent in some environments)
    from qiskit_nature.second_q.hamiltonians import HermitianSecondQuantizedOp  # type: ignore
    from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem  # type: ignore
    from qiskit_nature.mappers.second_quantization import JordanWignerMapper  # type: ignore
    from qiskit.opflow import PauliSumOp  # type: ignore
    from qiskit import Aer, transpile  # type: ignore
    _HAS_QISKIT_NATURE = True
except Exception:
    _HAS_QISKIT_NATURE = False
    logger.info("qiskit-nature not available -- using numpy/scipy fallbacks for fermionic models.")

# Optional scipy import for sparse matrices / solvers
try:
    import scipy.sparse as sp  # type: ignore
    import scipy.sparse.linalg as spla  # type: ignore
    _HAS_SCIPY = True
except Exception:
    sp = None  # type: ignore
    spla = None  # type: ignore
    _HAS_SCIPY = False


def _is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n > 0


def build_fermionic_hamiltonian(n_modes: int, coupling_scale: float = 1.0) -> Any:
    """
    Construct a toy fermionic Hamiltonian in second quantization.
    If qiskit_nature available, return a placeholder SecondQuantizedOp-like object
    that downstream code (jordan_wigner_transform) can recognize. Otherwise returns
    a dense numpy matrix describing a mock qubit Hamiltonian (dimension 2^n_modes).
    """
    if n_modes <= 0:
        raise ValueError("n_modes must be >=1")
    if _HAS_QISKIT_NATURE:
        # Return a clear placeholder identifying a fermionic operator with parameters
        logger.debug("qiskit-nature path selected for fermionic Hamiltonian (placeholder object).")
        return {"type": "fermionic_placeholder", "n_modes": n_modes, "scale": float(coupling_scale)}
    else:
        # Build a toy dense Hamiltonian in qubit space dimension 2^n_modes
        dim = 2 ** n_modes
        H = np.zeros((dim, dim), dtype=float)
        # simple diagonal occupation-dependent potential (mock)
        for i in range(dim):
            H[i, i] += coupling_scale * (bin(i).count("1") % 3 - 1)
        # small off-diagonals to mimic hopping
        for i in range(dim - 1):
            H[i, i + 1] = -0.5 * coupling_scale
            H[i + 1, i] = -0.5 * coupling_scale
        return H


def jordan_wigner_transform(fermionic_op: Any) -> Any:
    """
    Map fermionic operator to qubit operator.
    - If qiskit-nature available and input is a SecondQuantizedOp-like object (or our placeholder),
      attempt to use JordanWignerMapper to produce a qubit operator (may be a Pauli-like object).
    - Otherwise, if input is already a numpy ndarray or sparse matrix, return it unchanged as
      an assumed qubit Hamiltonian.
    Returns an object usable by ground_state_solver. The exact type depends on availability of qiskit.
    """
    if _HAS_QISKIT_NATURE:
        logger.debug("Attempting Jordan-Wigner mapping via qiskit-nature.")
        try:
            # If it's our placeholder dict, we can't produce a real mapped operator without molecular integrals,
            # so return a descriptive dict that indicates mapping would be required.
            if isinstance(fermionic_op, dict) and fermionic_op.get("type") == "fermionic_placeholder":
                return {"mapped": True, "original": fermionic_op, "note": "placeholder-mapped"}
            # If it's a HermitianSecondQuantizedOp (or similar), attempt mapping
            if 'HermitianSecondQuantizedOp' in str(type(fermionic_op)):
                mapper = JordanWignerMapper()
                mapped = mapper.map(fermionic_op)
                # Try to convert to PauliSumOp if possible
                try:
                    # PauliSumOp may not be available in runtime typing context; attempt conversion safely
                    from qiskit.opflow import PauliSumOp as _PauliSumOp  # type: ignore
                    if isinstance(mapped, _PauliSumOp):
                        return mapped
                    # If mapping returns a dict-like, return it
                    return mapped
                except Exception:
                    return mapped
            # If it's a numpy array, return it (assume it's already a qubit Hamiltonian)
            if isinstance(fermionic_op, np.ndarray):
                logger.debug("Input to jordan_wigner_transform is ndarray; returning unchanged.")
                return fermionic_op
            # Fallback: return the object unchanged but flagged
            return {"mapped": True, "original": fermionic_op}
        except Exception as e:
            logger.exception("Jordan-Wigner mapping failed: %s", e)
            # Fallback to returning the original operator
            return fermionic_op
    else:
        logger.debug("Fallback jordan_wigner: returning original numpy/sparse operator (assumed qubit H).")
        return fermionic_op


def build_ising_formation(n_spins: int, coupling: float = 1.0, local_fields: Optional[np.ndarray] = None, *, sparse: Optional[bool] = None) -> Any:
    """
    Build a simple Ising Hamiltonian that encourages certain spin alignments.
    Returns either a dense numpy array (2^n x 2^n) or a scipy.sparse matrix when sparse=True
    or when n_spins is large and sparse is None.
    """
    if n_spins <= 0:
        raise ValueError("n_spins must be >=1")
    if local_fields is None:
        local_fields = np.zeros(n_spins)
    dim = 2 ** n_spins

    # Heuristic: for moderately large n, prefer sparse representation
    if sparse is None:
        sparse = (n_spins >= 12) and _HAS_SCIPY

    # Pauli-Z and identity
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    I = np.eye(2)

    if sparse and _HAS_SCIPY:
        # Build sparse matrix by summing Kronecker products in sparse form
        H = sp.csr_matrix((dim, dim), dtype=float)
        for i in range(n_spins):
            ops = [sp.identity(2, format='csr')] * n_spins
            ops[i] = sp.csr_matrix(Z)
            term = ops[0]
            for op in ops[1:]:
                term = sp.kron(term, op, format='csr')
            if local_fields is not None:
                H = H + local_fields[i] * term
        for i in range(n_spins - 1):
            ops = [sp.identity(2, format='csr')] * n_spins
            ops[i] = sp.csr_matrix(Z)
            ops[i + 1] = sp.csr_matrix(Z)
            term = ops[0]
            for op in ops[1:]:
                term = sp.kron(term, op, format='csr')
            H = H + coupling * term
        # ensure hermitian
        H = 0.5 * (H + H.getH())
        return H
    else:
        H = np.zeros((dim, dim), dtype=float)
        for i in range(n_spins):
            ops = [I] * n_spins
            ops[i] = Z
            term = ops[0]
            for op in ops[1:]:
                term = np.kron(term, op)
            H += local_fields[i] * term
        for i in range(n_spins - 1):
            ops = [I] * n_spins
            ops[i] = Z
            ops[i + 1] = Z
            term = ops[0]
            for op in ops[1:]:
                term = np.kron(term, op)
            H += coupling * term
        H = 0.5 * (H + H.T.conj())
        return H


def ground_state_solver(operator: Any, method: str = "exact", backend: str = "statevector", maxiter: int = 200) -> Dict[str, Any]:
    """
    Solve for ground state energy and state.
    - If operator is numpy ndarray and method == 'exact' -> diagonalize with numpy
    - If scipy is available and operator is sparse -> use eigsh for the smallest eigenvalue
    - If qiskit available and operator is a PauliSumOp or similar and method in ('vqe','qaoa') -> try VQE
    - Otherwise fallback to exact diagonalization for matrix-like operators

    Returns dict with keys: energy, statevector (or None), method, runtime
    """
    t0 = time.time()

    # Helper: try sparse solver
    if _HAS_SCIPY and sp is not None and hasattr(operator, 'shape') and getattr(operator, 'nnz', None) is not None:
        # treat as scipy sparse
        try:
            # Compute smallest algebraic eigenvalue using eigsh
            vals, vecs = spla.eigsh(operator, k=1, which='SA')
            energy = float(vals[0])
            state = vecs[:, 0]
            return {"energy": energy, "state": state, "method": "exact_sparse", "runtime": time.time() - t0}
        except Exception as e:
            logger.exception("Sparse eigensolver failed, falling back: %s", e)

    # If operator is dense numpy
    if isinstance(operator, np.ndarray):
        dim = operator.shape[0]
        if method == 'exact':
            w, v = np.linalg.eigh(operator)
            idx = int(np.argmin(w))
            energy = float(w[idx])
            state = v[:, idx]
            return {"energy": energy, "state": state, "method": "exact", "runtime": time.time() - t0}

    # For variational methods, ensure operator dimension is power of two (or it's a PauliSumOp)
    if method in ("vqe", "qaoa"):
        # If qiskit available and operator looks like a PauliSumOp, try running a lightweight VQE
        if _HAS_QISKIT_NATURE:
            try:
                from qiskit import Aer
                from qiskit.algorithms import VQE  # type: ignore
                from qiskit.circuit.library import RealAmplitudes  # type: ignore
                from qiskit.algorithms.optimizers import COBYLA  # type: ignore
                from qiskit.opflow import PauliSumOp  # type: ignore
                # attempt VQE for operators that are PauliSumOp-like
                if isinstance(operator, PauliSumOp):
                    # PauliSumOp exposes num_qubits via .num_qubits property or len on register
                    try:
                        n_qubits = int(operator.num_qubits)
                    except Exception:
                        # fallback: attempt to infer from primitive shape (not always possible)
                        n_qubits = None
                    if n_qubits is None:
                        logger.warning("Operator qubit number ambiguous; falling back to exact diagonalization.")
                    else:
                        if not _is_power_of_two(2 ** n_qubits):
                            logger.warning("Operator qubit number inconsistent; falling back to exact diagonalization.")
                        else:
                            ansatz = RealAmplitudes(n_qubits, reps=1)
                            optimizer = COBYLA(maxiter=maxiter)
                            backend_sim = Aer.get_backend('statevector_simulator')
                            vqe = VQE(ansatz, optimizer=optimizer, quantum_instance=backend_sim)
                            res = vqe.compute_minimum_eigenvalue(operator=operator)
                            return {"energy": float(res.eigenvalue.real), "state": None, "method": "vqe", "runtime": time.time() - t0}
            except Exception as e:
                logger.exception("VQE attempt failed: %s", e)
        # If we can't run VQE, fall back to exact methods below

    # Final fallback: if operator is matrix-like, perform exact diagonalization
    if isinstance(operator, np.ndarray):
        w, v = np.linalg.eigh(operator)
        idx = int(np.argmin(w))
        energy = float(w[idx])
        state = v[:, idx]
        return {"energy": energy, "state": state, "method": "fallback_exact", "runtime": time.time() - t0}

    raise RuntimeError("Unsupported operator type for ground_state_solver")
