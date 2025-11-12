# core/hamiltonian_builder.py
"""
Constrói um Hamiltoniano simplificado a partir dos coeficientes fornecidos.

Nota: Para propósitos de desenvolvimento e testes, o Hamiltoniano é representado
como uma matriz Hermitiana d x d (d = 2^n_qubits) em numpy. Em integração com Qiskit,
poderíamos converter para um Operator ou SparsePauliOp.
"""
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def pauli_z(n: int, target: int) -> np.ndarray:
    """Matriz Z aplicada ao qubit 'target' em espaço de n qubits."""
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    ops = [np.eye(2) for _ in range(n)]
    ops[target] = Z
    mat = ops[0]
    for op in ops[1:]:
        mat = np.kron(mat, op)
    return mat


def pauli_x(n: int, target: int) -> np.ndarray:
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    ops = [np.eye(2) for _ in range(n)]
    ops[target] = X
    mat = ops[0]
    for op in ops[1:]:
        mat = np.kron(mat, op)
    return mat


def build_hamiltonian(n_qubits: int, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0, delta: float = 0.1) -> np.ndarray:
    """
    Constrói uma matriz de Hamiltoniano simplificado:
      H = alpha * H_ctrl + beta * H_form + gamma * H_term + delta * H_entropy

    - H_ctrl : acoplamento Z_i Z_j (sum nearest neighbor)
    - H_form : soma de Z_i (pesos de formação)
    - H_term : soma de Z_i (termo terminal aproximado)
    - H_entropy: soma de X_i X_{i+1} (difusão)

    Retorna:
        np.ndarray hermitiana (2^n x 2^n)
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be >= 1")

    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=float)

    # H_form + H_term: termo local Z
    for i in range(n_qubits):
        H += (beta + gamma) * pauli_z(n_qubits, i)

    # H_ctrl: pairwise nearest-neighbor Z_i Z_j (circular)
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        H += alpha * (pauli_z(n_qubits, i) @ pauli_z(n_qubits, j))

    # H_entropy: X_i X_{i+1}
    for i in range(n_qubits - 1):
        H += delta * (pauli_x(n_qubits, i) @ pauli_x(n_qubits, i + 1))

    # tornar hermitiano explícito numérico (simetria)
    H = 0.5 * (H + H.T.conj())
    logger.debug("Hamiltoniano construído com dimensão %d", dim)
    return H