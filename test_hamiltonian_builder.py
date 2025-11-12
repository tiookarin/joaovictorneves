# tests/test_hamiltonian_builder.py
import numpy as np
from core.hamiltonian_builder import build_hamiltonian, pauli_z, pauli_x


def test_pauli_ops_shape():
    Z = pauli_z(2, 0)
    X = pauli_x(2, 1)
    assert Z.shape == (4, 4)
    assert X.shape == (4, 4)


def test_build_hamiltonian_hermitian():
    H = build_hamiltonian(n_qubits=3, alpha=1.0, beta=0.5, gamma=0.2, delta=0.1)
    # hermitian: H == H.conj().T
    assert np.allclose(H, H.conj().T, atol=1e-8)


def test_build_hamiltonian_dimension():
    for n in range(1, 5):
        H = build_hamiltonian(n_qubits=n)
        assert H.shape == (2 ** n, 2 ** n)