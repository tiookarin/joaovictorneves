# tests/test_qiskit_nature_swarm.py
import numpy as np
from core.qiskit_nature_swarm import build_fermionic_hamiltonian, jordan_wigner_transform, build_ising_formation, ground_state_solver

def test_build_fermionic_and_map():
    op = build_fermionic_hamiltonian(3, coupling_scale=0.5)
    mapped = jordan_wigner_transform(op)
    assert mapped is not None

def test_ising_and_groundstate():
    H = build_ising_formation(3, coupling=0.5)
    res = ground_state_solver(H, method="exact")
    assert "energy" in res
    assert "state" in res