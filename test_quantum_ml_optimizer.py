# tests/test_quantum_ml_optimizer.py
import numpy as np
from core.quantum_ml_optimizer import QuantumMLOptimizer, cohesion_loss, formation_loss, energy_efficiency_loss

def test_loss_functions_simple():
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    slots = np.array([[0.0, 0.0], [1.0, 0.0]])
    controls = np.array([[0.1, 0.0], [0.1, 0.0]])
    assert cohesion_loss(positions) >= 0.0
    assert formation_loss(positions, slots) == 0.0
    assert energy_efficiency_loss(controls) > 0.0

def test_qnn_training_runs():
    n = 3
    rng = np.random.default_rng(seed=0)
    initial_positions = rng.normal(scale=1.0, size=(n,2))
    slots = np.stack([np.array([i*0.5, 0.0]) for i in range(n)])
    ml = QuantumMLOptimizer(num_qubits=n, optimizer="adam", lr=0.01)
    res = ml.train(initial_positions=initial_positions, slots=slots, iters=5, lambdas={"cohesion":1.0, "formation":1.0, "energy":0.1})
    assert "params" in res
    assert "history" in res
    assert len(res["history"]["loss"]) == 5