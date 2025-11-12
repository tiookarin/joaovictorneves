# demos/mission_critical_demo.py
"""
Script demo executável para executar uma missão de formação.

Exemplo:
    python demos/mission_critical_demo.py --robots 3 --target 10,10 --backend simulator --plot --save
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging

from quantum_swarm_mission import FormationMission

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_target(s: str):
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("target must be X,Y")
    return np.array([float(parts[0]), float(parts[1])])


def plot_trajectories(traj: np.ndarray, target: np.ndarray, outpath: Path):
    T, n, _ = traj.shape
    plt.figure(figsize=(6, 6))
    for i in range(n):
        plt.plot(traj[:, i, 0], traj[:, i, 1], label=f"robot_{i}")
        plt.scatter(traj[0, i, 0], traj[0, i, 1], marker="o", s=30, alpha=0.6)
        plt.scatter(traj[-1, i, 0], traj[-1, i, 1], marker="x", s=30)
    plt.scatter([target[0]], [target[1]], c="red", marker="*", s=120, label="target")
    plt.legend()
    plt.title("Trajetórias do Enxame")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Demo QuantumSwarm Formation")
    parser.add_argument("--robots", type=int, default=3, help="Número de robôs (default: 3)")
    parser.add_argument("--target", type=parse_target, default=np.array([10.0, 10.0]), help="Alvo X,Y")
    parser.add_argument("--backend", type=str, default="simulator", choices=["simulator", "ibmq_manila", "ibmq_lima"], help="Backend")
    parser.add_argument("--plot", action="store_true", help="Gerar plot")
    parser.add_argument("--save", action="store_true", help="Salvar outputs em outputs/")
    args = parser.parse_args()

    mission = FormationMission(n_robots=args.robots, target_pos=args.target, backend=args.backend)
    results = mission.execute_full_cycle(backend=args.backend)

    outdir = Path("outputs")
    if args.save:
        outdir.mkdir(parents=True, exist_ok=True)
        # salvar métricas
        with open(outdir / "metrics_summary.json", "w") as f:
            json.dump(results, f, indent=2, default=lambda o: (o.tolist() if hasattr(o, "tolist") else str(o)))
        # relatório simples
        with open(outdir / "mission_report.txt", "w") as f:
            f.write("QuantumSwarm Formation - Report\n")
            f.write(f"n_robots: {args.robots}\n")
            f.write(f"target: {args.target.tolist()}\n")
            f.write(f"final_cohesion: {results['simulation']['metrics']['final_cohesion']:.3f}\n")
            f.write(f"total_time: {results['total_time']:.3f}s\n")
    if args.plot:
        traj = results["simulation"]["trajectories"]
        plot_trajectories(traj, args.target, outdir / "quantum_formation_completed.png" if args.save else Path("quantum_formation_completed.png"))
        logger.info("Plot salvo.")

    print("Resultados principais:")
    print(f"  final_cohesion: {results['simulation']['metrics']['final_cohesion']:.3f}")
    print(f"  total_time: {results['total_time']:.3f}s")


if __name__ == "__main__":
    main()