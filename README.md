```markdown
# QuantumSwarm Formation — Documentação Essencial

Este repositório contém uma implementação híbrida (clássica + quântica *mockable*) para controle ótimo e formação de enxames de robôs baseada na ideia HJB-FP → Hamiltoniano quântico. O código foi escrito para ser executável localmente sem necessidade imediata de hardware quântico ou ROS — mas também preparado para integrar Qiskit / qiskit-nature / qiskit-machine-learning e ROS quando disponíveis.

Resumo rápido:
- Padrão: n_robots = 3 (configuração usada nas demos e exemplos)
- Estrutura: módulos em `core/`, demos em `demos/`, testes em `tests/`, CI em `.github/workflows/`
- Objetivo: oferecer um *pipeline* end-to-end que vai do encoding de features → Hamiltoniano → solver de estado fundamental → otimização por gradiente → QNN → deploy simulado ROS → análise de desempenho

Índice
- Visão Geral
- Instalação Rápida
- Execução Rápida
- Estrutura do Projeto & Descrição dos Módulos
- Notebooks e Demos
- Testes & CI
- Integração com Qiskit / Hardware / ROS
- Notas sobre Performance e Aceleração GPU
- Contribuição e Licença

---

Visão Geral
----------
O projeto explora uma arquitetura híbrida:
1. Mapear os termos HJB-FP (controle + difusão) para um Hamiltoniano qubitizado.
2. Resolver problemas contínuos por VQE/QNN (aqui: implementações de fallback que permitem testes locais).
3. Resolver atribuição (robô → slot) com um solver estilo QAOA (aqui: Hungarian ou heurística).
4. Simular a dinâmica unicycle com ruído para validar coesão e formação.
5. Oferecer ponte com ROS para deploy/experimentação com robôs reais (simulada se ROS não estiver presente).

Instalação Rápida
-----------------
Recomendado: criar um ambiente virtual.

Linux / macOS:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Observação:
- Qiskit, qiskit-nature e qiskit-machine-learning são opcionais; o código tem *fallbacks* que usam numpy para garantir execução sem dependências quânticas.
- Para integração ROS, instale/execute dentro de um workspace ROS (rospy é tipicamente instalado via pacotes do sistema e não via pip).

Execução Rápida
---------------
Demo CLI (rodar exemplo completo):
```bash
python demos/mission_critical_demo.py --robots 3 --target 10,10 --backend simulator --plot --save
```

Resultado esperado (quando `--save`):
- outputs/metrics_summary.json
- outputs/mission_report.txt
- outputs/quantum_formation_completed.png (quando `--plot`)

Uso programático:
```python
from quantum_swarm_mission import FormationMission
import numpy as np

mission = FormationMission(n_robots=3, target_pos=np.array([10.0, 10.0]))
results = mission.execute_full_cycle(backend='simulator')
print(results['simulation']['metrics']['final_cohesion'])
```

Estrutura do Projeto & Descrição dos Módulos
-------------------------------------------
Top-level:
- quantum_swarm_mission.py
  - Facade de alto nível que corre o pipeline mínimo: construir Hamiltoniano → VQE fallback → QAOA assignment → simulação unicycle.
  - Parâmetros: n_robots (default 3), target_pos, backend.

Módulos principais em `core/`:
- core/hamiltonian_builder.py
  - Constrói um Hamiltoniano denso em numpy com termos: controle (Z_i Z_j), formação (Z_i), terminal (Z_i) e entropia/difusão (X_i X_{i+1}).
  - Útil como referência e para testes locais; conversão para operadores Qiskit pode ser adicionada.

- core/vqe_formation_optimizer.py
  - Implementa um VQE "mock": diagonalização direta (exact diagonalization) para obter o menor autovalor e autovetor em ambientes sem Qiskit.
  - Quando Qiskit estiver disponível, pode ser estendido para usar `VQE` real.

- core/qaoa_assignment_solver.py
  - Resolve problema de atribuição robô→slot.
  - Usa `scipy.optimize.linear_sum_assignment` (Hungarian) quando disponível; fallback guloso caso contrário.
  - Interface: solve_assignment(cost_matrix) → (assignment, cost, runtime).

- core/swarm_quantum_simulator.py
  - Simula dinâmica unicycle simplificada com ruído browniano.
  - Oferece compute_cohesion() para medir coesão do enxame.
  - Parâmetros ajustáveis: dt, steps, noise_sigma.

- core/quantum_ml_optimizer.py
  - Otimizador híbrido com:
    - Otimizadores clássicos: SGD, RMSprop, Adam (simples, implementados localmente).
    - Gradiente por Parameter-Shift (numérico/analítico básico).
    - ZZFeatureMap builder (quando Qiskit presente retorna QuantumCircuit; senão um *feature_map* numpy).
    - QuantumNeuralNetwork (QNN) wrapper: usa Qiskit Machine Learning quando disponível; senão, uma função-fallback que emula o comportamento.
    - Loss functions: cohesion_loss, formation_loss, energy_efficiency_loss.
    - Treinamento de QNN via `QuantumMLOptimizer.train()` com history e runtime.

- core/qiskit_nature_swarm.py
  - Ferramentas para modelos fermiónicos e mapeamento Jordan-Wigner.
  - build_fermionic_hamiltonian: cria um operador em segunda quantização (placeholder quando qiskit-nature ausente).
  - build_ising_formation: constrói Hamiltoniano Ising denso (numpy) que codifica objetivos de formação.
  - ground_state_solver: resolve estado fundamental (diagonalização exata ou VQE via Qiskit, quando disponível).

- core/ros_quantum_bridge.py
  - Ponte para ROS (opcional):
    - RobotController: envia metas/publishing (simulado se `rospy` não estiver presente).
    - SwarmCoordinator: coordenação simples com múltiplos controllers.
    - generate_launch_file: cria um launch file XML de exemplo em `outputs/ros_launch/`.
    - deploy_to_ros_swarm: faz dispatch simulado de metas e captura odometria (snapshot).

Pipeline Avançado:
- advanced_quantum_mission.py
  - Implementa as 9 fases integradas (Feature encoding, Hamiltoniano fermiónico, ground state, gradient VQE, QNN training, loss optimization, quantum kernel mock, ROS deploy simulado, simulação + análise).
  - Projetado para rodar localmente mesmo sem dependências quânticas; quando Qiskit estiver presente, executa caminhos reais para VQE/Mapping/Estimator.

Demos e Notebooks
-----------------
- demos/mission_critical_demo.py — CLI executável (ex.: --robots, --target, --backend, --plot, --save).
- demos/notebook_vqe_live.ipynb — Notebook simples para visualizar espectro e experimentar com VQE-fallback; contém instruções para alternar quando Qiskit estiver presente.

Testes & CI
-----------
- Testes unitários em `tests/`:
  - tests/test_hamiltonian_builder.py
  - tests/test_quantum_ml_optimizer.py
  - tests/test_qiskit_nature_swarm.py

- CI:
  - `.github/workflows/ci.yml` já incluído. O workflow possui uma matrix mínima que roda com/sem Qiskit instalado:
    - `minimal` job: instala dependências básicas e roda pytest.
    - `qiskit` job: instala Qiskit + nature + ML e roda pytest (pode demorar mais).

Observações sobre testes:
- Os testes detectam presença de bibliotecas e usam implementações de fallback para garantir que os testes passem em ambientes sem Qiskit.
- Se optar por exigir Qiskit no CI, ajuste o workflow para instalar explicitamente as dependências (considere caches para acelerar builds).

Integração com Qiskit / Hardware / ROS
--------------------------------------
- Qiskit:
  - O código contém *hooks* para integrar ansätze reais, Estimator/QNN e VQE quando Qiskit e qiskit-machine-learning estiverem instalados.
  - Para fazer uso real de VQE/QAOA, substitua os fallbacks por `PauliSumOp`/`SparsePauliOp` e utilize `VQE`/`QAOA` de `qiskit.algorithms`.

- qiskit-nature:
  - `core/qiskit_nature_swarm.py` tem placeholders que facilitam a substituição por `SecondQuantizedOp` → `PauliSumOp` usando `JordanWignerMapper`.

- ROS:
  - `core/ros_quantum_bridge.py` suporta execução simulada sem ROS; se `rospy` estiver disponível e o sistema rodando um ROS master, o módulo tentará publicar goals.
  - `generate_launch_file()` produz um launch XML básico que pode ser adaptado ao seu stack (ex.: TurtleBot/Gazebo).

Performance & Aceleração GPU
---------------------------
- O código tem chamadas condicionais a CuPy (proxy para aceleração GPU) quando disponível.
- `cuQuantum` está listado como opcional em `requirements.txt` — sua utilização exige GPUs NVIDIA compatíveis e drivers CUDA apropriados.
- Para perfis reais em hardware quântico, siga práticas de mitigação de erro (Qiskit Ignis / Runtimes) e mapeamento/topologia de qubits.

Boas Práticas e Limitações
--------------------------
- Limitação NISQ: circuit depth, número de qubits e fidelidade HW limitam escalabilidade (os módulos here são projetados para N ≤ 5 qubits por padrão).
- Fallbacks: todos os módulos críticos têm implementações de fallback (numpy/classical) para permitir desenvolvimento sem hardware/stack quântico/ROS.
- Segurança: nunca inclua tokens/API keys no repositório; use secrets no GitHub Actions para credenciais IBM Quantum.

Contribuindo
-----------
1. Fork e clone o repositório.
2. Crie uma branch feature: `git checkout -b feature/your-feature`
3. Escreva testes para novas funcionalidades.
4. Abra Pull Request com descrição e referência aos testes.

Comandos úteis:
```bash
# instalar requisitos
pip install -r requirements.txt

# rodar testes
pytest -q

# executar demo
python demos/mission_critical_demo.py --robots 3 --target 10,10 --plot --save
```

Licença
-------
Este projeto segue a licença MIT. Veja o arquivo `LICENSE` para detalhes.
```
