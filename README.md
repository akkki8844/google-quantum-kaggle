# google-quantum-kaggle

My work on the Google Quantum AI **Verifiable Quantum Circuit Compilation** Kaggle competition
(invsqrt / DQI / DF-THC chemistry circuits in kickmix).

- `PLAN.md`: strategy, the scoring rubric, and a log of every experiment with its numbers.
- `tools/`: analysis tools (score math, chemistry cost model, invsqrt fast exact checker,
  liveness / dependency / role tracers, angle refit LP, Kaggle kernel push helper).
- `gen/`: our own generator experiments.
- `kaggle_nb/`: the Kaggle kernel scripts we run (verification, profiling, training screens).

Competition data, verifiers, the starter kit and other participants' circuits are **not** included
(the competition rules forbid redistributing the data). Public work by other participants that we
studied is credited in `PLAN.md`; the Syed Asad Ali lineage NOTICE applies to any circuit derived
from that line.

License: Apache-2.0.
