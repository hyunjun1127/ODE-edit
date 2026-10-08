# Source provenance

Baseline implementations retain their upstream notices and are distributed under
the corresponding licenses in `baselines/easyedit/LICENSE`,
`baselines/sphere/LICENSE`, and `baselines/blue/LICENSE`. `SOURCES.json` records
the repository, exact commit, original path/hash, installed hash and packaging
changes for each imported source file. Namespace and package-relative path
changes do not introduce a new baseline algorithm.

- [EasyEdit](https://github.com/zjunlp/EasyEdit): FT, MEMIT, AlphaEdit, MEMIT-FE, SPHERE.
- [SPHERE](https://github.com/PlusLabNLP/SPHERE): Llama3 MEMIT fallback and its dependencies.
- [BLUE](https://github.com/xpq-tech/BLUE): AlphaEdit-BLUE and its dependencies.
- [CAKE](https://github.com/zjh-vinky/CAKE): generation/metric reference used by the
  server1 shared evaluator; see `hparams/generation.lock.json` and the retained
  CAKE license in `evaluation/generation/CAKE_LICENSE`.

PRICE code comes from this repository's recorded execution-source lineage.
FLU/CON reference data is not redistributed here; runners reuse the existing
assets by exact hash. Model weights, datasets, C0/P tensors, raw generations and
checkpoints remain local assets and are excluded from Git.
