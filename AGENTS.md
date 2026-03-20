# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-first PyTorch project for YaTC training. Top-level entry points are `pre-train.py` for masked autoencoder pretraining, `fine-tune.py` for downstream classification, `data_process.py` for converting flow PCAPs into 40x40 MFR PNGs, `engine.py` for train/eval loops, and `models_YaTC.py` for model definitions. Shared helpers live in `util/` (`misc.py`, schedulers, positional embeddings, LARS, cropping). Datasets are not stored in git; training code expects image folders under `./data/<dataset>/{train,test}/<class>/*.png`. Generated checkpoints and TensorBoard logs are typically written under `./output_dir/`.

## Build, Test, and Development Commands
Use Python 3.8 with the versions documented in `readme.md` (`torch==1.9.0`, `timm==0.3.2`, `numpy==1.19.5`, `scikit-learn==0.24.2`).

- `python pre-train.py --batch_size 128 --blr 1e-3 --steps 150000 --mask_ratio 0.9`: run pretraining.
- `python fine-tune.py --blr 2e-3 --epochs 200 --data_path ./data/ISCXVPN2016_MFR --nb_classes 7`: fine-tune on a dataset split.
- `python fine-tune.py --eval --finetune ./output_dir/pretrained-model.pth --data_path ./data/ISCXVPN2016_MFR --nb_classes 7`: evaluate a saved checkpoint.
- `python -m py_compile pre-train.py fine-tune.py engine.py models_YaTC.py data_process.py util/*.py`: fast syntax smoke check before pushing changes.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, grouped imports, `snake_case` for functions, variables, and CLI flags, and descriptive class/model names that match current usage such as `MAE_YaTC` and `TraFormer_YaTC`. Keep changes local and avoid broad reformatting; there is no formatter or linter config checked in. Prefer relative repository paths like `./data/...` and `./output_dir/...` over machine-specific absolute paths.

## Testing Guidelines
There is no committed automated test suite yet. For any code change, run the `py_compile` smoke check and validate the affected path with at least one short train or eval invocation on a small dataset subset. When changing dataset handling, confirm both `train/` and `test/` folder layouts still load through `torchvision.datasets.ImageFolder`.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects such as `Update data_process.py`. Keep that tone, but make subjects more specific when possible, for example `Tune fine-tune defaults for CICIoT2022`. Pull requests should state the dataset used, exact command(s) run, metric impact (`acc1`, weighted F1), and any checkpoint or output-path changes. Do not commit large datasets, model weights, or notebook-generated artifacts.
