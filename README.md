# GermEval2021

This repo contains the code for GermEval 2021. For details, see https://germeval2021toxic.github.io/SharedTask.

## Quick start

- Install PyTorch 1.1.0 (see instructions [here](https://pytorch.org)).
- Install other requirements in `requirements.txt`.

### Prepare data

```shell
cd src
python prepare_data.py
```

### Train

```shell
cd bash
chmod +x *
./<filename>
```

- `<filename>` is `cv_*` for cross validation based training.
- `<filename>` is `train_best_models.sh` for training best models.

### Predict

```shell
cd bash
chmod +x *
./predict_using_best_models.sh
```

### Evaluate

```shell
cd src
python prepare_references.py
python evaluate.py  # enter predictions' file path inside file before running
```

### Other utilities

- Prepare submission files

```shell
cd src
python prepare_submission_files.py
```

- Analyze training logs

```shell
cd src
python read_log_file.py  # enter log file path inside file before running
```