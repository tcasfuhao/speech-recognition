# ASR Finetuning

GPU-ready preparation, training, language-model, inference, and evaluation tools for any language dataset. The repository contains code, configuration, and auditable CSV/JSON logs. Heavy data and model artifacts live with the configured external dataset.

## Storage layout

The default `data_root` is:

```text
~.../language-downloads/yonghe-qiang_01/
├── <source recordings and annotations>
├── normalised/<timestamp>/       # produced by ../data-normalisation
└── processed/
    ├── splits/wav/               # extracted clips
    ├── asr/                      # trained models and inference output
    └── lm/                       # LM corpus, ARPA, and KenLM binary
```

The lightweight records of exactly what was used remain here:

```text
speech-recognition/
├── config/
├── logs/
│   ├── prep/yonghe_qiang/
│   │   ├── metadata.csv
│   │   ├── skip_metadata.csv
│   │   └── splits/{train,dev,test}.csv
│   └── evaluation/
├── scripts/
├── src/
└── legacy/                       # retired, non-imported code
```

All paths are explicit in YAML configuration. Update the shared root in the configs if the dataset moves.

## Requirements

This project must be run in the shared `tcas_asr_python3.10` Conda environment. Create it once with Python 3.10, activate it, and install the requirements from any shared project:

```bash
conda create -n tcas_asr_python3.10 python=3.10
conda activate tcas_asr_python3.10
python -m pip install -r requirements.txt
```

`ffmpeg` is required to read and split some recording formats.

## 1. Normalize transcripts separately

Run the sibling `data-normalisation` workflow first. It writes a timestamped copy beneath `<data_root>/normalised/` and keeps its own normalization logs. Then set `annotations_dir` in `config/prep/prepare.yaml` to that exact run. The original recordings are found separately through `audio_root`.

ASR preparation keeps the normalised transcriptions and their single word-boundary spaces intact; it does not rewrite clips to mono/16 kHz. Model loaders perform required channel conversion and resampling in memory during training or inference.

Training configs use `remove_spaces: true` by default. The model loader removes all Unicode whitespace from targets in memory, without changing the manifests or the normalised source data. Set it to `false` for CTC, Whisper, or Granite when a researcher deliberately wants the ASR model to learn spaces.

The choice is saved with each trained model. Inference reads it automatically; `remove_spaces: true` or `false` in the inference config can override it. CER always ignores whitespace and is reported only as `cer`—there is no space-sensitive CER metric. The unchanged spaced normalisation run remains the gold input for the sibling `space-recognition` project.

## 2. Extract clips and make splits

```bash
python scripts/prepare_asr_training.py --config config/prep/prepare_yq.yaml
```

Stage 1 writes clips to `<data_root>/processed/splits/wav/` and writes local `metadata.csv` and `skip_metadata.csv` logs. Stage 2 writes the local 80/10/10 train, development, and test manifests plus `split_summary.json`.

Stages can be selected independently:

```bash
python scripts/prepare_asr_training.py --config config/prep/prepare_yq.yaml --start_stage 2 --stop_stage 2
```

## 3. Validate and train an ASR backend

Every checkpoint has a dedicated YAML file and an explicit active `backend`: `ctc`, `whisper`, or `granite`. The dispatcher rejects text-only, G2P, TTS, unknown, and backend-incompatible checkpoints before it creates a run.

```bash
# Always validate first. The report stays in logs/validation/.
python -m src.finetune.train_asr --config config/finetune/ctc/finetune_cer90.yaml --validate-only

# Full training or a fixed, one-epoch smoke subset.
python -m src.finetune.train_asr --config config/finetune/ctc/finetune_cer90.yaml
python -m src.finetune.train_asr --config config/finetune/whisper/ipa_whisper_base_cer90.yaml --smoke
```

The supported configurations are MMS and XLS-R (CTC), IPA-Whisper Base and Whisper Large-v3 (Whisper Seq2Seq), and Granite 4.0 Speech (multimodal LoRA). Large-v3 and Granite default to BF16 LoRA with gradient checkpointing. Granite uses its required `<|audio|>` chat prompt and multimodal processor rather than the Whisper collator. Allosaurus experiments are retired and retained only beneath `legacy/`; they are not accepted by the active dispatcher.

Training reads local split manifests and external clips. All checkpoints and models are written beneath `<data_root>/processed/asr/<backend>/`; validation, smoke configuration, manifests, predictions, failures, and summaries stay in `logs/`.

### Run several training configurations sequentially

The queue runner validates every configuration before starting the first job, then launches each trainer in a separate Python process. CTC is a training objective/backend; MMS and XLS-R are two different pretrained models that use it. Every queued model starts independently from the checkpoint named in its own training YAML—weights do not carry from one job to the next.

The included comparison runs MMS CTC, XLS-R CTC, IPA Whisper Base, Whisper Large-v3, and Granite in that order. A full run is expensive, so validate or smoke-test it first:

```bash
python -m src.finetune.train_queue --config config/finetune/queues/yonghe_model_comparison.yaml --validate-only
python -m src.finetune.train_queue --config config/finetune/queues/yonghe_model_comparison.yaml --smoke
python -m src.finetune.train_queue --config config/finetune/queues/yonghe_model_comparison.yaml
```

Queue YAML paths are resolved relative to the queue file. Job names and training configs must be unique. By default, a failure stops the queue; set `stop_on_failure: false` in YAML or pass `--continue-on-error` to attempt the remaining jobs.

Each run writes `queue_state.json` and one terminal log per job beneath `logs/queues/<queue-name>/<timestamp>/`. The state records the model output directory produced by each trainer. Resume an interrupted or failed run explicitly; successful jobs are skipped and incomplete jobs restart from their original training configuration:

```bash
python -m src.finetune.train_queue --resume logs/queues/yonghe_model_comparison/<timestamp>
```

## 4. Build KenLM outside the repository

Clone and compile KenLM at the exact revision used by the Python binding. The
default `kenlm_path` is `~/projects/download-projects/kenlm/build/bin`:

```bash
cd ~/projects/download-projects
git clone https://github.com/kpu/kenlm.git
cd kenlm
git checkout 4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
mkdir -p build
cd build
cmake ..
cmake --build . --parallel
```

Then run:

```bash
python -m src.lm.build_kenlm --config config/lm/lm_yq.yaml
```

Only the local training manifest is used to build the language model. The generated corpus, ARPA file, and binary are stored beneath `<data_root>/processed/lm/`.
KenLM is optional: it is loaded only when inference is given an `lm_path`;
otherwise CTC uses greedy decoding. Silero VAD is not part of this workflow.

## 5. Inference and evaluation

Replace `<run>` placeholders in the relevant YAML with the selected model or inference run, then execute:

```bash
python -m src.inference.transcribe --config config/inference/inference_yq.yaml
python -m src.evaluation.evaluate_preds --config config/evaluation/evaluation_yq.yaml
python -m src.evaluation.plot_train_log --config config/evaluation/plot_train_log_yq.yaml
```

Inference output remains with the external model data. Evaluation summaries and plots are written under this repository's `logs/evaluation/` directory.

## CER-filtered manifests

The optional filtering helper now keeps all resulting metadata and split files under `logs/prep/yonghe_qiang/cer90/`:

```bash
python src/data/minus_10_percent.py
```
