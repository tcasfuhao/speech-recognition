# ASR Model Training and Comparison

GPU-ready tools for preparing speech data, training multiple supported ASR models, and comparing their predictions across datasets and transcription choices. YAML configurations define each model and dataset; queues run independent training or inference jobs and save auditable CSV/JSON results. The included language comparisons are worked examples of this reusable workflow. Heavy data and model artifacts live with the configured external dataset.

## Storage layout

The included configurations use an external `data_root` with a layout like this:

```text
~.../language-downloads/<language>/
├── normalised/<timestamp>/         # produced by ../data-normalisation
└── processed/
    ├── splits/wav/                 # extracted clips
    ├── asr/                        # trained models grouped by model family
    │   ├── mms/<model_timestamp>/
    │   ├── xlsr/<model_timestamp>/
    │   └── ipa-whisper-base/<model_timestamp>/
    └── lm/                         # LM corpus, ARPA, and KenLM binary
```

The lightweight records of exactly what was used remain here:

```text
speech-recognition/
├── config/
├── logs/
│   ├── prep/<dataset_run>/
│   │   ├── metadata.csv
│   │   ├── skip_metadata.csv
│   │   └── splits/{train,dev,test}.csv
│   └── evaluation/
├── scripts/
├── src/
└── legacy/                       # retired code
```

All paths are explicit in YAML configuration. Update the shared root in the configs if the dataset moves.

## Requirements

Use Python 3.10 and install the dependencies in `requirements.txt`. For example, with Conda:

```bash
conda create -n asr-models python=3.10
conda activate asr-models
python -m pip install -r requirements.txt
```

`ffmpeg` is required to read and split some recording formats.

## 1. Normalise transcripts separately

Run the sibling `data-normalisation` workflow first. It writes a timestamped copy beneath `<data_root>/normalised/` and keeps its own normalisation logs. Then set `annotations_dir` in `config/prep/prepare.yaml` to that exact run. The original recordings are found separately through `audio_root`.

ASR preparation keeps the normalised transcriptions and their single word-boundary spaces intact; it does not rewrite clips to mono/16 kHz. Model loaders perform required channel conversion and resampling in memory during training or inference.

Training configs use `remove_spaces: true` by default. The model loader removes all Unicode whitespace from targets in memory, without changing the manifests or the normalised source data. Set it to `false` when you want the ASR model to learn spaces.

The choice is saved with each trained model. Inference reads it automatically, and evaluates it using the same configuration. `remove_spaces: true` or `false` in the inference config overrides it. There is no space-sensitive CER metric, i.e. `cer_with_space` vs `cer_without_space`, it simply all gets squashed into one.

## 2. Extract clips and make splits

Stage 1 writes clips to `<data_root>/processed/splits/wav/` and writes local `metadata.csv` and `skip_metadata.csv` logs. Stage 2 first creates the local 80/10/10 train, development, and test splits, then deterministically shuffles and caps each split at its configured duration target while retaining the final clip that crosses the target. A null target keeps the complete split. The manifests and their configured and retained durations are recorded in `split_summary.json`.

```bash
python scripts/prepare_asr_training.py --config config/prep/prepare.yaml
```

Preparation outputs are immutable timestamped runs. For example, `logs_dir: logs/prep/<dataset>` writes to `logs/prep/<dataset>_<timestamp>/`; an existing directory is never reused. The command prints both its preparation run ID and resolved directory. Set `prep_dir` in a fine-tuning YAML to that exact directory; training then reads `metadata.csv` and `splits/{train,dev,test}.csv` from it. Any `metadata`, `train_csv`, `dev_csv`, or `test_csv` value explicitly set in the YAML overrides its derived path.

Stages can be selected independently:

```bash
python scripts/prepare_asr_training.py --config config/prep/prepare.yaml --prep_run_id <timestamp-from-stage-1> --start_stage 2 --stop_stage 2
```

Resuming after stage 1 requires its printed `--prep_run_id`; this prevents stage 2 from silently reading a different run.

## 3. Validate and train an ASR backend

Every checkpoint has a dedicated YAML file and an explicit active `backend`: `ctc`, `whisper`, or `granite`. The dispatcher rejects text-only, G2P, TTS, unknown, and backend-incompatible checkpoints before it creates a run.

```bash
# Always validate first. The report stays in logs/validation/.
python -m src.finetune.train_asr --config config/finetune/ctc/facebook/wav2vec2-xls-r-1b.yaml --validate-only

# Full training or a fixed, one-epoch smoke subset.
python -m src.finetune.train_asr --config config/finetune/ctc/facebook/wav2vec2-xls-r-1b.yaml --smoke
python -m src.finetune.train_asr --config config/finetune/ctc/facebook/wav2vec2-xls-r-1b.yaml
```

The supported configurations can be found at `src/finetune/asr_config.py`, however, this is just a simple check. The project explicitly supports the models listed in `asr_config.py`; other models may be compatible, but they must be added and validated before this training workflow accepts them.

`facebook/wav2vec2-xls-r-1b` illustrates how to assess a future CTC checkpoint: it is a candidate because it uses the Wav2Vec2/XLS-R architecture expected by the CTC trainer. Inspect the Hugging Face configuration to confirm `model_type: wav2vec2`, then verify that its feature extractor and `AutoModelForCTC` load successfully with the project-generated vocabulary. A base XLS-R checkpoint need not include a project-specific CTC head because fine-tuning replaces or initializes that head for the vocabulary. Once the load check succeeds, add the model ID to the dispatcher's explicit supported-model list and run `--validate-only`; the current validator rejects unregistered model IDs before it performs its architecture check.

Training reads local split manifests and external clips. All checkpoints and models are written beneath `<data_root>/processed/asr/<model-family>/`; validation, smoke configuration, manifests, predictions, failures, and summaries stay in `logs/`.

### Run several training configurations sequentially

List the training YAMLs you want to compare in a queue file. The queue runner validates every configuration before starting the first job, then launches each trainer in a separate Python process. CTC is a training objective/backend; MMS and XLS-R are two different pretrained models that use it. Every queued model starts independently from the checkpoint named in its own training YAML -- weights do not carry from one job to the next.

The included normalisation comparison queue is an example with MMS CTC, XLS-R CTC, and IPA-Whisper Base. Adapt its job list and training YAMLs for your datasets and supported models. A full run is expensive, so prepare the manifests and validate or smoke test the queue first:

```bash
python -m src.finetune.train_queue --config config/finetune/queues/queue_comparison.yaml --validate-only
python -m src.finetune.train_queue --config config/finetune/queues/queue_comparison.yaml --smoke
python -m src.finetune.train_queue --config config/finetune/queues/queue_comparison.yaml
```

Queue YAML paths are resolved relative to the queue file. Job names and training configs must be unique. By default, a failure stops the queue; set `stop_on_failure: false` in YAML or pass `--continue-on-error` to attempt the remaining jobs.

Each run writes `queue_state.json` and one terminal log per job beneath `logs/queues/<queue-name>/<timestamp>/`. For the included comparison configs, validation reports use the same batch ID beneath `logs/validation/comparison/<timestamp>/<language>/<edition>/`; preflight, job-start, and resumed validations receive unique timestamped filenames and are never overwritten. Standalone validation creates its own timestamped batch. The state records the validation batch and model output directory produced by each trainer. Resume an interrupted or failed run explicitly; successful jobs are skipped and incomplete jobs restart from their original training configuration:

In an interactive terminal, queued jobs show the trainer's live overall training bar, a separate development evaluation bar, and metric lines. After evaluation, training continues at its existing overall step count on a new line. Standalone trainers show the same bars. When queue output is redirected, it prints occasional readable progress snapshots. Each job's `.log` keeps completed bars and metric lines as plain text; redraws update the current bar instead of adding a line for every step. After a failure or interruption, the final line of that job's log shows the last training or evaluation position reached. The `train_log.tsv` metrics and evaluation and checkpoint schedules are unchanged.

```bash
python -m src.finetune.train_queue --resume logs/queues/normalisation_model_comparison/<timestamp>
```

#### Worked example: 45-model normalisation comparison

The comparison configuration defines 45 language–edition–model combinations across Japhug, Yonghe-Qiang, and Yongning-Na. The training queue selects which of those jobs to run. The five completed editions are `unnormalised`, `tones`, `map-chars`, `brackets`, and `full`; the models are MMS CTC, XLS-R CTC, and IPA-Whisper Base. Preparation is intentionally per language-edition: every config extracts only the selected tier, discards texts blanked by that edition, and creates its own deterministic utterance-level 80/10/10 splits. All language manifests are then capped at approximately 2 hours of training audio and 12 minutes each of development and test audio.

All models that consume a language-edition manifest use the same capped, deterministic rows. Duration limits belong only in preparation configuration; no model-specific cap is added to CTC, Whisper, or Granite fine-tuning YAMLs.

```bash
# Prepare all language-edition manifests beneath one comparison batch timestamp.
comparison_prep_id=$(date +%Y%m%d_%H%M%S)
for config in config/prep/comparison/*.yaml; do python scripts/prepare_asr_training.py --config "$config" --prep_run_id "$comparison_prep_id"; done

# Pin config/finetune/comparison/*.yaml to the emitted
# logs/prep/comparison/<timestamp>/... paths before validation or training.

# This checks manifest separation, audio paths, and model configs.
python -m src.finetune.train_queue --config config/finetune/queues/queue_comparison.yaml --validate-only

# Smoke test the jobs currently listed in the queue.
python -m src.finetune.train_queue --config config/finetune/queues/queue_comparison.yaml --smoke

# Launch those independent training jobs.
python -m src.finetune.train_queue --config config/finetune/queues/queue_comparison.yaml
```

The production queue continues after a failed job and records every terminal status, validation report, and completed output path in its queue state.

Comparison preparation inserts the shared run ID immediately below `logs/prep/comparison/`, giving `logs/prep/comparison/<timestamp>/<language>/<edition>/`. Reusing a timestamp for the same language-edition is rejected rather than overwritten.

All training backends write timestamped runs beneath a short model-family directory such as `<data_root>/processed/asr/mms/`, `asr/xlsr/`, or `asr/ipa-whisper-base/`. Backend, `comparison`, and normalisation-edition wrapper directories are not used. Every new run includes a `NOTE.md` describing its base model, backend, source configuration, manifests, text policy, and training schedule; use that note to distinguish normalisation editions within each model-family directory.

## 4. Build KenLM outside the repository

Clone and compile KenLM at the exact revision used by the Python binding. The default `kenlm_path` is `~/projects/download-projects/kenlm/build/bin`:

```bash
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

Replace the relevant information from the YAML files with the selected model or inference run that you are working with. Then execute:

```bash
python -m src.inference.transcribe --config config/inference/inference.yaml
python -m src.evaluation.evaluate_preds --config config/evaluation/evaluation.yaml
python -m src.evaluation.plot_train_log --config config/evaluation/plot_train_log.yaml
```

Inference output remains with the external model data. Evaluation summaries and plots are written under this repository's `logs/evaluation/` directory.
 
### Batch comparison of trained models

The inference queue accepts a list of CTC or Whisper model configs. Each config supplies a local checkpoint or pinned Hugging Face model, its train, development, and test manifests, audio root, text policy, and saved test CER. Copy and adapt the supplied queue and job YAMLs to compare other supported models or datasets.

The included `config/inference/comparison/queue.yaml` is a worked example with 45 saved checkpoints. Run a queue on a GPU machine with the project environment installed; private Hub models require access (`hf auth login`). The commands below use this example queue:

```bash
# Check model access, manifests, and audio without transcribing.
python -m src.inference.transcribe_queue --config config/inference/comparison/queue.yaml --validate-only

# Pilot one complete model, including all three splits, and compare its rerun test CER.
python -m src.inference.transcribe_queue --config config/inference/comparison/queue.yaml --job japhug-unnormalised-mms-ctc

# Evaluate every model listed in this queue. A one-clip-per-split check can use --limit 1.
python -m src.inference.transcribe_queue --config config/inference/comparison/queue.yaml

# Continue a failed or interrupted run, skipping completed splits.
python -m src.inference.transcribe_queue --resume logs/evaluation/comparison_inference/<run-id>
```

Every invocation creates a timestamped directory beneath `logs/evaluation/comparison_inference/`. It contains `queue_state.json`, per-split predictions and logs, and `summary.csv` with one row per completed model. Its `train_cer`, `dev_cer`, and `test_cer` columns are mean CER for each split; `dev_test_cer` pools the development and test utterances, and `train_dev_test_cer` pools all three splits. `saved_test_cer` is the test CER from the original training run, while `test_cer` is measured again by this inference run. `dev_test_minus_train_cer` and `test_minus_saved_test_cer` are unscaled decimal differences: for example, `0.03` means a CER difference of `0.03` (three percentage points). Pooled means are weighted by the number of scored utterances. `--limit` writes `partial_summary.csv` instead; it never produces a full-run summary. A failed split remains visible in the state file and is rerun from its start on resume. Scoring matches training: mean per-utterance CER, punctuation and whitespace ignored, empty references skipped. A model enters the summary only after all expected rows in all three splits have predictions. Per-utterance CER values remain in each split's `preds_scored.csv`.
