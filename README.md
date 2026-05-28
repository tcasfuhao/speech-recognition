# Yonghe Qiang ASR (Wav2Vec2 fine-tuning)

Local, GPU-ready pipeline to prepare ELAN data and fine-tune a CTC ASR model
without Colab/Google Drive. This mirrors the structure of the Lo-Rig
`speech-recognition` project but is tailored to the Yonghe Qiang dataset.

## Project layout
```
asr-models/yonghe-qiang
├─ config/
│  ├─ prep/prepare_yq.yaml
│  ├─ split/split_yq.yaml
│  ├─ finetune/finetune_yq.yaml
│  └─ inference/inference_yq.yaml
├─ data/
│  ├─ raw/                # copy the dataset here
│  └─ processed/
│     ├─ wav/             # 16k mono segments
│     ├─ metadata.csv
│     ├─ skip_metadata.csv
│     └─ splits/
├─ scripts/
│  └─ prepare_asr_training.py
└─ src/
   ├─ prep/               # EAF ingestion + segmentation
   ├─ data/               # schema + split helpers
   ├─ finetune/           # Wav2Vec2 training
   └─ inference/          # batch transcription
```

## Requirements
Create a Python 3.10+ conda environment and install:
```bash
conda create -n yq_asr python=3.10
conda activate yq_asr
python -m pip install -r requirements.txt
```

**System dependencies:** `ffmpeg` is required by `pydub` for some audio formats.

## 1) Copy raw data into the repo
From WSL, copy the dataset into `data/raw/`:
```bash
rsync -a "/mnt/c/Users/tcasf/Downloads/GrosFichiers - Nathan-002/Yonghe Qiang/" \
  ./data/raw/
```

Expected raw structure (example):
```
data/raw/
├─ YH-758/
│  ├─ <recording>.wav
│  ├─ <recording>.eaf
│  ├─ <recording>.pfsx
│  └─ *.txt
├─ YH-999/
└─ YH-868/
```

## 2) Prepare segments + metadata + 80/10/10 splits
```bash
python scripts/prepare_asr_training.py --config config/prep/prepare_yq.yaml
```

This generates:
```
data/processed/
├─ wav/
├─ metadata.csv
├─ skip_metadata.csv
└─ splits/
   ├─ train.csv
   ├─ dev.csv
   ├─ test.csv
   └─ split_summary.json
```

Note the cleaner removes all punctuation marks, as well hashtag symbols that were present in the original transcription 

## 3a) Fine-tune CTC Models
```bash
python -m src.finetune.train_ctc --config config/finetune/ctc/finetune_yq_cer90.yaml
```
python -m src.finetune.train_ctc --config config/finetune/ctc/finetune_yq.yaml


## 3b) Fine-tine Seq2Seq Models
```bash
python -m src.finetune.train_seq2seq --config config/finetune/seq2seq/finetune_yq_cer90.yaml
```
python -m src.finetune.train_ctc --config config/finetune/seq2seq/finetune_yq_cer90.yaml

Outputs:
```
data/processed/asr/finetune/<run_name>/
├─ checkpoints/
├─ best/
├─ vocab.json
├─ train_log.tsv
└─ test_metrics.json
```

### Training on CER-filtered data (top 10% noisiest removed)
First generate the filtered metadata from the scored predictions:
```bash
python src/data/minus_10_percent.py
```

This writes `data/processed/metadata_cer90.csv` and
`data/processed/splits_cer90/removed_noisy_top10.csv`, then rebuilds
`data/processed/splits_cer90/train.csv`, `dev.csv`, `test.csv`, and
`split_summary.json`.

Then train with the CER-filtered data. This uses
`data/processed/splits_cer90/` and `metadata_cer90.csv`:
```bash
python -m src.finetune.train_ctc --config config/finetune/ctc/finetune_yq_cer90.yaml
```
python -m src.finetune.train_seq2seq --config config/finetune/seq2seq/finetune_yq_cer90.yaml


## 4) Inference (batch transcription)
```bash
python -m src.inference.transcribe --config config/inference/inference_yq.yaml
```

Predictions are saved under:
```
data/processed/asr/inference/<run_name>/
```

## 5) Evaluation + plots
Score predictions with CER summaries:
```bash
python -m src.evaluation.evaluate_preds --config config/evaluation/evaluation_yq.yaml
```

Plot training curves from `train_log.tsv`:
```bash
python -m src.evaluation.plot_train_log --config config/evaluation/plot_train_log_yq.yaml
```

## Notes
- Tier selection is automatic: any ELAN tier with non-empty annotations is used.
- If you need to restrict tiers, edit `include_tier_regex` / `exclude_tier_regex`
  in `config/prep/prepare_yq.yaml`.
- Optional LM decoding requires `pyctcdecode` and a KenLM binary. Set `lm_path`
  in the inference config if you use it.
