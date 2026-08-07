from pathlib import Path
from os import mkdir
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "data/full-unedited-data"
CSV_DIR = BASE_DIR / "csvs"
AUDIO_SEGMENTS_DIR = BASE_DIR / "audio_segments"

AUDIO_SEGMENTS_DIR.mkdir(exist_ok=True, parents=True)

csv_files = {
    "1_tone.csv": CSV_DIR / "1_tone.csv",
    "capital_letter.csv": CSV_DIR / "capital_letter.csv",
    "contains_comma.csv": CSV_DIR / "contains_comma.csv",
    "hashtag.csv": CSV_DIR / "hashtag.csv",
    "many_tones.csv": CSV_DIR / "many_tones.csv",
    "multiple_question_markers.csv": CSV_DIR / "multiple_question_markers.csv",
    "quotes.csv": CSV_DIR / "quotes.csv"
}

for name in csv_files:
    csv_path = csv_files[name]

    if not csv_path.exists():
        print(f"CSV file {csv_path} does not exist, skipping.")
        continue

    output_dir = AUDIO_SEGMENTS_DIR / name.replace(".csv", "")
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header
        wav_path = line.split(",", 1)[0]

        source_audio_path = BASE_DIR / wav_path
        target_audio_path = output_dir / Path(wav_path).name

        if source_audio_path.exists():
            shutil.copy2(source_audio_path, target_audio_path)
        else:
            print(f"Audio file {source_audio_path} does not exist, skipping.")