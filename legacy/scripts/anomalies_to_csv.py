
from os import mkdir
import re
import pandas as pd
# print(pd.__version__)
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed/asr/finetune/mms-1b-all_20260529_161500/best_20260531_141654"
DATA_NO_BEAM = PROJECT_ROOT / "data/processed/asr/finetune/mms-1b-all_20260529_161500/best_20260531_124419/preds_scored.csv"
DATA_BEAM = PROJECT_ROOT / "data/processed/asr/finetune/mms-1b-all_20260529_161500/best_20260531_141654/preds_scored.csv"
BASE_DIR = PROJECT_ROOT / "data/full-unedited-data"
OUTPUT_DIR = BASE_DIR / "csvs"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
(BASE_DIR / "with_beam").mkdir(exist_ok=True, parents=True)
(BASE_DIR / "without_beam").mkdir(exist_ok=True, parents=True)


df1=pd.read_csv(DATA_DIR / "preds_scored.csv")
df2=pd.read_csv(BASE_DIR / "metadata.csv")
df3=pd.read_csv(DATA_NO_BEAM)
df4=pd.read_csv(DATA_BEAM)

'''
higher_than_80 = df1[df1['cer'] > 0.80]
higher_than_80.to_csv(BASE_DIR / "higher_than_80.csv", index=False)
'''

contains_hashtag = df2[df2['text'].str.contains('#')]
contains_multiple_question_markers = df2[df2["text"].str.contains(r'\?{2,}')]
contains_3_or_more_numbers = df2[df2['text'].str.contains(r'\d{3,}')]
contains_1_number = df2[df2['text'].str.contains(r'\D\d\D')]
contains_capital_letter = df2[df2['text'].str.contains(r'[A-Z]')]

contains_comma = df2[df2['text'].str.contains(',', na=False)]
contains_comma.to_csv(OUTPUT_DIR / "contains_comma.csv", index=False)

with open(BASE_DIR / "metadata.csv", encoding="utf-8") as f:
    lines = f.readlines()

quotes = [line for line in lines if re.search(r'"[^"]*"', line)]



contains_hashtag.to_csv(OUTPUT_DIR / "hashtag.csv", index=False)
contains_multiple_question_markers.to_csv(OUTPUT_DIR / "multiple_question_markers.csv", index=False)
contains_3_or_more_numbers.to_csv(OUTPUT_DIR / "many_tones.csv", index=False)
contains_1_number.to_csv(OUTPUT_DIR / "1_tone.csv", index=False)
contains_capital_letter.to_csv(OUTPUT_DIR / "capital_letter.csv", index=False)
with open(OUTPUT_DIR / "quotes.csv", "w", encoding="utf-8") as f:
    f.writelines(quotes)

for i in range(1, 11):
    speaker = f"SP{i:02d}"  # SP01 ... SP10
    df3[df3['speaker_id'] == speaker].to_csv(
        BASE_DIR / f"without_beam/{speaker.lower()}.csv",
        index=False
    )

for i in range(1, 11):
    speaker = f"SP{i:02d}"  # SP01 ... SP10
    df4[df4['speaker_id'] == speaker].to_csv(
        BASE_DIR / f"with_beam/{speaker.lower()}.csv",
        index=False
    )

