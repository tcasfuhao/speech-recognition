# Whisper Seq2Seq models

Whisper is an encoder-decoder speech-to-text model. Its encoder converts audio into internal features, then its decoder autoregressively generates the transcription one token at a time, with each generated token conditioned on the audio and the preceding generated tokens.

Seq2Seq is the general name for this encoder-decoder pattern; it is not a separate model family in this project. The trainer named `train_seq2seq.py` is the Whisper trainer, and both IPA-Whisper Base and Whisper Large-v3 use it.
