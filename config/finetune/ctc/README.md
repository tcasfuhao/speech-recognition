# CTC models

Connectionist Temporal Classification (CTC) uses an audio encoder to produce a probability distribution over transcription tokens for every audio frame. The output is decoded without an autoregressive text decoder: repeated tokens and blank tokens are collapsed into the final transcription.

The MMS and XLS-R checkpoints in this directory are Wav2Vec2-style CTC models. They use the same CTC trainer because they have the same audio-encoder and frame-level token-prediction structure, even though they begin with different pretrained weights.
