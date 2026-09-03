# Granite Speech models

Granite Speech is a speech-capable instruction/chat model. It receives audio together with a chat prompt, then autoregressively generates the assistant's text response. During fine-tuning, the prompt tokens are masked so that loss is calculated only on the intended transcription response.

Although Granite and Whisper both generate text autoregressively, Granite uses its own processor inputs, chat-prompt formatting, label masking, generation flow, and LoRA setup. It therefore has a dedicated Granite trainer rather than sharing the Whisper Seq2Seq trainer.
