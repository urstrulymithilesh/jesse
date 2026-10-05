# Custom wake word and own-voice training

Runtime stays local. These are training instructions, not evidence that custom
models have been trained. Keep datasets/checkpoints in ignored `data/` or `models/`.
Use a separate Linux/GPU environment for training; do not replace the runtime venv.

## Record the voice pilot locally

Close Jesse so it does not share the microphone. List endpoints with
`.venv\Scripts\python.exe -m jesse devices`, then start:

```powershell
.venv\Scripts\python.exe -m jesse voice-record --device 1 --output-device 4
```

Those indices were the measured Realtek endpoints; verify the current list first.
The recorder waits for Enter, counts down, captures eight seconds, and plays the take
back. Keep only takes that match the prompt exactly, have the full sentence and sound
clean. Speak in the accent and delivery you want Jesse to use. Clipped/near-silent
takes are rejected. Accepted WAVs and `metadata.csv` stay in `data/voice-training/`.
Quit any time; rerunning skips accepted prompts. Nothing is uploaded or trained.

The 40 original prompts are a **pilot**, not a sufficient production dataset.
Add diverse, accurately transcribed prompts with `--prompts FILE --limit N`.
Use `--seconds 12` for longer sentences. Aim initially for at least 30 minutes of
reviewed speech, then judge held-out quality and collect more as needed. That is a
project collection target, not a quality guarantee. Keep a separate audition script
out of training. Do not mix another speaker into the dataset.

## Train and export the voice

Follow the [official Piper training setup](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)
in the separate environment. Its single-speaker manifest uses `filename.wav|text`,
which the recorder produces. With the dataset and a compatible fine-tuning checkpoint:

```bash
python -m piper.train fit --data.voice_name jesse \
  --data.csv_path /datasets/voice-training/metadata.csv \
  --data.audio_dir /datasets/voice-training/audio \
  --model.sample_rate 22050 --data.espeak_voice en-us \
  --data.cache_dir /training/cache --data.config_path /training/voice.json \
  --data.batch_size 8 --trainer.max_epochs 100 \
  --ckpt_path /training/base-medium.ckpt
python -m piper.train.export_onnx --checkpoint /training/chosen.ckpt \
  --output-file /training/en_US-jesse-medium.onnx
```

Batch size/epoch count are starting settings to evaluate, not an accepted recipe.
Copy the exported model and the **config written during training** into `models/`
as `en_US-jesse-medium.onnx` and `en_US-jesse-medium.onnx.json`. An inference ONNX
voice is not a fine-tuning checkpoint. Training hardware and the checkpoint are
still to be provisioned; do not upload recordings to a service without choosing it
with Mithilesh first.

Audition locally: `.venv\Scripts\python.exe -m jesse say "Good to hear from you." --voice en_US-jesse-medium`.
Listen to held-out questions, numbers, names and long sentences before accepting it.

## Train and measure the wake phrase

Use the [upstream automatic training notebook](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb).
Merge the values in `yo_jesse.json` into its configuration; retain the notebook's
dataset, augmentation and feature-file paths. It is an overlay, not a standalone
training environment. Synthetic positives and broad negatives are both required.
Export ONNX for this Windows runtime. See the [upstream configuration](https://github.com/dscripka/openWakeWord/blob/main/examples/custom_model.yml).

Place held-out, human-spoken positives (one wake per clip) and negative speech/room
audio in separate folders. They must be mono, 16-bit PCM WAV at 16 kHz.

```powershell
.venv\Scripts\python.exe -m jesse wake-check --model models/yo_jesse.onnx --positive data/wake-eval/positive --negative data/wake-eval/negative --report data/wake-eval/report.json
```

The report includes the model hash, threshold, individual clips and sampled metrics.
Exit 0 requires at least 20 positives and one hour of negatives, at least 95% detected
positives and at most 0.5 false activations/hour. Exit 1 means insufficient coverage
or a missed gate; exit 2 means invalid input/evaluation failure. Events are score
threshold crossings spaced at least one second apart. These are sampled gates,
not statistical guarantees. Synthetic-only checks cannot establish human recall.
Also test real interruption and playback echo before changing defaults.

## Try selected assets without editing defaults

```powershell
.\Start-Jesse.cmd --device 1 --gain 14.8 --threshold 1004 --wake-model models/yo_jesse.onnx --wake-phrase "yo Jesse" --wake-threshold 0.5 --voice en_US-jesse-medium
```

The wake model is used for waking and interrupting. The separate phrase controls
transcript cleanup, so a model filename never becomes the spoken name. Explicit
voice selection requires both model/config files and never falls back to a stub.
Omit custom-asset flags to keep the current stock wake word and Ryan voice.
