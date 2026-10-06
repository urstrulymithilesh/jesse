"""Generate the checked-in notebook after changing its helper or configuration."""

import hashlib
import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent


def build_notebook():
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {},
                "source": dedent(source).strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """
    # Train “yo Jesse” on Colab's free GPU tier

    Select **Runtime → Change runtime type → T4 GPU**, then run cells in order.
    If a free GPU is unavailable, stop and try later. No paid plan, compute-unit
    purchase, Drive mount, personal recording or account token is needed here.
    [Colab's limits](https://research.google.com/colaboratory/faq.html#resource-limits)
    vary; completion within a free session is not guaranteed. Use the notebook UI.

    This uses synthetic speech and public datasets. It does **not** clone your voice.
    Budget roughly 25 GB of downloads and at least 60 GiB free disk before setup.
    The largest single input is 17.28 GB. Download the result before the VM expires.
    An expired VM loses its data; rerunning within the same VM reuses downloads.
    Training restarts if interrupted; this is not a persistent checkpoint service.

    Status: locally tested orchestration; a full Colab GPU run is still required.
    A completed ONNX export is a candidate, not accepted wake-word quality.
    Based on the [openWakeWord automatic trainer](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb).
    """)
    add("code", """
    import hashlib, json, pathlib, shutil, subprocess, sys, urllib.request
    ROOT = pathlib.Path('/content/jesse-wake')
    if not pathlib.Path('/content').is_dir():
        raise RuntimeError('Run this notebook in a Colab Linux GPU runtime.')
    subprocess.run(['nvidia-smi'], check=True)
    ROOT.mkdir(exist_ok=True)
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    if free_gib < 60 and not (ROOT / 'environment.txt').exists():
        raise RuntimeError(f'Only {free_gib:.1f} GiB free; need 60 GiB before setup.')
    print(f'{free_gib:.1f} GiB free. These cells do not request paid resources.')

    def run(args):
        subprocess.run([str(x) for x in args], cwd=ROOT, check=True)
    """)
    hashes = {name: hashlib.sha256((HERE / name).read_text(encoding="utf-8").encode()).hexdigest()
              for name in ("wake_support.py", "yo_jesse.json")}
    add("code", f"""
    # Retrieve only public training helpers; verify the version this notebook expects.
    expected = {hashes!r}
    base = 'https://raw.githubusercontent.com/urstrulymithilesh/jesse/main/training/'
    for name, digest in expected.items():
        content = urllib.request.urlopen(base + name, timeout=60).read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError('Notebook/helper versions differ. Reopen the latest notebook.')
        (ROOT / name).write_bytes(content)
    sys.path.insert(0, str(ROOT))
    from wake_support import OPENWAKEWORD_REV, PIPER_REV, fix_boolean_defaults, write_once

    # A separate Python 3.10 venv avoids changing Colab's ML kernel packages.
    run([sys.executable, '-m', 'pip', 'install', 'uv==0.8.22'])
    UV = [sys.executable, '-m', 'uv']
    PY = ROOT / '.venv/bin/python'
    if not PY.exists():
        run(UV + ['venv', '--python', '3.10', str(ROOT / '.venv')])
    run(['apt-get', 'update', '-qq'])
    run(['apt-get', 'install', '-y', '-qq', 'espeak-ng', 'libsndfile1', 'ffmpeg', 'build-essential'])
    """)
    add("code", """
    def checkout(url, name, revision):
        path = ROOT / name
        if not path.exists():
            run(['git', 'clone', url, path])
            run(['git', '-C', path, 'checkout', '--detach', revision])
        actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise RuntimeError(f'{name}: wrong revision; use a fresh runtime.')
        return path

    OWW = checkout('https://github.com/dscripka/openWakeWord.git', 'openwakeword', OPENWAKEWORD_REV)
    PIPER = checkout('https://github.com/dscripka/piper-sample-generator.git', 'piper-sample-generator', PIPER_REV)
    run(UV + ['pip', 'install', '--python', str(PY), 'torch==2.5.1', 'torchaudio==2.5.1',
              '--index-url', 'https://download.pytorch.org/whl/cu121'])
    requirements = [
        'numpy==1.26.4', 'scipy==1.13.1', 'scikit-learn==1.5.2',
        'onnx==1.17.0', 'onnxruntime-gpu==1.20.2', 'PyYAML==6.0.2',
        'torchinfo==1.8.0', 'torchmetrics==1.2.0', 'speechbrain==0.5.16',
        'audiomentations==0.33.0', 'torch-audiomentations==0.11.0',
        'acoustics==0.2.6', 'mutagen==1.47.0', 'pronouncing==0.2.0',
        'datasets==2.14.6', 'pyarrow==14.0.2', 'huggingface-hub==0.25.2',
        'deep-phonemizer==0.0.19', 'espeak-phonemizer==1.3.1',
        'webrtcvad-wheels==2.0.14', 'librosa==0.10.2.post1', 'soundfile==0.12.1',
        'setuptools<81', 'requests', 'tqdm',
    ]
    run(UV + ['pip', 'install', '--python', str(PY)] + requirements)
    # ONNX only: skip optional LiteRT/Speex packages and a second ORT wheel.
    run(UV + ['pip', 'install', '--python', str(PY), '--no-deps', '-e', str(OWW)])
    source = (OWW / 'openwakeword/train.py').read_text()
    write_once(ROOT / 'train_onnx.py', fix_boolean_defaults(source))
    (ROOT / 'environment.txt').write_text(subprocess.check_output(
        UV + ['pip', 'freeze', '--python', str(PY)], text=True))

    def script(name, content):
        path = ROOT / name
        write_once(path, content)
        run([PY, '-u', path])
    """)
    add("markdown", """
    ## Check the stack before large dataset downloads
    Load the public multi-speaker generator, synthesize two test clips, and run the
    CUDA feature extractor. Failure stops the workflow before the 17 GB download.
    These clips test setup only; they are excluded from training and acceptance.
    """)
    add("code", r'''
    script('probe.py', """
import pathlib, subprocess, sys
import numpy as np
import torch
import openwakeword.train
from openwakeword.utils import AudioFeatures
root = pathlib.Path(__file__).parent
sys.path.insert(0, str(root / 'piper-sample-generator'))
from generate_samples import generate_samples
if not torch.cuda.is_available():
    raise RuntimeError('No CUDA GPU. Choose a free GPU runtime or try later.')
print(torch.cuda.get_device_name(0), flush=True)
model = root / 'piper-sample-generator/models/en-us-libritts-high.pt'
if not model.exists():
    part = model.with_suffix('.part')
    subprocess.run(['wget', '-c', '-q', '--show-progress', '-O', str(part),
        'https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt'], check=True)
    if part.stat().st_size != 255226835:
        raise RuntimeError('Incomplete generator download')
    part.rename(model)
resources = root / 'openwakeword/openwakeword/resources/models'
resources.mkdir(parents=True, exist_ok=True)
for name in ('melspectrogram.onnx', 'embedding_model.onnx'):
    target = resources / name
    if not target.exists():
        part = target.with_suffix('.part')
        subprocess.run(['wget', '-q', '-O', str(part),
            'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/' + name], check=True)
        part.rename(target)
features = AudioFeatures(device='gpu')
if features.onnx_execution_provider != 'CUDAExecutionProvider':
    raise RuntimeError('ONNX CUDA failed; inspect the library error above.')
embedded = features.embed_clips(np.zeros((1, 32000), dtype=np.int16))
if embedded.shape[-1] != 96 or not np.isfinite(embedded).all():
    raise RuntimeError('Feature extractor returned invalid embeddings')
probe = root / 'setup-probe'
probe.mkdir(exist_ok=True)
generate_samples(['yo Jesse'], str(probe), max_samples=2, batch_size=2)
if len(list(probe.glob('*.wav'))) < 2:
    raise RuntimeError('Synthetic speech setup did not produce its probe clips')
print('GPU, imports, embeddings, and synthetic speech probe passed.', flush=True)
""")
    ''')
    add("markdown", """
    ## Public data and configuration
    Fetch separate training and validation features, MIT room responses, and one
    AudioSet background shard. Dataset revisions are pinned. Negative features use
    memory mapping, so the entire file is not loaded into RAM together.
    This baseline data mix still needs real room and headset evaluation.
    """)
    add("code", r'''
    script('prepare_data.py', """
import json, math, pathlib, tarfile
import datasets
import numpy as np
import scipy.signal
import soundfile as sf
import yaml
from huggingface_hub import hf_hub_download
from wake_support import check_features, make_config, write_once
root = pathlib.Path(__file__).parent
def fetch(repo, name, revision):
    return pathlib.Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision))
feature_rev = '985bf1b47e7f19c07741af82bfe32d5a9dc56096'
features = fetch('davidscripka/openwakeword_features',
    'openwakeword_features_ACAV100M_2000_hrs_16bit.npy', feature_rev)
validation = fetch('davidscripka/openwakeword_features', 'validation_set_features.npy', feature_rev)
for path, is_validation in ((features, False), (validation, True)):
    print(path.name, check_features(path, validation=is_validation), flush=True)

def save_audio(path, audio, rate):
    if path.exists():
        return
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(int(rate), 16000)
        audio = scipy.signal.resample_poly(audio, 16000 // divisor, int(rate) // divisor)
    sf.write(path, audio, 16000, subtype='PCM_16')

rirs = root / 'mit_rirs'
rirs.mkdir(exist_ok=True)
rir_data = datasets.load_dataset('davidscripka/MIT_environmental_impulse_responses',
    revision='b824a1ef2821f112fda0b9cb26e4278c62b425bb', split='train', streaming=True)
for i, row in enumerate(rir_data):
    save_audio(rirs / f'{i:05d}.wav', row['audio']['array'], row['audio']['sampling_rate'])
archive = fetch('agkphysics/AudioSet', 'data/bal_train09.tar',
    '0c609e8302cf139307f639c57652032af0a88041')
raw = root / 'audioset'
if not (root / 'audioset-extracted').exists():
    with tarfile.open(archive) as tar:
        tar.extractall(raw, filter='data')
    (root / 'audioset-extracted').touch()
background = root / 'background'
background.mkdir(exist_ok=True)
for i, path in enumerate(sorted(raw.rglob('*.flac'))):
    audio, rate = sf.read(path)
    save_audio(background / f'{i:06d}.wav', audio, rate)
if not list(background.glob('*.wav')) or not list(rirs.glob('*.wav')):
    raise RuntimeError('Missing augmentation audio')
defaults = yaml.safe_load((root / 'openwakeword/examples/custom_model.yml').read_text())
overlay = json.loads((root / 'yo_jesse.json').read_text())
config = make_config(defaults, overlay, root, features, validation)
write_once(root / 'yo_jesse.yaml', yaml.safe_dump(config, sort_keys=True))
print(yaml.safe_dump(config), flush=True)
""")
    ''')
    add("markdown", """
    ## Generate, augment, then train
    Run one cell at a time. Logs remain in the runtime. Generation can resume in
    the same VM. Augmentation explicitly regenerates derived feature files on each
    attempt, because interrupted augmentation can leave incomplete arrays.
    Training uses 20,000 synthetic positives plus negatives, 2,000 validation
    positives plus negatives, and 50,000 classifier steps. Duration is unmeasured.
    """)
    add("code", """
    def train_stage(flag, log_name):
        command = [str(PY), '-u', str(ROOT / 'train_onnx.py'),
                   '--training_config', str(ROOT / 'yo_jesse.yaml'), flag]
        if flag == '--augment_clips':
            command.append('--overwrite')
        with (ROOT / log_name).open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end='')
                    log.write(line)
                if process.wait():
                    raise RuntimeError(f'{flag} failed; see {log_name}. Do not continue.')
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
    train_stage('--generate_clips', 'generate.log')
    """)
    add("code", "train_stage('--augment_clips', 'augment.log')")
    add("code", "train_stage('--train_model', 'train.log')")
    add("markdown", """
    ## Download the candidate
    The bundle contains ONNX, a provenance/hash report, config, installed versions
    and logs. Copy **yo_jesse.onnx** into Jesse's local **models/** folder.
    Keep the default wake unchanged until held-out human WAVs pass **wake-check**
    and live interruption/echo checks. See **training/README.md** for commands.
    The notebook cannot approve its own training data.
    """)
    add("code", r'''
    script('export_candidate.py', """
from pathlib import Path
from wake_support import package_model
root = Path(__file__).parent
bundle = package_model(root)
(root / 'latest-bundle.txt').write_text(str(bundle))
print(bundle, flush=True)
""")
from google.colab import files
files.download((ROOT / 'latest-bundle.txt').read_text())
    ''')
    return {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"name": "Train yo Jesse.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }, "nbformat": 4, "nbformat_minor": 4}


if __name__ == "__main__":
    (HERE / "train_yo_jesse.ipynb").write_text(
        json.dumps(build_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
