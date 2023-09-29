# Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis

[Audio samples](https://anonymous5425.github.io/anonymous-submission/)


## Installation

To use Vocos only in inference mode, clone the repository, and run from the root directory:

```bash
pip install .
```

If you wish to train the model, install it with additional dependencies:

```bash
pip install .[train]
```

## Usage

### Reconstruct audio from mel-spectrogram

```python
import torch

from vocos import Vocos

vocos = Vocos.from_pretrained(<anonymized>)

mel = torch.randn(1, 100, 256)  # B, C, T

with torch.no_grad():
    audio = vocos.decode(mel)
```

Copy-synthesis from a file:

```python
import torchaudio

y, sr = torchaudio.load(YOUR_AUDIO_FILE)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

with torch.no_grad():
    y_hat = vocos(y)
```

### Reconstruct audio from EnCodec

Additionally, you need to provide a `bandwidth_id` which corresponds to the lookup embedding for bandwidth from the
list: `[1.5, 3.0, 6.0, 12.0]`.

```python
vocos = Vocos.from_pretrained(<anonymized>)

quantized_features = torch.randn(1, 128, 256)
bandwidth_id = torch.tensor([3])  # 12 kbps

with torch.no_grad():
    audio = vocos.decode(quantized_features, bandwidth_id=bandwidth_id)  
```

Copy-synthesis from a file: It extracts and quantizes features with EnCodec, then reconstructs them with Vocos in a
single forward pass.

```python
y, sr = torchaudio.load(YOUR_AUDIO_FILE)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)

with torch.no_grad():
    y_hat = vocos(y, bandwidth_id=bandwidth_id)
```

## Training

Prepare a filelist of audio files for the training and validation set:

```bash
find $TRAIN_DATASET_DIR -name *.wav > filelist.train
find $VAL_DATASET_DIR -name *.wav > filelist.val
```

Fill a config file, e.g. [vocos.yaml](configs%2Fvocos.yaml), with your filelist paths and start training with:

```bash
python train.py -c configs/vocos.yaml
```

Refer to [Pytorch Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for details about customizing the
training pipeline.
