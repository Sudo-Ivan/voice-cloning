# Voice Cloning

## 001

[Chatterbox TTS](https://github.com/resemble-ai/chatterbox)

### Install and Run

Python 3.12 is required.

```bash
pip install git+https://github.com/Sudo-Ivan/voice-cloning
```

or

```bash
uv venv .venv --python 3.12.10
source .venv/bin/activate
uv pip install git+https://github.com/Sudo-Ivan/voice-cloning
```

```bash
voice-clone
```

use `--cuda` if you have a NVIDIA GPU (fallback to CPU if unavailable)

#### Development

**Requirements**
- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation) or [uv](https://docs.astral.sh/uv/)

```bash
pyenv install 3.12
```

```bash
poetry env use 3.12
poetry install
poetry run python voice_cloning_001/main.py 
```

use `--cuda` if you have a NVIDIA GPU (fallback to CPU if unavailable)

**uv**

```bash
uv python install 3.12.10
uv venv .venv --python 3.12.10
source .venv/bin/activate && uv pip install chatterbox-tts gradio
```