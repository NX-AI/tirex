# Install

Install TiRex in your goto Python environment:
```sh
pip install tirex-ts
```

Install with additional input/output adapter:
```sh
pip install 'tirex-ts[gluonts,hfdataset]'
```

You can also install TiRex with all extras at once using:
```sh
pip install 'tirex-ts[all]'
```

TiRex is currently only tested on Linux and MacOS.

## Requirements
- Python 3.10 or newer.
- PyTorch installed (CUDA optional).
- For Hugging Face model download, outbound network access or pre-downloaded cache.
