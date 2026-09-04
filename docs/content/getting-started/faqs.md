# FAQs

In the following section, we address common questions and potential issues you might encounter. If your question is not answered here, please feel free to reach out to us by opening an issue on [Github](https://github.com/NX-AI/tirex/issues/new?template=BLANK_ISSUE).

<details>
  <summary>What is TiRex?</summary>

  TiRex is a 35M parameter pre-trained time series forecasting model. It is built upon the [xLSTM](https://github.com/NX-AI/xlstm) architecture and was developed by [NXAI](https://www.nx-ai.com/). For a comprehensive overview, please see our dedicated [Introduction](./../introduction) section.
</details>

<details>
  <summary>How many parameters does TiRex have?</summary>

  TiRex has **35 million (35M) parameters**.
</details>

<details>
  <summary>Why are my TiRex forecasts running slowly?</summary>

  If you are experiencing slow forecasts, please check the following:
  - Enable model compilation when loading the model.
  - Use hardware acceleration devices like CUDA (for NVIDIA GPUs) or MPS (for Apple Silicon) whenever possible.
  - For maximum speed, consider using our custom backends.
  - Trade accuracy for speed with `model.forecast(..., full_rollout=True)` (predicts the whole horizon in one forward pass) or `dynamic_padding=True` (pads short contexts to the next patch instead of the full training context length). See the [forecasting workflow](../how-to/forecasting/workflow.md).

  A quick estimation of the speed–accuracy trade-off (Setting: batch size 5, context length 128, prediction length 64, `torch` backend, average of 50 calls; MASE and CRPS are lower-is-better):

  | Option | CPU | GPU | MASE (Gifteval) | CRPS (Gifteval) |
  | --- | --- | --- | --- | --- |
  | Default (both off) | 416 ms | 233 ms | 0.724 | 0.499 |
  | `full_rollout=True` | 223 ms (~1.9x) | 116 ms (~2.0x) | 0.753 (+4%) | 0.519 (+4%) |
  | `dynamic_padding=True` | 44 ms (~9x) | 30 ms (~8x) | 0.734 (+1.5%) | 0.508 (+2%) |

  The speedup from `dynamic_padding` shrinks as your context approaches the full 2,048 steps, and the accuracy cost of `full_rollout` grows with the prediction length. More detailed analysis and full setup can be seen in: [PR #34](https://github.com/NX-AI/tirex/pull/34).

  For detailed instructions on optimization, please refer to the API.
  If you are interested in optimizing TiRex for dedicated hardware platforms (especially for edge or embedded use cases), please get in touch: [contact@nx-ai.com](mailto:contact@nx-ai.com)
</details>

<details>
  <summary>What is the maximum prediction length (context length) of TiRex?</summary>

  Currently, TiRex supports a maximum **context length of up to 2,048**.

  We are actively working on a version that supports longer contexts. In the meantime, if you need to forecast longer horizons, you may try **downsampling your time series data**.

  To stay updated on the progress of this feature, please follow [this Github issue](https://github.com/NX-AI/tirex/issues/27).

</details>

<details>
  <summary>Which data input formats does the framework support?</summary>

  Out of the box, we support **NumPy** arrays and **PyTorch** tensors.
  To use other popular formats, you can install the respective extras:
  - **GluonTS datasets**: Install with `tirex-ts[gluonts]`
  - **Hugging Face datasets**: Install with `tirex-ts[hfdataset]`
</details>

<details>
  <summary>How can I finetune TiRex on my own data?</summary>

  Fine-tuning is offered as a commercial service by [NXAI](https://nx-ai.com). Please get in touch with us directly at [contact@nx-ai.com](mailto:contact@nx-ai.com) to discuss your needs.
</details>
