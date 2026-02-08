# llm_007

`LLMmini.ipynb` is a tutorial-style Jupyter notebook that walks through building and using a small LLM pipeline end-to-end: tokenization, model components, training/evaluation utilities, loading GPT‑2 weights, instruction fine-tuning, and a small evaluation step that can query a locally running Ollama model.

## What’s in `LLMmini.ipynb`
The notebook is organized as sequential “steps” (mostly via Markdown headings). At a high level it covers:
- Tokenization (simple regex tokenizer + BPE via `tiktoken`)
- Preparing datasets / dataloaders (includes downloading small public datasets)
- Defining a GPT-like model in PyTorch and training/evaluation helpers
- Downloading and loading GPT‑2 weights (via `gpt_download3.download_and_load_gpt2`)
- Fine-tuning on instruction-style data and saving model checkpoints
- Optional evaluation by calling a local Ollama HTTP endpoint

## Requirements
- Python 3.10+ (the notebook metadata shows Python 3.12.x)
- Jupyter (or VS Code notebooks)
- Key Python packages used in the notebook (not exhaustive):
  - `torch`
  - `tiktoken`
  - `numpy`
  - `matplotlib`
  - `tqdm`
  - `psutil` (only for checking whether Ollama is running)
  - `pandas` (used in some sections)
  - `tensorflow` (referenced for a later section; installed via a `%pip` cell)

Note: The notebook contains `%pip` / `!pip` install cells. If you prefer a clean environment, install dependencies up-front in your venv and skip those cells.

## Setup (recommended)
From this repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Then install the packages you need (at minimum PyTorch + common utilities). The exact set depends on which sections you run.

## Running the notebook
Open `LLMmini.ipynb` in Jupyter or VS Code and run **top-to-bottom**.

Important notes:
- The notebook is stateful: some later cells assume earlier imports/variables exist. Running out of order may fail.
- Some sections download data/models if files are missing (see “Downloads & data”).
- Some sections can be compute-heavy depending on model size and epochs.

## Downloads & data
The notebook may download or create the following files/folders in the repo directory (if they don’t already exist):
- `the-verdict.txt` (downloaded from a public GitHub URL in one section)
- `sms_spam_collection.zip` and an extracted `sms_spam_collection/` directory (downloaded from the UCI repository)
- `instruction-data.json` (downloaded from a public GitHub URL)
- `instruction-data-with-response.json` (referenced elsewhere in the notebook)
- `gpt2/` (model weights folder created by the GPT‑2 download helper)

If you are working offline, pre-download these assets or expect those cells to fail.

## Model artifacts (saved files)
The notebook saves and loads PyTorch checkpoints (examples referenced in the notebook):
- `review_classifier.pth`
- `model.pth`
- `model_and_optimizer.pth`
- `*-sft.pth` (name derived from the chosen GPT‑2 config)

Security note: `torch.load()` uses Python pickle under the hood. Only load checkpoints you trust.

## Ollama integration (optional)
One section queries a locally running Ollama server:
- URL: `http://localhost:11434/api/chat`
- Default model string in the notebook: `"llama3"`

To use it:
- Install Ollama and start it before running that section.
- Ensure the model name you pass (e.g. `llama3`) exists locally in your Ollama installation.

## Security & safety notes
- The notebook performs network downloads (datasets, JSON, GPT‑2 weights). Review URLs before running in restricted environments.
- Checkpoints should be treated as untrusted input unless you created them yourself.
- The notebook now uses verified TLS contexts for `urllib` downloads (no SSL verification disabling).

## Troubleshooting
- **Import errors**: install the missing package(s) in your active environment, then re-run the cell.
- **CUDA not available**: the notebook will fall back to CPU if CUDA isn’t detected; training will be slower.
- **Ollama errors**: confirm Ollama is running and reachable at `localhost:11434` before the query section.
