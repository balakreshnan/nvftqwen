# NVIDIA NeMo + Qwen Hands-On Series

This repository is a terminal-first, notebook-based learning path for preparing data, fine-tuning Qwen with NVIDIA NeMo, evaluating the result, and exploring three inference/deployment options. It is designed to remain useful on a CPU-only laptop: dataset work and dry runs are safe locally, while every GPU-heavy operation is explicit and opt-in.

The primary NeMo path uses `Qwen/Qwen3-1.7B`. The smaller `Qwen/Qwen2.5-1.5B-Instruct` model is included as an optional local inference path. Model downloads are disabled by default in the notebooks.

## What the four NVIDIA/serving tools do

| Tool | Role in this series | What it is not |
| --- | --- | --- |
| **NVIDIA NeMo** | Training and customization framework. Notebook 3 uses a NeMo 2.0 recipe and LoRA configuration. | It is not primarily an online API server. |
| **SGLang** | High-throughput model serving and inference with an OpenAI-compatible API. | It is not the primary fine-tuning framework in this project. |
| **TensorRT-LLM** | NVIDIA inference optimization/runtime for compiling or directly serving models with GPU-specific acceleration. | It is not a dataset preparation or training framework. |
| **NVIDIA NIM** | Packaged, supported model microservices distributed as containers, with standardized APIs and operational defaults. | It does not replace model training; supported models, adapters, and profiles depend on the selected NIM release. |

These components can form a pipeline: customize with NeMo, export a deployment-compatible checkpoint, serve directly with SGLang or optimize with TensorRT-LLM, and use NIM when a supported packaged microservice is the right production target.

## 1. Create an environment from a terminal

Run the commands from this repository's root directory.

PowerShell:

```powershell
cd nemo-qwen-hands-on-series
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

POSIX shell (Linux/macOS):

```bash
cd nemo-qwen-hands-on-series
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Python 3.12 is a conservative cross-platform choice, while the local notebook path also supports Python 3.13. On Python 3.13, the local requirements select NumPy 2.1 or newer because NumPy 1.26 has no Python 3.13 wheels. The local notebooks work on Windows, macOS, and Linux. NeMo LLM training, SGLang, TensorRT-LLM, and NIM should normally be run on a supported Linux system with NVIDIA drivers and CUDA; NVIDIA containers are often the most reproducible option for those stages.

## 2. Install dependencies

For laptop-safe data work, notebooks, and optional small-model inference:

```bash
python -m pip install -r requirements-local.txt
```

For local CUDA inference on a Windows or Linux workstation with a recent NVIDIA GPU and driver:

```bash
python -m pip install --upgrade -r requirements-gpu.txt
```

`requirements-gpu.txt` installs a matched CUDA 13.0 package family: PyTorch 2.11.0, TorchVision 0.26.0, and TorchAudio 2.11.0. These releases provide CPython 3.13 Windows wheels. Verify the installation before downloading a model:

```bash
python -c "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
python scripts/check_gpu.py
```

For the NeMo-oriented learning environment on a compatible CUDA/Linux host:

```bash
python -m pip install -r requirements.txt
```

On native Windows, `requirements.txt` installs the notebook and Hugging Face helpers but intentionally skips NeMo, which NVIDIA does not support there. The full stack is sensitive to the host CUDA driver, PyTorch build, and NeMo release. On a GPU system, prefer the PyTorch build selected for that CUDA environment and consult the current [NeMo installation guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html). SGLang, TensorRT-LLM, and NIM are introduced in their notebooks because their installation and container versions are platform-specific; they are intentionally not forced into the base Python environment.

### Python 3.13 and NumPy 1.26 troubleshooting

If pip downloads `numpy-1.26.4.tar.gz` and reports that no C compiler was found, stop that installation. NumPy 1.26 supports Python only through 3.12; installing a compiler does not make it a supported Python 3.13 combination. Upgrade the packaging tools and install the local requirements, which select a compatible NumPy wheel:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: "numpy>=2.1,<3"
python -m pip install -r requirements-local.txt
python -m pip check
```

## 3. Launch Jupyter

JupyterLab:

```bash
jupyter lab
```

Classic-style Notebook UI:

```bash
jupyter notebook
```

Open the `notebooks/` directory in Jupyter and use the kernel from `.venv`. No editor-specific extension is required.

## 4. Validate and convert the data

Validate one split:

```bash
python scripts/validate_dataset.py data/train.jsonl
```

Validate every split:

```bash
python scripts/validate_dataset.py data/train.jsonl data/validation.jsonl data/test.jsonl
```

Convert a mixed file to chat-message JSONL. Existing `messages` records are preserved; `input`/`output` records are converted:

```bash
python scripts/convert_chat_jsonl.py data/train.jsonl outputs/train_messages.jsonl
```

Inspect the local Python/PyTorch/CUDA environment:

```bash
python scripts/check_gpu.py
```

Run the small local Qwen inference example from Notebook 1 directly in a terminal:

```bash
python scripts/local_inference.py "Explain LoRA in two short sentences."
```

The default model is `Qwen/Qwen2.5-1.5B-Instruct`. Its files are downloaded from Hugging Face on first use and then reused from the local cache. CPU inference is supported but can be slow and may require roughly 8 GB or more of available system memory. Useful options include:

```bash
# Force CPU and limit the response length.
python scripts/local_inference.py "What is NeMo?" --device cpu --max-new-tokens 64

# Use the Qwen3 model introduced in the NeMo path (requires more resources).
python scripts/local_inference.py "What is LoRA?" --model Qwen/Qwen3-1.7B

# Use already-cached files without a network request.
python scripts/local_inference.py "What is LoRA?" --offline
```

Run `python scripts/local_inference.py --help` for every option. The script defaults to deterministic generation; pass a positive `--temperature` to enable sampling.

### Windows Command Prompt: NVIDIA GPU inference

The following commands are for Windows Command Prompt (`cmd.exe`), not PowerShell. Open Command Prompt and change to the project directory. The `/d` option also changes drives when the project is not on the current drive:

```cmd
cd /d C:\Code\finetuning\nvftqwen\nemo-qwen-hands-on-series
```

Activate the virtual environment:

```cmd
.venv\Scripts\activate.bat
```

The prompt should now begin with `(.venv)`. Confirm that `python` resolves to the virtual environment:

```cmd
where python
python --version
```

Upgrade the Python packaging tools, then install the complete pinned GPU environment. `requirements-gpu.txt` already selects the CUDA 13.0 PyTorch index and installs the compatible PyTorch, TorchVision, and TorchAudio packages, so a separate unpinned `pip install torch torchvision torchaudio` command is not needed:

```cmd
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade -r requirements-gpu.txt
python -m pip check
```

Verify the installed package versions and confirm that PyTorch can access the NVIDIA GPU:

```cmd
python -c "import torch, torchvision, torchaudio; print('PyTorch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('TorchAudio:', torchaudio.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
python scripts\check_gpu.py
```

Run the first local inference example on the GPU:

```cmd
python scripts\local_inference.py "Explain LoRA in two short sentences." --device cuda --dtype bfloat16 --max-new-tokens 2000
```

Run a second prompt:

```cmd
python scripts\local_inference.py "Explain quantum computing" --device cuda --dtype bfloat16 --max-new-tokens 2000
```

`--max-new-tokens 2000` is an upper limit rather than a required response length. For the first smoke test, use `--max-new-tokens 128` if you want a faster result. If generation runs out of GPU memory, reduce this value or use `Qwen/Qwen2.5-0.5B-Instruct` with `--model`.

To monitor GPU memory and utilization continuously, open a second Command Prompt window and run:

```cmd
nvidia-smi -l 1
```

Press `Ctrl+C` to stop monitoring. When finished with the virtual environment, run:

```cmd
deactivate
```

## 5. Run the notebooks in order

| Order | Notebook | Default behavior | Infrastructure |
| --- | --- | --- | --- |
| 1 | `01_install_nemo_load_qwen.ipynb` | Checks the environment; downloads nothing unless enabled. | Laptop-safe by default. Optional small-model inference benefits from a GPU and several GB of disk/RAM. |
| 2 | `02_prepare_validate_dataset.ipynb` | Validates, normalizes, and previews JSONL records. | Laptop-safe. |
| 3 | `03_finetune_lora_nemo.ipynb` | Loads config and prepares a NeMo recipe; real training is disabled. | Real training requires Linux, an NVIDIA CUDA GPU, substantial VRAM/disk, and compatible NeMo/PyTorch versions. |
| 4 | `04_evaluate_model.ipynb` | Builds evaluation prompts and scoring helpers; inference calls are disabled. | Data/eval setup is laptop-safe. Running base and tuned models requires suitable local hardware or reachable endpoints. |
| 5 | `05_serve_with_sglang.ipynb` | Shows server/client commands; no server is started automatically. | Linux + NVIDIA CUDA GPU recommended/required for practical use. |
| 6 | `06_optimize_with_tensorrt_llm.ipynb` | Explains and prints version-sensitive placeholder commands. | Advanced: Linux, supported NVIDIA GPU, CUDA, Docker/NVIDIA Container Toolkit, and ample storage. |
| 7 | `07_deploy_with_nim.ipynb` | Shows a credential-safe NIM workflow and API client. | NVIDIA GPU, Docker/NVIDIA Container Toolkit, NGC access, accepted model terms, and a supported NIM profile. |

To execute a laptop-safe notebook headlessly:

```bash
jupyter nbconvert --to notebook --execute notebooks/02_prepare_validate_dataset.ipynb --output 02_prepare_validate_dataset.executed.ipynb
```

Do not headlessly execute GPU notebooks until their opt-in flags, model paths, and hardware assumptions have been reviewed.

## 6. Laptop-safe versus GPU/cloud steps

Laptop-safe tasks include creating the environment, validating/converting JSONL, reading every notebook, building the evaluation prompt set, inspecting YAML, and running all dry-run cells. Optional CPU inference with Qwen2.5-1.5B can still be slow and memory-intensive, so it is disabled by default.

GPU or cloud infrastructure is required for realistic NeMo LoRA training, production SGLang serving, TensorRT-LLM optimization, and NIM. A CUDA GPU alone is not sufficient: verify driver/CUDA/PyTorch compatibility, model memory needs, local disk space, and the tool's supported operating system. Multi-GPU or cloud instances may be needed when experimenting with longer contexts, larger batches, or production throughput.

NGC credentials are needed only for protected NVIDIA containers or artifacts (not for the laptop-safe notebooks). Keep `NGC_API_KEY` in an environment variable or secret manager. Never paste it into a notebook, configuration file, shell history, or Git commit. Notebook 7 uses placeholders and environment-variable checks only.

## Dataset formats

The validator accepts either format on each JSONL line:

```json
{"input": "Explain LoRA briefly.", "output": "LoRA trains small low-rank adapter matrices while the base weights stay frozen."}
```

```json
{"messages": [{"role": "user", "content": "Explain LoRA briefly."}, {"role": "assistant", "content": "LoRA trains small low-rank adapter matrices while the base weights stay frozen."}]}
```

Blank content, malformed JSON, missing keys, unsupported roles, and conversations without both a user and assistant turn are reported with file and line numbers.

## Project outputs

Generated checkpoints, model exports, converted datasets, caches, and executed notebooks belong under `outputs/`, `checkpoints/`, or `models/`; these paths are ignored by Git. The sample data is intentionally tiny and demonstrates mechanics, not model quality.

## Current reference documentation

- [NeMo Qwen3 recipes](https://docs.nvidia.com/nemo-framework/user-guide/25.07/llms/qwen3.html)
- [NeMo fine-tuning data](https://docs.nvidia.com/nemo-framework/user-guide/25.04/data/finetune_data.html)
- [SGLang documentation](https://docs.sglang.ai/)
- [TensorRT-LLM quick start](https://nvidia.github.io/TensorRT-LLM/quick-start-guide.html)
- [NVIDIA NIM for LLM installation](https://docs.nvidia.com/nim/large-language-models/latest/get-started/installation.html)

Always check the current support matrix and release notes before copying deployment commands to a GPU host; these projects evolve quickly.
