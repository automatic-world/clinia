<p align="center">
  <img src="./assets/app-icon.png" alt="clinia" width="160" />
</p>

A GPU-based LLM fine-tuning and serving pipeline. This document describes the project's **LLM model composition** from the perspectives of base model, training, data, and inference/serving.

---

## 1. Model Composition Overview

| Category | Details |
|------|------|
| Base model | **Llama 3.1 8B Instruct** |
| Extended pipeline | **Qwen3-32B** multi-phase (SFT → DPO → GRPO → Export) |
| Fine-tuning method | LoRA (PEFT) |
| Deployed models | LoRA adapter / base+LoRA merged model |
| Distribution channel | HuggingFace Hub — `the-platforms/MediCPX` |
| Domain | CPX medical (clinical diagnosis) instruction-following |

**Two pipeline tracks**

- **Llama single pipeline** — `FineTuningPipeline`: preprocessing → tokenization → LoRA training → save
- **Qwen3 multi-phase pipeline** — `MultiPhaseFineTuningPipeline`: SFT → DPO → GRPO → Export

---

## 2. Base Model & Weights

At the end of training, artifacts are produced under a per-run timestamped directory.

```
{output_dir}/{yyyymmddhhmm}/final_model/
├── adapter/          # LoRA adapter (tens of MB) — for shared-base serving
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── merged/           # base + LoRA merged model (tens of GB) — for standalone deployment
│   ├── config.json
│   └── model.safetensors
└── model_info.json   # base model, training config, and path metadata
```

| Format | Size | Loading | Use |
|------|------|------|------|
| `adapter/` | Small | `PeftModel.from_pretrained(base, adapter)` | Experiments & versioning |
| `merged/` | Large | `AutoModelForCausalLM.from_pretrained(merged)` | Dependency-free standalone serving |

**HuggingFace Hub upload** — `runs/upload_to_hf.py`

```bash
export HF_TOKEN=hf_xxx
python runs/upload_to_hf.py --folder-path /ai_models/merged --repo-id the-platforms/MediCPX
```

- Training-only state (`optimizer.pt`, `scheduler.pt`, `rng_state*`) is excluded by default
- Details: [`llama/huggingface_upload.md`](./llama/huggingface_upload.md)

---

## 3. Training Configuration

**4-step flow**: preprocessing → tokenization → LoRA training → save

**Key hyperparameters**

| Item | Value |
|------|-----|
| LoRA rank | `r = 8` (α = r × 2) |
| Learning rate | `2e-4` |
| max_length | `2048` |
| Precision | bf16 (auto) |
| Other | Response-region loss masking + sequence packing |

**Key code paths**

| Category | Path |
|------|------|
| Pipeline package | `app/finetuning/` |
| Llama single pipeline | `app/finetuning/pipeline.py` (`FineTuningPipeline`) |
| Qwen3 multi-phase | `app/finetuning/pipeline.py` (`MultiPhaseFineTuningPipeline`) |
| LoRA trainer | `app/finetuning/training/llama_trainer.py` |
| GRPO trainer (TRL) | `app/finetuning/training/grpo_trainer.py` |
| Reward functions | `app/finetuning/training/rewards/` (`AgentReward`, `SIDomainReward`) |
| LLaMA-Factory integration | `app/finetuning/llamafactory/` |

**Run**

```bash
# Llama LoRA fine-tuning
python runs/run_finetuning.py --model-path /ai_models --data-path /data/output/1105 --epochs 3
# Multi-GPU (DDP)
torchrun --nproc_per_node=2 runs/run_finetuning.py --model-path /ai_models --data-path /data/output/1105
# Qwen3 multi-phase
python runs/run_qwen3_finetuning.py
```

Details: [`llama/process.md`](./llama/process.md) · [`qwen3_finetuning_process.md`](./qwen3_finetuning_process.md)

---

## 4. Dataset Configuration

PDF source documents → instruction-response JSON generation (4-Stage).

```
PDFLoader(PyMuPDF4LLM) → MarkdownConverter → InstructionParser(GPT-4) → BatchedJSONWriter
```

**Output schema** (identical to training input)

```json
[{ "instruction": "...", "context": "(optional)", "response": "..." }]
```

**Data composition by training method**

| Method | Format | Fields |
|------|------|------|
| SFT | ShareGPT `messages` | system/user/assistant — one gold answer |
| DPO | `sharegpt_dpo` (ranking) | `conversations` + `chosen` + `rejected` |
| GRPO | Prompt + reward function | No gold answers or pairs needed; scored via `RewardFunction.compute()` |

**Code paths**

| Category | Path |
|------|------|
| Preprocessing package | `app/data_handling/data_pre_processing/` |
| Qwen3 data pipeline | `app/data_handling/` (chunker, qa_generation, quality, ragas_eval, trajectory) |
| Entry points | `runs/run_cpx_processing.py`, `runs/run_qwen3_data_pipeline.py` |

```bash
INPUT_PATH=/data/input/cpx.pdf OUTPUT_PATH=/data/output/1105 python runs/run_cpx_processing.py
```

Details: [`llama/data_preprocessing.md`](./llama/data_preprocessing.md)

---

## 5. Inference & Serving

| Category | Path | Description |
|------|------|------|
| Lightweight inference server | `server.py` | Single-model startup; `/generate`, `/stream`, `/health` |
| Full pipeline API | `main.py` | Integrated model load/training/inference; OpenAI-compatible chat |
| vLLM serving | `app/serving/vllm_config.py` | For Qwen3 Agent; Hermes tool parser |
| Model load helper | `runs/load_llama_3_1_8b.py` | Llama 3.1 8B load verification |

```bash
# Lightweight inference server
MODEL_DIR=/path/to/model USE_4BIT=1 uvicorn server:app --host 0.0.0.0 --port 8080
# vLLM (with tool calling)
./scripts/serve_qwen3_vllm.sh
```

**OpenAI-compatible REST API** — two entry points

`main.py` — Full Pipeline API (`:8000`)

| Category | Endpoints |
|------|-----------|
| Health | `GET /health` |
| Model | `POST /api/v1/models/load`, `GET /api/v1/models/{id}/verify` |
| Preprocess | `POST /api/v1/preprocessing/run` |
| Tokenize | `POST /api/v1/tokenization/tokenize` |
| Training | `POST /api/v1/training/start`, `GET /api/v1/training/{job_id}/status` |
| Multi-Phase | `POST /api/v1/training/multi-phase/start` |
| Data Pipeline | `POST /api/v1/data-pipeline/generate-qa` |
| Inference | `POST /api/v1/inference/load`, `GET /api/v1/inference/models` |
| Chat (OpenAI-compatible) | `POST /api/v1/chat/completions`, `POST /v1/chat/completions` |

`server.py` — Inference API (`:8080`): `POST /generate`, `POST /stream` (SSE), `GET /health`

```bash
./scripts/api_server.sh start    # start main.py
```

Request examples: [`API_SAMPLE.md`](./API_SAMPLE.md)

---

## 6. Evaluation

| Category | Path |
|------|------|
| Agent evaluation | `runs/eval_agent.py` |
| Domain evaluation | `runs/eval_domain.py` |
| Model evaluation | `runs/eval_model.py` |
| NLG surface metrics (BLEU/ROUGE/METEOR/BERTScore) | `runs/eval_nlg.py` |
| RAGAS evaluation | `runs/eval_ragas.py` |
| Model comparison | `runs/eval_compare.py` |

Details: [`evaluation.md`](./evaluation.md) · [`llama/evaluation_usage.md`](./llama/evaluation_usage.md) · [`llama/evaluation_runbook.md`](./llama/evaluation_runbook.md)

---

## 7. Demo

| Category | Path | Description |
|------|------|------|
| Medical chat loop demo | `scripts/medical_chat_loop.sh` | Calls the chat API in parallel with N random medical questions (repeats periodically) |
| Question set | `scripts/medical_questions.txt` | Question pool for demo input |
| System prompt | `scripts/medical_system_prompt.txt` | System prompt for the demo |
| Frontend integration guide | [`frontend_request_cancel_guide.md`](./frontend_request_cancel_guide.md) | Request/cancel integration |

```bash
# After starting the inference server
API_URL=http://localhost:8000 MODEL=MediCPX ./scripts/medical_chat_loop.sh
```

---

## Appendix — Deliverables Inventory

| # | Item | Key Location | Form |
|---|------|-----------|------|
| 1 | Model weights | `final_model/`, HF Hub `the-platforms/MediCPX` | LoRA adapter / merged model |
| 2 | Training code | `app/finetuning/`, `runs/run_finetuning.py` | Python package + entry point |
| 3 | Inference code | `server.py`, `main.py`, `app/serving/` | FastAPI + vLLM |
| 4 | Dataset | `app/data_handling/`, `runs/run_cpx_processing.py` | PDF → instruction JSON |
| 5 | Demo | `scripts/medical_chat_loop.sh` | CLI chat demo |
| 6 | API | `main.py` (Full) / `server.py` (Inference) | OpenAI-compatible REST |
| 7 | Technical docs | `docs/` | Markdown + papers/patents |

**Technical documentation**

| Category | Document |
|------|------|
| Project overview | [`README.md`](./README.md), [`QUICKSTART.md`](./QUICKSTART.md) |
| End-to-end summary | [`llama/end_to_end_summary.md`](./llama/end_to_end_summary.md) |
| Data preprocessing | [`llama/data_preprocessing.md`](./llama/data_preprocessing.md) |
| Training walkthrough | [`llama/process.md`](./llama/process.md) |
| Qwen3 multi-phase | [`qwen3_finetuning_process.md`](./qwen3_finetuning_process.md) |
| Evaluation | [`evaluation.md`](./evaluation.md), [`llama/evaluation_usage.md`](./llama/evaluation_usage.md), [`llama/evaluation_runbook.md`](./llama/evaluation_runbook.md) |
| Improvement history | [`llama/improvements.md`](./llama/improvements.md) |
| Research & papers | [`llama/research.md`](./llama/research.md), [`llama/papers.md`](./llama/papers.md) |
| HF upload | [`llama/huggingface_upload.md`](./llama/huggingface_upload.md) |
| Deployment | [`deployments/README.md`](./deployments/README.md) (Docker, K3s) |
| Patent/paper drafts | `docs/paper/` (CPX medical-domain IMRaD, patent application drafts & figures) |