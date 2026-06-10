<p align="center">
  <img src="./assets/app-icon.png" alt="clinia" width="160" />
</p>

<h1 align="center">clinia</h1>

<p align="center">Clinical + AI — 임상 진단 지원을 위한 Opensource Library</p>

GPU 기반 LLM 파인튜닝·서빙 파이프라인. 본 문서는 프로젝트의 **LLM 모델 구성**을
베이스 모델 · 학습 · 데이터 · 추론/서빙 관점에서 정리한다.

---

## 1. 모델 구성 개요 (Model Composition)

| 구분 | 내용 |
|------|------|
| 베이스 모델 | **Llama 3.1 8B Instruct** |
| 확장 파이프라인 | **Qwen3-32B** 멀티페이즈 (SFT → DPO → GRPO → Export) |
| 파인튜닝 방식 | LoRA (PEFT) |
| 배포 모델 | LoRA 어댑터 / base+LoRA 병합 모델 |
| 배포 채널 | HuggingFace Hub — `the-platforms/MediCPX` |
| 도메인 | CPX 의료(임상 진단) instruction-following |

**두 갈래 파이프라인**

- **Llama 단일 파이프라인** — `FineTuningPipeline` : 전처리 → 토크나이즈 → LoRA 학습 → 저장
- **Qwen3 멀티페이즈 파이프라인** — `MultiPhaseFineTuningPipeline` : SFT → DPO → GRPO → Export

---

## 2. 베이스 모델 & 가중치 (Base Model & Weights)

학습 종료 시 실행별 타임스탬프 디렉터리 아래에 산출된다.

```
{output_dir}/{yyyymmddhhmm}/final_model/
├── adapter/          # LoRA 어댑터 (수십 MB) — 베이스 공유 서빙용
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── merged/           # base + LoRA 병합 모델 (수십 GB) — 단독 배포용
│   ├── config.json
│   └── model.safetensors
└── model_info.json   # 베이스 모델·학습 설정·경로 메타데이터
```

| 형식 | 크기 | 로딩 | 용도 |
|------|------|------|------|
| `adapter/` | 작음 | `PeftModel.from_pretrained(base, adapter)` | 실험·버전관리 |
| `merged/` | 큼 | `AutoModelForCausalLM.from_pretrained(merged)` | 의존성 없는 단독 서빙 |

**HuggingFace Hub 업로드** — `runs/upload_to_hf.py`

```bash
export HF_TOKEN=hf_xxx
python runs/upload_to_hf.py --folder-path /ai_models/merged --repo-id the-platforms/MediCPX
```

- 학습 전용 상태(`optimizer.pt`, `scheduler.pt`, `rng_state*`)는 기본 제외
- 상세: [`llama/huggingface_upload.md`](./llama/huggingface_upload.md)

---

## 3. 학습 구성 (Training Configuration)

**4단계 흐름**: 전처리 → 토크나이즈 → LoRA 학습 → 저장

**주요 하이퍼파라미터**

| 항목 | 값 |
|------|-----|
| LoRA rank | `r = 8` (α = r × 2) |
| learning rate | `2e-4` |
| max_length | `2048` |
| precision | bf16 (auto) |
| 기타 | 응답영역 손실 마스킹 + sequence packing |

**핵심 코드 경로**

| 구분 | 경로 |
|------|------|
| 파이프라인 패키지 | `app/finetuning/` |
| Llama 단일 파이프라인 | `app/finetuning/pipeline.py` (`FineTuningPipeline`) |
| Qwen3 멀티페이즈 | `app/finetuning/pipeline.py` (`MultiPhaseFineTuningPipeline`) |
| LoRA 학습기 | `app/finetuning/training/llama_trainer.py` |
| GRPO 학습기 (TRL) | `app/finetuning/training/grpo_trainer.py` |
| 보상함수 | `app/finetuning/training/rewards/` (`AgentReward`, `SIDomainReward`) |
| LLaMA-Factory 연동 | `app/finetuning/llamafactory/` |

**실행**

```bash
# Llama LoRA 파인튜닝
python runs/run_finetuning.py --model-path /ai_models --data-path /data/output/1105 --epochs 3

# 다중 GPU (DDP)
torchrun --nproc_per_node=2 runs/run_finetuning.py --model-path /ai_models --data-path /data/output/1105

# Qwen3 멀티페이즈
python runs/run_qwen3_finetuning.py
```

상세: [`llama/process.md`](./llama/process.md) · [`qwen3_finetuning_process.md`](./qwen3_finetuning_process.md)

---

## 4. 데이터 구성 (Dataset Configuration)

PDF 원문 → instruction-response JSON 생성 (4-Stage).

```
PDFLoader(PyMuPDF4LLM) → MarkdownConverter → InstructionParser(GPT-4) → BatchedJSONWriter
```

**출력 스키마** (학습 입력과 동일)

```json
[{ "instruction": "...", "context": "(선택)", "response": "..." }]
```

**학습 방식별 데이터 구성**

| 방식 | 포맷 | 필드 |
|------|------|------|
| SFT | ShareGPT `messages` | system/user/assistant 정답 1개 |
| DPO | `sharegpt_dpo` (ranking) | `conversations` + `chosen` + `rejected` |
| GRPO | 프롬프트 + 보상함수 | 정답·쌍 불필요, `RewardFunction.compute()` 점수화 |

**코드 경로**

| 구분 | 경로 |
|------|------|
| 전처리 패키지 | `app/data_handling/data_pre_processing/` |
| Qwen3 데이터 파이프라인 | `app/data_handling/` (chunker, qa_generation, quality, ragas_eval, trajectory) |
| 진입점 | `runs/run_cpx_processing.py`, `runs/run_qwen3_data_pipeline.py` |

```bash
INPUT_PATH=/data/input/cpx.pdf OUTPUT_PATH=/data/output/1105 python runs/run_cpx_processing.py
```

상세: [`llama/data_preprocessing.md`](./llama/data_preprocessing.md)

---

## 5. 추론 & 서빙 구성 (Inference & Serving)

| 구분 | 경로 | 설명 |
|------|------|------|
| 경량 추론 서버 | `server.py` | 단일 모델 기동, `/generate`·`/stream`·`/health` |
| 풀 파이프라인 API | `main.py` | 모델 로드/학습/추론 통합, OpenAI 호환 chat |
| vLLM 서빙 | `app/serving/vllm_config.py` | Qwen3 Agent용, Hermes tool parser |
| 모델 로드 헬퍼 | `runs/load_llama_3_1_8b.py` | Llama 3.1 8B 로드 검증 |

```bash
# 경량 추론 서버
MODEL_DIR=/path/to/model USE_4BIT=1 uvicorn server:app --host 0.0.0.0 --port 8080

# vLLM (툴콜 지원)
./scripts/serve_qwen3_vllm.sh
```

**OpenAI 호환 REST API** — 두 진입점

`main.py` — Full Pipeline API (`:8000`)

| 분류 | 엔드포인트 |
|------|-----------|
| Health | `GET /health` |
| Model | `POST /api/v1/models/load`, `GET /api/v1/models/{id}/verify` |
| Preprocess | `POST /api/v1/preprocessing/run` |
| Tokenize | `POST /api/v1/tokenization/tokenize` |
| Training | `POST /api/v1/training/start`, `GET /api/v1/training/{job_id}/status` |
| Multi-Phase | `POST /api/v1/training/multi-phase/start` |
| Data Pipeline | `POST /api/v1/data-pipeline/generate-qa` |
| Inference | `POST /api/v1/inference/load`, `GET /api/v1/inference/models` |
| Chat (OpenAI 호환) | `POST /api/v1/chat/completions`, `POST /v1/chat/completions` |

`server.py` — Inference API (`:8080`): `POST /generate`, `POST /stream`(SSE), `GET /health`

```bash
./scripts/api_server.sh start    # main.py 기동
```

호출 예시: [`API_SAMPLE.md`](./API_SAMPLE.md)

---

## 6. 평가 구성 (Evaluation)

| 구분 | 경로 |
|------|------|
| Agent 평가 | `runs/eval_agent.py` |
| 도메인 평가 | `runs/eval_domain.py` |
| 모델 평가 | `runs/eval_model.py` |
| 자연어 표면지표(BLEU/ROUGE/METEOR/BERTScore) | `runs/eval_nlg.py` |
| RAGAS 평가 | `runs/eval_ragas.py` |
| 모델 비교 | `runs/eval_compare.py` |

상세: [`evaluation.md`](./evaluation.md) · [`llama/evaluation_usage.md`](./llama/evaluation_usage.md) · [`llama/evaluation_runbook.md`](./llama/evaluation_runbook.md)

---

## 7. 데모 (Demo)

| 구분 | 경로 | 설명 |
|------|------|------|
| 의료 챗 루프 데모 | `scripts/medical_chat_loop.sh` | 랜덤 의료 질문 N개를 chat API로 병렬 호출(주기 반복) |
| 질문 셋 | `scripts/medical_questions.txt` | 데모 입력 질문 풀 |
| 시스템 프롬프트 | `scripts/medical_system_prompt.txt` | 데모용 system 프롬프트 |
| 프론트엔드 연동 가이드 | [`frontend_request_cancel_guide.md`](./frontend_request_cancel_guide.md) | 요청/취소 연동 |

```bash
# 추론 서버 기동 후
API_URL=http://localhost:8000 MODEL=MediCPX ./scripts/medical_chat_loop.sh
```

---

## 부록 — 산출물 인벤토리 (Deliverables)

| # | 항목 | 핵심 위치 | 형태 |
|---|------|-----------|------|
| 1 | 모델 가중치 | `final_model/`, HF Hub `the-platforms/MediCPX` | LoRA 어댑터 / 병합 모델 |
| 2 | 학습 코드 | `app/finetuning/`, `runs/run_finetuning.py` | Python 패키지 + 진입점 |
| 3 | 추론 코드 | `server.py`, `main.py`, `app/serving/` | FastAPI + vLLM |
| 4 | 데이터셋 | `app/data_handling/`, `runs/run_cpx_processing.py` | PDF→instruction JSON |
| 5 | 데모 | `scripts/medical_chat_loop.sh` | CLI 챗 데모 |
| 6 | API | `main.py` (Full) / `server.py` (Inference) | OpenAI 호환 REST |
| 7 | 기술문서 | `docs/` | Markdown + 논문/특허 |

**기술문서 목록**

| 분류 | 문서 |
|------|------|
| 프로젝트 개요 | [`README.md`](./README.md), [`QUICKSTART.md`](./QUICKSTART.md) |
| End-to-End 요약 | [`llama/end_to_end_summary.md`](./llama/end_to_end_summary.md) |
| 데이터 전처리 | [`llama/data_preprocessing.md`](./llama/data_preprocessing.md) |
| 학습 walkthrough | [`llama/process.md`](./llama/process.md) |
| Qwen3 멀티페이즈 | [`qwen3_finetuning_process.md`](./qwen3_finetuning_process.md) |
| 평가 | [`evaluation.md`](./evaluation.md), [`llama/evaluation_usage.md`](./llama/evaluation_usage.md), [`llama/evaluation_runbook.md`](./llama/evaluation_runbook.md) |
| 개선 적용 이력 | [`llama/improvements.md`](./llama/improvements.md) |
| 연구·논문 | [`llama/research.md`](./llama/research.md), [`llama/papers.md`](./llama/papers.md) |
| HF 업로드 | [`llama/huggingface_upload.md`](./llama/huggingface_upload.md) |
| 배포 | [`deployments/README.md`](./deployments/README.md) (Docker, K3s) |
| 특허/논문 초안 | `docs/paper/` (CPX 의료 도메인 IMRaD, 특허출원 초안·도면) |
