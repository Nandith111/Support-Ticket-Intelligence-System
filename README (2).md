# Support-Ticket Intelligence System

An end-to-end ML/DL system for **classifying, prioritizing, and drafting replies to customer-support tickets**, with a strong focus on production-oriented inference optimization.

The project combines a fast **DistilBERT intent classifier** with a lightweight **Qwen2.5-1.5B-Instruct + LoRA reply generator**, and evaluates optimized inference using **ONNX Runtime, TensorRT, vLLM, and 4-bit quantization**.

> **Note:** The project report describes the implemented methodology and benchmark pipeline. Numerical accuracy, latency, throughput, and memory results are generated when the notebook is executed and saved as CSV artifacts; they are not hard-coded in this README.

---

## 🚀 Project Highlights

- **27-class customer-support intent classification** using DistilBERT.
- **Priority/urgency estimation** derived from the predicted intent.
- **Automated reply drafting** using Qwen2.5-1.5B-Instruct fine-tuned with LoRA.
- Classical baselines using **TF-IDF + Logistic Regression** and **TF-IDF + LightGBM**.
- Reply-generation baselines using **BM25 retrieval** and **zero-shot Qwen2.5-1.5B**.
- Classifier optimization with **ONNX, ONNX Runtime, and TensorRT**.
- LLM serving optimization with **vLLM, PagedAttention, continuous batching, prefix caching, AWQ 4-bit, and NF4 quantization**.
- **FastAPI** inference service with `/classify` and `/ticket` endpoints.
- **Locust** load testing, **Docker** packaging, and **MLflow** experiment tracking.
- Robustness evaluation using noisy inputs, slice analysis, calibration, bootstrap testing, PSI, and KS drift checks.

---

## 🧩 Problem Statement

Customer-support teams receive large volumes of free-text tickets. Each ticket needs to be:

1. Understood — **What is the customer asking for?**
2. Prioritized — **Does it require urgent attention?**
3. Answered — **What response should the support agent provide?**

This system addresses these requirements through a two-model pipeline:

```text
Customer Ticket
      │
      ▼
Text Cleaning / PII Masking
      │
      ▼
DistilBERT Intent Classifier
      │
      ├──────────────► Intent / Category / Priority
      │
      ▼
Prompt Builder
(ticket + intent + category)
      │
      ▼
Qwen2.5-1.5B + LoRA
      │
      ▼
Drafted Reply
      │
      ▼
Human Agent Review
```

The classifier output is used both as structured routing information and as context for reply generation.

---

## 🎯 Objectives

- Build a complete ML lifecycle from data preprocessing to deployment.
- Compare transformer-based classification against classical ML baselines.
- Compare fine-tuned LLM generation against retrieval and zero-shot baselines.
- Validate models using clean and noisy test sets.
- Measure the impact of inference optimizations on:
  - Latency
  - Throughput
  - GPU/CPU memory
  - Model size
  - Prediction/generation quality
- Identify speed-quality trade-offs using Pareto analysis.
- Expose the final pipeline through a production-style FastAPI service.
- Load-test the service and track experiments with MLflow.

---

## 📊 Dataset

The project uses the public **Bitext customer-support LLM chatbot training dataset**.

**Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`

| Property | Details |
|---|---|
| Approx. rows | 26.9K before cleaning |
| Intents | 27 |
| Categories | 11 |
| Main text field | `instruction` |
| Classification target | `intent` |
| Generation target | `response` |
| Additional fields | `category`, `flags` |

Example intents include:

- `cancel_order`
- `payment_issue`
- `track_order`
- `recover_password`
- `get_refund`

The dataset is synthetic/template-based, so the project includes noisy-test evaluation and slice analysis to make robustness comparisons more meaningful.

Source-derived dataset details: fileciteturn0file0L80-L99

---

## 🧹 Data Preprocessing

All preprocessing and prompt-building logic is centralized in `ticket_utils.py`. The same module is used during training and serving to reduce train/serve skew.

### Processing pipeline

```text
Raw Ticket
   │
   ├── Unicode normalization
   ├── PII masking
   ├── Placeholder normalization
   ├── Signature removal
   ├── Whitespace cleanup
   ├── Language / encoding filtering
   ├── Deduplication
   │
   ▼
Stratified 80 / 10 / 10 Split
   │
   ├── Train
   ├── Validation
   └── Test
```

### PII protection

The preprocessing pipeline masks:

- Email addresses
- URLs
- Phone numbers

This prevents unnecessary exposure and reduces the risk of memorizing personal information.

---

## 🔎 Feature Engineering

Feature engineering is used primarily for the classical ML baselines.

### Feature groups

| Feature Group | Features |
|---|---|
| Length | Character count, word count |
| Style | Capital ratio, punctuation ratio, exclamation count, question count, digit ratio |
| Sentiment | VADER compound score |
| Structure | Placeholder indicator |
| Keywords | Refund, cancel, urgent, problem, account, tracking, invoice, human agent |
| Text | TF-IDF word 1–2 grams |

The transformer model learns contextual representations directly from tokenized text.

---

# 🤖 Models

## 1. Intent Classifier

### Baseline 1 — TF-IDF + Logistic Regression

- Word 1–2 gram TF-IDF features
- Engineered features
- Balanced class weights
- 5-fold stratified cross-validation

### Baseline 2 — TF-IDF + LightGBM

- Top 5,000 TF-IDF features
- Engineered features
- Balanced class weights
- Validation-based early stopping

### Main Model — DistilBERT

`distilbert-base-uncased`

Configuration:

- Sequence length: 64
- Batch size: 32
- Optimizer: AdamW
- Learning rate: `5e-5`
- Epochs: 3
- FP16 mixed-precision training
- Linear warm-up and decay
- Best checkpoint selected using validation macro-F1

DistilBERT is used because it provides contextual understanding while remaining lightweight enough for aggressive inference optimization.

---

## 2. Reply Generator

### Baseline 1 — BM25 Retrieval

Retrieves the response from the most similar training ticket.

### Baseline 2 — Zero-Shot Qwen2.5-1.5B-Instruct

Uses the base model without task-specific fine-tuning.

### Main Model — Qwen2.5-1.5B-Instruct + LoRA

LoRA configuration:

- Rank: 16
- Applied to attention and MLP projections
- Base weights: FP16
- Adapter: FP32 during training
- Gradient checkpointing
- Loss calculated on reply tokens
- LoRA adapter merged into base model for serving

The inference prompt contains:

```text
System instructions
        +
Predicted category
        +
Predicted intent
        +
Customer ticket
        ↓
Generated support reply
```

The shared system prompt also enables vLLM prefix caching.

---

# ⚡ Inference Optimization

The project deliberately uses different optimization technologies for the classifier and LLM.

## ONNX

The trained DistilBERT model is exported to ONNX with dynamic batch and sequence axes.

ONNX provides:

- Framework-independent model representation
- Easier deployment
- Reduced serving dependency on PyTorch
- A common model representation for ONNX Runtime and TensorRT

The ONNX graph is validated against PyTorch using logits and top-1 prediction agreement.

---

## ONNX Runtime

ONNX Runtime is used for CPU and GPU inference.

### Tested variants

```text
ORT FP32
ORT FP16
ORT INT8 Dynamic
ORT INT8 Static / QDQ
```

FP16 is used by the FastAPI classifier service on GPU.

INT8 CPU variants use dynamic or calibrated static quantization.

---

## TensorRT

TensorRT compiles the ONNX model into an optimized NVIDIA GPU engine.

The project evaluates:

- TensorRT FP16
- TensorRT INT8

The TensorRT optimization profile supports:

- Batch size: 1–64
- Sequence length: 64

INT8 uses calibration data, while FP16 remains enabled so sensitive layers can fall back where appropriate.

TensorRT engines are hardware/version-specific, so their quality and performance are benchmarked against the PyTorch FP32 reference.

---

# 🧠 vLLM Optimization

The Qwen2.5-1.5B model is served using vLLM.

### Key optimizations

#### Continuous Batching

New requests can join an already-running batch instead of waiting for a static batch to finish.

**Benefit:** Better throughput under concurrent traffic.

#### PagedAttention

KV cache memory is divided into manageable blocks.

**Benefit:** Better GPU memory utilization and less fragmentation.

#### Prefix Caching

The shared system prompt is computed once and reused across requests.

**Benefit:** Reduces repeated prefill computation.

---

## 🗜️ LLM Quantization

The project evaluates:

- FP16
- AWQ 4-bit
- bitsandbytes NF4

### AWQ

Activation-aware Weight Quantization compresses model weights to 4-bit precision while attempting to preserve important model behavior.

### NF4

bitsandbytes NF4 provides 4-bit quantization during model loading without a separate AWQ calibration process.

The project compares these variants based on speed, memory, model size, and reply quality.

---

# 📏 Evaluation

## Classifier Metrics

- Accuracy
- Macro-F1
- Per-class precision
- Per-class recall
- Confusion matrix
- Expected Calibration Error (ECE)
- Reliability diagram
- Urgent-ticket precision/recall

Macro-F1 is particularly important because all 27 intent classes should contribute to evaluation.

---

## Statistical Validation

A paired bootstrap procedure is used:

```text
Test Set
   │
   ▼
1,000 bootstrap resamples
   │
   ▼
Macro-F1 difference
   │
   ▼
95% confidence interval
```

This is applied to comparisons between DistilBERT and the classification baselines.

---

## Robustness Testing

A noisy test set is created by applying:

- Character-level typos
- Random lower-casing
- Punctuation removal

Additional analysis includes:

- Bitext flag-based slice analysis
- Misclassification inspection
- Population Stability Index (PSI)
- Kolmogorov-Smirnov (KS) testing

---

## Reply Generation Metrics

The generator is evaluated using:

- ROUGE-1
- ROUGE-L
- BERTScore F1
- Baseline comparison against BM25 and zero-shot Qwen
- Optional LLM-as-judge evaluation
- Manual review of 50 responses

The optional LLM judge scores:

- Helpfulness
- Relevance
- Tone
- Faithfulness / absence of invented order details

---

# ⏱️ Benchmarking

The benchmark measures request-level performance including GPU data transfer.

### Latency

Measured at:

```text
Batch 1
Batch 8
Batch 32
```

Using:

- p50
- p95
- p99

### Throughput

Classifier:

- Requests/second

LLM:

- Requests/second
- Tokens/second

### Memory

Measured using:

- GPU memory delta after model loading
- Batch-32 warm-up
- Weight size on disk

### Quality

Classifier:

- Clean-set macro-F1
- Noisy-set macro-F1
- Prediction agreement

LLM:

- ROUGE-L
- BERTScore
- Quality delta versus vLLM FP16

---

# 📈 Expected Optimization Trade-offs

The notebook measures the actual values; the following are general tendencies rather than claimed benchmark results:

| Optimization | Expected Benefit |
|---|---|
| FP16 | Lower GPU latency and memory |
| ONNX Runtime | Optimized graph execution and portable serving |
| TensorRT FP16 | Highly optimized NVIDIA GPU inference |
| TensorRT INT8 | Further latency/memory reduction with possible quality trade-off |
| vLLM | Higher LLM serving throughput under concurrency |
| Prefix caching | Reduced repeated prompt-processing cost |
| AWQ 4-bit | Lower model size and memory traffic |
| NF4 | Lower memory footprint through 4-bit loading |

The project explicitly measures quality alongside performance rather than assuming that every optimization is lossless.

---

# 🌐 API

The deployment layer exposes the models through FastAPI.

## `POST /classify`

Returns structured classification information:

```json
{
  "intent": "track_order",
  "category": "ORDER",
  "confidence": 0.95,
  "urgent_score": 0.08,
  "priority": "normal"
}
```

> Example response structure only; values above are illustrative.

## `POST /ticket`

Runs the complete pipeline:

```text
Ticket
  ↓
Classification
  ↓
Priority
  ↓
Prompt construction
  ↓
Qwen generation
  ↓
Draft reply
```

The response includes the predicted intent, category, priority, drafted reply, and per-stage timings.

---

# 🧪 Load Testing

Locust is used to simulate concurrent users.

The load test covers:

- Classification-only requests
- Full pipeline requests
- Request count
- Failure count
- Latency percentiles

---

# 📦 Deployment Architecture

```text
                    ┌──────────────────┐
                    │  Customer Ticket │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     FastAPI      │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │ DistilBERT Classifier   │
                 │ ONNX / ORT / TensorRT   │
                 └───────────┬─────────────┘
                             │
                  Intent + Category + Priority
                             │
                             ▼
                 ┌─────────────────────────┐
                 │     Prompt Builder      │
                 └───────────┬─────────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │ Qwen2.5-1.5B + LoRA     │
                 │          vLLM            │
                 └───────────┬─────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Draft Reply    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Human Review    │
                    └──────────────────┘
```

Supporting infrastructure:

```text
Docker
  │
  ├── FastAPI service
  │
  └── vLLM server

MLflow ──► Experiment tracking
Locust ──► Load testing
```

---

# 🛠️ Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data | Hugging Face Datasets, pandas, NumPy |
| Classical ML | scikit-learn, LightGBM |
| NLP | Transformers, VADER |
| Deep Learning | PyTorch |
| Fine-tuning | PEFT / LoRA |
| Classification | DistilBERT |
| Generation | Qwen2.5-1.5B-Instruct |
| Retrieval | BM25 |
| Model Format | ONNX |
| Inference | ONNX Runtime |
| GPU Optimization | TensorRT |
| LLM Serving | vLLM |
| Quantization | AWQ, bitsandbytes NF4 |
| Evaluation | scikit-learn, ROUGE, BERTScore |
| API | FastAPI, Uvicorn, HTTPX |
| Load Testing | Locust |
| Tracking | MLflow |
| Containerization | Docker |
| Visualization | Matplotlib, Seaborn |
| Hardware Target | NVIDIA T4 16 GB |

---

# 💻 Hardware Target

The project is designed around a **Google Colab free-tier NVIDIA T4 GPU with 16 GB VRAM**.

The architecture intentionally avoids requiring a large multi-GPU environment.

The LLM and optimization environments are isolated because vLLM and AutoAWQ may require different PyTorch versions from the ONNX Runtime/TensorRT stack.

---

# ▶️ Running the Project

## 1. Open Google Colab

Open:

```text
support_ticket_intelligence.ipynb
```

Select:

```text
Runtime → Change runtime type → T4 GPU
```

## 2. Run the notebook

Execute the cells in order.

The complete workflow includes:

```text
Dataset loading
      ↓
Preprocessing
      ↓
Feature engineering
      ↓
Classical baselines
      ↓
DistilBERT training
      ↓
Qwen + LoRA training
      ↓
Evaluation
      ↓
ONNX export
      ↓
ONNX Runtime benchmarks
      ↓
TensorRT benchmarks
      ↓
vLLM benchmarks
      ↓
Quantization benchmarks
      ↓
FastAPI deployment
      ↓
Locust load testing
      ↓
MLflow tracking
```

The full run is expected to take approximately **2–3 hours**, depending on the enabled experiments and Colab environment.

---

# ⚙️ Configuration

The notebook includes configuration flags such as:

```python
N_LLM_TRAIN
RUN_TRT_INT8
RUN_AWQ
RUN_DEPLOY_DEMO
```

These can be disabled to shorten development/debugging runs.

---

# 📁 Generated Artifacts

Benchmark and evaluation outputs are written under:

```text
/content/work/artifacts/
```

Important outputs include:

```text
classifier_benchmark.csv
llm_benchmark.csv
pareto.png
manual_review.csv
```

MLflow runs are stored under:

```text
/content/work/mlruns/
```

Because Google Colab runtimes can reset, important artifacts should be copied to persistent storage such as Google Drive.

---

# 📌 Limitations

### Synthetic dataset

The Bitext dataset is template-based, which can result in near-paraphrases across splits and unusually high clean-test performance.

**Improvement:** Evaluate on real production tickets and use grouped/template-aware splitting.

### Proxy urgency

Urgency is derived from a predefined set of intents rather than real customer-priority labels.

**Improvement:** Collect real priority/urgency annotations and train a dedicated urgency model.

### No timestamps

The dataset has no timestamps, so genuine temporal validation is unavailable.

**Improvement:** Use real timestamped tickets and perform temporal/rolling validation.

### Limited generation evaluation

The generation evaluation uses 96 test tickets and a single random seed.

**Improvement:** Use a larger evaluation set and multiple random seeds.

### Generic replies

The generated replies are not grounded in real company policies or order databases.

**Improvement:** Add retrieval-augmented generation over a verified support-policy knowledge base and relevant customer/order data.

### Hardware-specific TensorRT engines

TensorRT engines are tied to the GPU and TensorRT environment used to build them.

**Improvement:** Build and validate target-specific engines through CI/CD.

---

# 🔮 Future Work

- Integrate **RAG** with a verified customer-support knowledge base.
- Add real-time order/account information to generated responses.
- Train a dedicated urgency/priority model.
- Introduce temporal validation and live drift monitoring.
- Add **Triton Inference Server** for production inference.
- Explore dynamic batching.
- Evaluate speculative decoding with vLLM.
- Support per-tenant LoRA adapters.
- Evaluate larger models such as Qwen2.5-3B with AWQ.
- Add Evidently-based production monitoring.
- Expand from synthetic support data to real-world ticket datasets.

---

# 📚 Project Structure

A recommended repository structure is:

```text
support-ticket-intelligence/
│
├── support_ticket_intelligence.ipynb
├── ticket_utils.py
│
├── api/
│   ├── main.py
│   └── ...
│
├── models/
│   ├── classifier/
│   └── generator/
│
├── artifacts/
│   ├── classifier_benchmark.csv
│   ├── llm_benchmark.csv
│   ├── pareto.png
│   └── manual_review.csv
│
├── docker/
│   └── Dockerfile
│
├── locust/
│   └── locustfile.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

The exact repository layout may differ from the notebook-generated layout.

---

# 🧠 Key Engineering Takeaway

This project is not only about building accurate NLP models. It demonstrates the complete path from **model development to optimized inference and deployment**.

The central engineering principle is:

```text
Model Quality
      +
Inference Efficiency
      +
Operational Reliability
      =
Production-Ready ML System
```

Every optimization is evaluated against the reference model so that improvements in latency, throughput, or memory are considered together with their effect on prediction or generation quality.

---

## 📄 Project Report

The detailed project methodology, architecture, evaluation strategy, optimization techniques, deployment design, limitations, and future work are documented in the accompanying project report. fileciteturn0file0L2-L16

---

## 👤 Author

**Nandith Sreekumar**

Data Science | Machine Learning | NLP | LLMs | Model Optimization

---

## ⭐ If You Found This Project Useful

Consider starring the repository and exploring the notebook to reproduce the training, optimization, benchmarking, and deployment workflow.
