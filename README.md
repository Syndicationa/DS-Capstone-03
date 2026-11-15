# Dialogue Summarization for Acme Communications

**Bottom Line Up Front:** This project fine-tunes a BERT-encoder/GPT-2-decoder (`bert-base-cased` + `gpt2` via `EncoderDecoderModel`) on the full SAMSum dataset (14,731 dialogues) for abstractive dialogue summarisation. The best validation checkpoint (step 3900) achieved ROUGE-1 = 0.317, ROUGE-2 = 0.094, and ROUGE-L = 0.258. On the held-out test set the model scored ROUGE-1 = 0.320, ROUGE-2 = 0.107, and ROUGE-L = 0.261 — below the pitch targets of 0.40, 0.15, and 0.35 but demonstrating steady learning over 4.56 epochs (72.5 minutes of training before early stopping). The model beats both the extractive baseline (ROUGE-1: 0.226, ROUGE-2: 0.064, ROUGE-L: 0.201) and greedy decoding.

## Business Problem

Acme Communications users participate in group chats where important decisions and updates are discussed. Users cannot always be online — when they return, catching up requires reading through every message. In busy chats, this creates information overload: key details get lost, users spend too much time reading, and engagement drops. An automated summarization feature provides a concise first-pass summary so users can understand the main points without reading every message.

### Stakeholders

- **Acme users**: Quickly catch up on conversations they missed.
- **Product managers**: Validate the feature using measurable summarisation metrics.
- **Engineering team**: Build toward a deployment-ready model and inference service.

## Repository Structure

```text
Project 3/
├── DS_Capstone_Notebook.ipynb   # Full project notebook (CRISP-DM)
├── README.md                    # This file
├── Data/
│   ├── train.csv                # Training conversations (14,731)
│   ├── validation.csv           # Validation set (818)
│   └── test.csv                 # Test set (819)
├── full_run/                    # Saved model checkpoints from full training
├── sanity_check/                # Overfit sanity check checkpoint
└── final_model/                 # Best model after full training (saved by notebook)
```

## Data

The project uses the [SAMSum dataset](https://huggingface.co/datasets/knkarthick/samsum), which contains messenger-style conversations paired with human-written summaries.

| Split | Rows |
|---|---|
| Training | 14,731 |
| Validation | 818 |
| Test | 819 |

Columns: `id`, `dialogue`, `summary`. Task: abstractive dialogue summarisation.

## How to Run

1. **Open the repository**
```bash
cd "Project 3"
```

2. **Activate the environment**
```bash
source language_model_env/bin/activate
```

3. **Install dependencies** (if not already installed)
```bash
pip install torch transformers datasets accelerate evaluate rouge-score nltk pandas numpy matplotlib seaborn tqdm
```

4. **Run the notebook**
```bash
jupyter notebook DS_Capstone_Notebook.ipynb
```
Run all cells sequentially. The notebook covers:
- Business understanding and data exploration
- Dual-tokeniser input preparation (BERT encoder + GPT-2 decoder)
- Overfit sanity check (64 samples, 30 epochs)
- Full training on 14,731 samples with early stopping on ROUGE-L
- ROUGE evaluation on the held-out test set
- Qualitative analysis of 20 test examples with failure tagging
- Inference timing measurement
- Model saving and inference verification

Training is configured for an NVIDIA GPU with bf16 support. CPU execution will be substantially slower.

## Model Summary

| Component | Configuration |
|---|---|
| Base model | `bert-base-cased` (encoder) + `gpt2` (decoder) |
| Architecture | BERT encoder + GPT-2 decoder with cross-attention (Hugging Face `EncoderDecoderModel`) |
| Training task | Abstractive summarisation |
| Optimiser | AdamW |
| Learning rate | `5e-5` |
| Epochs | 4.56 of 6 (early stopping on ROUGE-L) |
| Batch size | 4 (effective 16 with gradient accumulation ×4) |
| Precision | BF16 (RTX 4060) |
| Early stopping | Patience 3 on validation ROUGE-L |
| Best checkpoint | Step 3900 |
| Training duration | 72.45 minutes |
| Generation | Beam search (beam=5, length_penalty=1.0, no_repeat_ngram_size=3) |
| Hardware | NVIDIA RTX 4060 with BF16 support |

## Two-Tokenizer Approach

The model uses separate tokenisers for its two pretrained components:
- The BERT tokenizer (`bert-base-cased`) tokenises each dialogue for the encoder.
- The GPT-2 tokenizer (`gpt2`) tokenises each reference summary and generated sequence for the decoder.
- A custom pad token (`<|pad|>`) and BOS token (mapped to EOS) are added to the GPT-2 tokenizer.
- EOS is explicitly appended to each label sequence so the decoder learns when to stop generating.

## Deviations from the Pitch

| Pitch plan | Actual implementation | Reason |
|---|---|---|
| Early stopping on validation loss | Early stopping on ROUGE-L | ROUGE-L is more directly tied to the quality targets |
| FP16 precision | BF16 where supported | RTX 4060 supports BF16, better numerical stability |
| Subset for development | Full training set used | The subset produced severe hallucination |
| Decoding grid search | Fixed beam=5 config | Deferred; fixed config provides a reproducible baseline |

## Evaluation and Target Metrics

### Test-Set Scores (beam search, beam=5)

| Metric | Target | Result | Met? |
|---|---|---|---|
| ROUGE-1 | ≥ 0.40 | 0.320 | ✗ |
| ROUGE-2 | ≥ 0.15 | 0.107 | ✗ |
| ROUGE-L | ≥ 0.35 | 0.261 | ✗ |

### Validation Scores (best checkpoint, step 3900)

| Metric | Target | Result | Met? |
|---|---|---|---|
| ROUGE-1 | ≥ 0.40 | 0.317 | ✗ |
| ROUGE-2 | ≥ 0.15 | 0.094 | ✗ |
| ROUGE-L | ≥ 0.35 | 0.258 | ✗ |

### Baseline Comparison

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| Extractive baseline | 0.226 | 0.064 | 0.201 |
| Greedy decoding (beam=1) | 0.314 | 0.097 | 0.259 |
| Beam search (beam=5) | **0.320** | **0.107** | **0.261** |

The model shows steady improvement throughout training — validation ROUGE-L rose from 0.126 at step 100 to 0.258 at step 3900 — and beats both baselines on all metrics. The gap to the pitch targets suggests the BERT–GPT-2 cross-attention architecture may be limited for this task, though the early stopping criteria and learning-rate schedule have not been exhaustively tested.

### Training Progression

| Step | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| 100 | 0.149 | 0.018 | 0.126 |
| 1000 | 0.230 | 0.051 | 0.190 |
| 2500 | 0.260 | 0.072 | 0.216 |
| **3900 (best)** | **0.317** | **0.094** | **0.258** |

## Business Impact

A successful summarisation feature could:
- Reduce the time users spend reading long conversation threads
- Improve discoverability of decisions, topics, and action items
- Make group-chat participation more accessible to users returning after an absence
- Provide a measurable starting point for an Acme Communications summarisation product

ROUGE should be treated as an offline development signal rather than a complete business KPI.

## Limitations

- ROUGE measures word overlap, not meaning, usefulness, or factual faithfulness.
- The BERT encoder truncates dialogues beyond 512 tokens, potentially losing context.
- The model may generate plausible but unsupported details (hallucination).
- Separate BERT and GPT-2 tokenisers require distinct padding and special-token logic.
- The cross-attention between independently pre-trained BERT and GPT-2 components may be limited for this task compared to native encoder-decoder architectures like BART or T5; this has not yet been exhaustively tested with alternative learning rates or patience settings.
- Computational constraints limited the number of learning-rate experiments.

## Future Work

1. **Run the fallback BERT2BERT or BART-small comparison**: a native encoder-decoder architecture would show whether the gap to the pitch targets comes from the BERT–GPT-2 cross-attention design or from the training setup (early-stopping patience, learning rate).
2. Conduct human evaluation for faithfulness, coverage, conciseness, and usefulness.
3. Add factuality and hallucination detection.
4. Support longer input sequences and long-conversation handling.
5. Add multi-language summarisation support.
6. Build an API inference service with monitoring and rate controls.
7. Tune generation parameters for Acme's preferred summary style.
8. Evaluate parameter-efficient fine-tuning with LoRA or QLoRA.
9. Fine-tune on Acme-specific, privacy-reviewed conversation data.

## Privacy and Responsible Use

Conversation data can contain personal, confidential, or sensitive information. Any production version should use approved data-governance practices, access controls, retention limits, and privacy review. Generated summaries should be presented as assistance, not as an authoritative record, until factuality has been validated.
