# Dialogue Summarization for Acme Communications

**Bottom Line Up Front:** This project fine-tunes a BERT-encoder/GPT-2-decoder (`bert-base-cased` + `gpt2` via `EncoderDecoderModel`) on all 14,731 SAMSum training dialogues. The latest full run completed all 6 epochs in 97.6 minutes on an NVIDIA RTX 4060 with BF16; its best validation checkpoint was step 5500 (ROUGE-L = 0.2677). On the held-out test set with beam search (`beam=5`), it achieved ROUGE-1/ROUGE-2/ROUGE-L of **0.333/0.111/0.272**. Entity consistency remains the main risk: 70.9% of generated summaries contain unsupported capitalized names, compared with 7.6% of human references. Average inference latency is 205 ms (P95 338 ms), passing the <2,000 ms target.

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
├── .gitignore
├── Data/                        # Local dataset CSVs; NOT committed
├── full_run/                    # Training checkpoints; NOT committed
├── sanity_check/                # Sanity check checkpoint; NOT committed
└── final_model/                 # Best model weights; NOT committed
```

Only the notebook, README, and `.gitignore` are committed. `Data/`, `full_run/`, `sanity_check/`, and `final_model/` are excluded via `.gitignore`. The dataset can be regenerated from the Hugging Face source, and model weights are too large for the repository.

## Data

The project uses the [SAMSum dataset](https://huggingface.co/datasets/knkarthick/samsum), which contains messenger-style conversations paired with human-written summaries.

| Split | Rows |
|---|---:|
| Training | 14,731 |
| Validation | 818 |
| Test | 819 |

Columns: `id`, `dialogue`, `summary`. Task: abstractive dialogue summarisation.

### Data Provenance

The CSV files in `Data/` were downloaded from the [SAMSum dataset on Hugging Face](https://huggingface.co/datasets/knkarthick/samsum). `Data/` is not committed to the repository. To regenerate the files:

```python
from datasets import load_dataset
samsum = load_dataset("knkarthick/samsum")
samsum["train"].to_pandas().to_csv("Data/train.csv", index=False)
samsum["validation"].to_pandas().to_csv("Data/validation.csv", index=False)
samsum["test"].to_pandas().to_csv("Data/test.csv", index=False)
```

## How to Run

1. **Create and activate a virtual environment**

```bash
python3.11 -m venv venv
source venv/bin/activate
```

2. **Install dependencies**

```bash
pip install torch transformers==5.23.0 datasets accelerate evaluate rouge-score nltk pandas numpy matplotlib seaborn tqdm ipywidgets
```

3. **Launch the notebook**

```bash
jupyter notebook DS_Capstone_Notebook.ipynb
```

Run all cells sequentially. The notebook covers:

- Business understanding and data exploration
- Dual-tokeniser input preparation (BERT encoder + GPT-2 decoder)
- Overfit sanity check (64 samples, 30 epochs)
- Full training on 14,731 samples with validation monitoring
- ROUGE evaluation on the held-out test set
- Entity consistency and qualitative analysis
- Inference timing measurement
- Model saving and inference verification

The **Full Training Run cell** takes approximately 98 minutes on an NVIDIA RTX 4060 with BF16 support. CPU execution will be substantially slower.

## Model Reproduction

The trained model weights are not committed to this repository. To reproduce the trained model:

1. Follow the **How to Run** steps above using Python 3.11 and `transformers==5.23.0`.
2. Ensure the regenerated dataset is available in `Data/`.
3. Open `DS_Capstone_Notebook.ipynb` in Jupyter.
4. Run all cells from top to bottom. The **Full Training Run cell** takes approximately 98 minutes on an NVIDIA RTX 4060 with BF16 support.
5. After training completes, the model is saved to `final_model/` by the notebook.

`Data/` and `final_model/` are not committed to the repository. Results may vary slightly on different hardware due to floating-point non-determinism.

Note: the reload verification check in the notebook shows a discrepancy between the saved and in-memory model on 2 of 3 spot checks. The cause is under investigation; the reported test scores come from the in-memory model immediately after training.

## Model Summary

| Component | Configuration |
|---|---|
| Base model | `bert-base-cased` (encoder) + `gpt2` (decoder) |
| Architecture | BERT encoder + GPT-2 decoder with cross-attention (Hugging Face `EncoderDecoderModel`) |
| Training task | Abstractive summarisation |
| Training data | 14,731 SAMSum dialogues |
| Optimiser | AdamW |
| Learning rate | `5e-5` |
| Epochs | 6.00 of 6 (patience 10, never triggered) |
| Batch size | 4 (effective 16 with gradient accumulation ×4) |
| Precision | BF16 (RTX 4060) |
| Best checkpoint | Step 5500 |
| Training duration | 97.6 minutes |
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
| Early stopping on validation loss | Validation monitoring with patience 10 on ROUGE-L | ROUGE-L is more directly tied to the quality targets; patience was never triggered |
| FP16 precision | BF16 where supported | RTX 4060 supports BF16, providing stable mixed-precision training |
| Subset for development | Full training set used | The full dataset was used to give the cross-attention layers more signal |
| Decoding grid search | Fixed beam=5 configuration | Provides a reproducible baseline |

## Evaluation and Target Metrics

### Test-Set Scores (beam search, beam=5)

| Metric | Target | Result | Met? |
|---|---:|---:|:---:|
| ROUGE-1 | ≥ 0.40 | **0.3334** | ✗ |
| ROUGE-2 | ≥ 0.15 | **0.1106** | ✗ |
| ROUGE-L | ≥ 0.35 | **0.2719** | ✗ |

### Validation Scores (best checkpoint, step 5500)

| Metric | Target | Result | Met? |
|---|---:|---:|:---:|
| ROUGE-1 | ≥ 0.40 | 0.3284 | ✗ |
| ROUGE-2 | ≥ 0.15 | 0.1023 | ✗ |
| ROUGE-L | ≥ 0.35 | 0.2677 | ✗ |

### Baseline Comparison

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| Extractive baseline | 0.2265 | 0.0640 | 0.2015 |
| Greedy decoding (beam=1) | 0.3244 | 0.1022 | 0.2624 |
| Beam search (beam=5) | **0.3334** | **0.1106** | **0.2719** |

Beam search improves over greedy decoding and the extractive baseline on all three metrics. The scores remain below the original pitch targets, indicating room for architectural and factuality improvements.

### Training Progression

| Step | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---:|---:|---:|---:|
| 100 | 0.1491 | 0.0185 | 0.1263 |
| 1000 | 0.2086 | 0.0433 | 0.1734 |
| 2500 | 0.2586 | 0.0720 | 0.2158 |
| 4000 | 0.3063 | 0.0900 | 0.2500 |
| **5500 (best)** | **0.3284** | **0.1023** | **0.2677** |

### Inference Performance

| Measurement | Result | Target | Status |
|---|---:|---:|:---:|
| Average latency | 204.9 ms | < 2,000 ms | **PASS** |
| P95 latency | 338.4 ms | < 2,000 ms | **PASS** |

## Entity Consistency

Entity consistency measures whether generated summaries introduce unsupported capitalized names, a useful diagnostic for hallucinated entities.

| Source | Summaries with unsupported capitalized names |
|---|---:|
| Model-generated summaries | 70.9% |
| Human reference summaries | 7.6% |
| Gap | **+63.3 percentage points** |

An entity filter that replaced the 70.9% flagged outputs with an extractive fallback produced a hybrid ROUGE-1 of **0.2731**, versus **0.3334** for beam search (reported change: **−0.060 absolute points (−18% relative)**). This shows the trade-off between factuality-oriented filtering and lexical quality. The filter is diagnostic only, not a usable mitigation: replacing most model outputs with extractive text substantially changes system behaviour and still does not establish factual correctness.

## Length Buckets

Test-set ROUGE-L varies with dialogue length:

| Bucket | ROUGE-L | n |
|---|---:|---:|
| Short (≤50 words) | 0.3291 | 269 |
| Medium (51–100) | 0.2599 | 264 |
| Long (101–200) | 0.2413 | 208 |
| Very long (>200) | 0.1927 | 78 |

Performance declines as dialogues get longer, reflecting the greater difficulty of summarizing longer conversations. Only approximately 5% of dialogues exceed the 356-token input cap, so the cap affects a small minority of the test set, although those examples are among the more difficult cases.

## Business Impact

A successful summarisation feature could:

- Reduce the time users spend reading long conversation threads
- Improve discoverability of decisions, topics, and action items
- Make group-chat participation more accessible to users returning after an absence
- Provide a measurable starting point for an Acme Communications summarisation product

ROUGE should be treated as an offline development signal rather than a complete business KPI.

## Limitations

- ROUGE measures word overlap, not meaning, usefulness, or factual faithfulness.
- The BERT encoder truncates input dialogues beyond 356 tokens (the 95th percentile of token lengths); about 5% of dialogues are cut during encoding.
- The model frequently generates plausible but unsupported details, especially capitalized names (70.9% versus 7.6% for references).
- Separate BERT and GPT-2 tokenisers require distinct padding and special-token logic.
- The generation output is capped at 53 tokens (max_target_length); about 5% of human reference summaries exceed this length.
- The cross-attention between independently pre-trained BERT and GPT-2 components may be limited compared with native encoder-decoder architectures such as BART or T5.
- ROUGE improves steadily but plateaus below the pitch targets.
- Computational constraints limited the number of architecture, learning-rate, and decoding experiments.
- Offline metrics do not replace human evaluation of faithfulness, coverage, conciseness, or usefulness.

## Future Work

1. **Run a BERT2BERT or BART-small architectural comparison** to test whether a native encoder-decoder architecture improves over the BERT–GPT-2 cross-attention design.
2. Conduct human evaluation for faithfulness, coverage, conciseness, and usefulness.
3. Develop a validated factuality and hallucination detection or correction method; retain the entity filter as a diagnostic rather than a production mitigation.
4. Support longer input sequences and long-conversation handling.
5. Tune generation parameters for Acme's preferred summary style.
6. Add multi-language summarisation support.
7. Build an API inference service with monitoring and rate controls.
8. Evaluate parameter-efficient fine-tuning with LoRA or QLoRA.
9. Fine-tune on Acme-specific, privacy-reviewed conversation data.

## Privacy and Responsible Use

Conversation data can contain personal, confidential, or sensitive information. Any production version should use approved data-governance practices, access controls, retention limits, and privacy review. Generated summaries should be presented as assistance, not as an authoritative record, until factuality has been validated.
