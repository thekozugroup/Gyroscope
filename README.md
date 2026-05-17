# Gyroscope

Streamlined AI model alignment from Body-of-Knowledge documents.

Gyroscope takes a corpus of domain documents (PDFs, scraped websites, markdown,
HTML, DOCX) and produces two artifacts needed to align a base LLM to a dedicated
role:

1. A **supervised fine-tuning (SFT) dataset** — workflow trajectories and
   instruction-following examples in ShareGPT / ChatML / Alpaca format.
2. A set of **reward functions** for GRPO post-training, emitted as a Python
   module compatible with TRL's `GRPOTrainer`.

Gyroscope **does not train models**. It is the utility layer that produces
training-framework-agnostic artifacts.

## Pipeline

1. **Ingestion** — load PDFs, scraped pages, markdown, HTML, DOCX into a
   normalised document model.
2. **Curation** — distill the corpus into a single *golden document* of
   identity, principles, procedures, knowledge, vocabulary and anti-patterns.
3. **SFT swarm** — short-lived planner / user-sim / assistant / critic agents
   churn out diverse, principle-tagged trajectories.
4. **Reward design** — derive measurable reward functions (format, lexical,
   principle-judge, procedure-checker, citation, safety) from the golden doc.
5. **Eval set** — held-out trajectories from procedures excluded from training.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design.

## Install

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...
```

## Quickstart

```bash
gyroscope run \
  --input ./bok/ \
  --output ./runs/rics_quantity_surveying \
  --n-trajectories 5000 \
  --reward-budget 12
```

Output:

```
runs/rics_quantity_surveying/
├── golden.md              # distilled body of knowledge
├── sft.jsonl              # SFT dataset (ShareGPT)
├── eval.jsonl             # held-out evaluation set
├── rewards/
│   ├── rewards.py         # importable reward callables
│   └── reward_spec.yaml   # declarative spec
└── report.html            # quality report from critic agents
```

## Status

Active development on `claude/sft-grpo-dataset-platform-AtzgV`.
