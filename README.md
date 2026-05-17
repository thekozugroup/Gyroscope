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
  --reward-budget 12 \
  --threshold 95 \
  --max-iterations 5
```

`run` is driven by the `AutonomousRunner`: it grades the outputs against
five deterministic axes (coverage, faithfulness, diversity, trainability,
reward soundness) and re-runs any phase whose axis fell below `--threshold`
up to `--max-iterations` times. The per-phase subcommands (`ingest`,
`curate`, `sft`, `rewards`, `report`) remain available for resumability.

Output:

```
runs/rics_quantity_surveying/
├── documents.jsonl        # normalised source documents
├── golden.md              # distilled body of knowledge
├── golden.json            # same, machine-readable
├── sft.jsonl              # SFT dataset (ShareGPT by default)
├── eval.jsonl             # held-out evaluation set (procedure-disjoint)
├── rewards/
│   ├── rewards.py         # importable reward callables
│   ├── _lib.py            # self-contained runtime primitives
│   ├── __init__.py        # exposes REWARDS, WEIGHTS
│   └── reward_spec.yaml   # declarative spec
├── history.json           # per-iteration axis scores
├── report.md              # human-readable quality report
├── report.html            # self-contained HTML report
└── report.json            # machine-readable report
```

## Status

Active development on `claude/sft-grpo-dataset-platform-AtzgV`.
