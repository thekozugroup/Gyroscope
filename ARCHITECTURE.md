# Gyroscope Architecture

Gyroscope is a utility platform that converts a corpus of Body-of-Knowledge (BoK)
documents into the two artifacts needed to align a base LLM to a dedicated role:

1. A high-quality SFT (supervised fine-tuning) dataset.
2. A set of reward functions ready for GRPO post-training.

Gyroscope does **not** train models. It produces drop-in artifacts that can be
fed to any training framework (TRL, Axolotl, Unsloth, custom).

---

## End-to-end Pipeline

```
┌─────────────────┐   ┌────────────────┐   ┌──────────────────┐
│  Ingestion      │ → │  Curation      │ → │  Golden Document │
│  (PDF/web/...)  │   │  (distillation)│   │  (markdown)      │
└─────────────────┘   └────────────────┘   └────────┬─────────┘
                                                    │
                ┌───────────────────────────────────┼───────────────────────────────────┐
                ▼                                   ▼                                   ▼
        ┌───────────────┐                 ┌──────────────────┐                 ┌─────────────────┐
        │ SFT Swarm     │                 │ Reward Designer  │                 │ Evaluation Set  │
        │ (trajectories)│                 │ (python rewards) │                 │ (held-out)      │
        └───────┬───────┘                 └────────┬─────────┘                 └────────┬────────┘
                ▼                                  ▼                                    ▼
        ┌───────────────┐                 ┌──────────────────┐                 ┌─────────────────┐
        │ SFT JSONL     │                 │ rewards.py +     │                 │ eval.jsonl      │
        │ (ShareGPT/    │                 │ reward_spec.yaml │                 │                 │
        │  ChatML)      │                 │                  │                 │                 │
        └───────────────┘                 └──────────────────┘                 └─────────────────┘
```

---

## Package layout

```
gyroscope/
├── core/                 # data models, config, logging, llm client wrapper
│   ├── models.py         # pydantic schemas (Document, Chunk, Trajectory, RewardSpec)
│   ├── config.py         # pipeline config
│   ├── llm.py            # async Anthropic client w/ prompt caching, retries
│   └── io.py             # JSONL, YAML, markdown helpers
├── ingestion/            # phase 1
│   ├── base.py           # Loader interface
│   ├── pdf.py
│   ├── web.py
│   ├── markdown.py
│   ├── html.py
│   ├── docx.py
│   └── pipeline.py
├── curation/             # phase 2 — BoK → golden document
│   ├── chunker.py        # semantic chunking
│   ├── dedup.py          # near-dup removal (MinHash)
│   ├── extractor.py      # principle / fact / procedure extraction
│   ├── synthesizer.py    # merges extracts into golden doc
│   └── pipeline.py
├── sft/                  # phase 3 — golden doc → SFT dataset
│   ├── personas.py       # role personas seeded from golden doc
│   ├── scenarios.py      # scenario generator
│   ├── trajectory.py     # short-lived agent: produces one trajectory
│   ├── swarm.py          # parallel execution + diversity sampling
│   ├── formats.py        # ShareGPT, ChatML, Alpaca writers
│   └── pipeline.py
├── rewards/              # phase 4 — golden doc → reward functions
│   ├── spec.py           # RewardSpec model + DSL
│   ├── designer.py       # extracts measurable criteria from golden doc
│   ├── codegen.py        # emits reward .py modules
│   ├── library.py        # built-in reusable reward primitives
│   └── pipeline.py
├── eval/                 # held-out eval set generator
│   └── pipeline.py
├── cli.py                # `gyroscope` CLI entry point (typer)
└── __init__.py
```

---

## Phase contracts

### Phase 1 — Ingestion
**Input:** filesystem paths, URLs, globs.
**Output:** `List[Document]` where each Document has:
- `source: str`
- `kind: Literal["pdf","web","md","html","docx","txt"]`
- `text: str` (cleaned)
- `metadata: dict` (title, authors, headings, page map, scrape date)

### Phase 2 — Curation
**Input:** `List[Document]`.
**Output:** `GoldenDocument` (markdown) with stable sections:
- `# Identity` (who the agent is)
- `# Mission`
- `# Principles` (numbered, atomic, testable)
- `# Procedures` (numbered, step-by-step playbooks)
- `# Knowledge` (facts, citations to source chunks)
- `# Vocabulary` (domain glossary)
- `# Anti-patterns` (what NOT to do)

Each atomic item carries a stable `id` so SFT/reward phases can cite it.

### Phase 3 — SFT Swarm
**Input:** `GoldenDocument`, swarm config (n_samples, diversity targets).
Output: `dataset.jsonl` (ShareGPT default; ChatML/Alpaca on flag).
Each row is a tagged `Trajectory`:
- `system` (derived from Identity + selected Principles/Procedures)
- `messages: [user, (tool?, assistant)…, final_assistant]`
- `tags: {procedure_ids, principle_ids, persona, difficulty}`

Swarm operates as short-lived agents:
1. **Planner** picks scenario (procedure id + persona + difficulty).
2. **User-sim** plays the user.
3. **Assistant** plays the role using the golden doc as context.
4. **Critic** scores the trajectory against principles; failed rows are dropped or repaired.
5. **De-duplicator** runs at end to prune semantic duplicates.

### Phase 4 — Reward design
**Input:** `GoldenDocument`.
**Output:**
- `rewards.py` — importable module exposing `REWARDS: list[Callable]`.
- `reward_spec.yaml` — declarative specification of each reward
  (name, kind, weight, principle_ids it enforces, scorer config).

Reward kinds emitted (composable):
- `format` — regex/JSON-schema/section presence.
- `lexical` — required/forbidden vocabulary.
- `principle` — LLM-judge against a specific principle id.
- `procedure` — checks that response follows a procedure's steps.
- `safety` — anti-pattern detector.
- `citation` — answer cites a knowledge id that supports the claim.

All reward functions follow the TRL GRPO signature:
`def reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]`.

### Phase 5 — Eval set
A small held-out set generated with the same swarm but seeded from
procedures/scenarios that are excluded from training to prevent leakage.

---

## Quality loop

After each phase the **Critic** agents grade output 0–100 along axes:
- Coverage (does it represent the BoK?)
- Faithfulness (no hallucinations vs. source)
- Diversity (no degenerate repetition)
- Trainability (format correctness, token budget)
- Reward soundness (rewards are gameable-resistant, calibrated)

The pipeline iterates the failing phase until every axis ≥ 95 (target 100).

---

## Tech choices

- Python 3.11+, `uv`/`pip` installable.
- `pydantic` v2 for data contracts.
- `anthropic` SDK with prompt caching for the golden doc.
- `typer` CLI, `rich` for progress.
- `pypdf` + `pdfplumber` for PDFs, `trafilatura` for web, `markdown-it-py` for md, `python-docx` for DOCX.
- `datasketch` (MinHash) for dedup.
- `pytest` + `hypothesis` for tests.
- No training deps; outputs are framework-agnostic.
```
