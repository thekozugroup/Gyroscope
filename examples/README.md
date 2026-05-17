# Example: tiny RICS Quantity Surveyor BoK

This directory contains a small, synthetic Body of Knowledge for the RICS
Quantity Surveyor role, intended as a smoke-test fixture for Gyroscope. The
content is invented for testing purposes — it is **not** an authoritative
reference.

Files:

- `bok_qs/identity.md` — role definition, principles, procedures, knowledge,
  anti-patterns, vocabulary.
- `bok_qs/contracts_supplement.md` — supplementary notes on JCT and NEC
  contract families and two worked examples.

## End-to-end smoke run

```bash
export ANTHROPIC_API_KEY=...
gyroscope run \
  --input examples/bok_qs/ \
  --output runs/qs-smoke \
  --n-trajectories 30 \
  --reward-budget 8
```

Expected output:

```
runs/qs-smoke/
├── documents.jsonl
├── golden.md
├── golden.json
├── sft.jsonl
├── eval.jsonl
├── rewards/
│   ├── _lib.py
│   ├── rewards.py
│   └── reward_spec.yaml
├── history.json
├── report.json
├── report.html
└── report.md
```

The run will iterate up to five times if the quality report scores any axis
below the configured threshold (default 95). Each iteration re-runs only the
phases whose axes failed.
