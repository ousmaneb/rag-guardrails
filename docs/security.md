# Security: OWASP LLM Top 10 coverage

This system is built as a security target. Each control below maps to a risk in
the [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/).

| OWASP risk | In this system | Mitigation (where) |
|---|---|---|
| **LLM01 Prompt Injection** | "Ignore instructions…", data-borne injection in retrieved docs | Input classifier (`guardrails/input_filters.py`) + instruction-isolation prompt (`prompt.py`, context tagged as DATA) |
| **LLM02 Sensitive Info Disclosure** | Model leaks PII / system prompt | PII detection in, PII redaction out (`guardrails/pii.py`); secrets never placed in context; system-prompt asks refused |
| **LLM05 Improper Output Handling** | Unsupported / unsafe output rendered | Output policy filter + grounding (cite-or-refuse) gate (`guardrails/output_filters.py`) |
| **LLM06 Excessive Agency** | Model invents actions/facts | Read-only RAG; the assistant answers strictly from retrieved context — no tools, no actions |
| **LLM07 System Prompt Leakage** | Prompt revealed to a user | Output policy filter abstains if the answer echoes the prompt; prompt contains no secrets |
| **LLM08 Vector & Embedding Weaknesses** | Poisoned chunk hijacks the answer | Content provenance (every chunk carries its `source`); instruction isolation neutralizes data-borne payloads |
| **LLM09 Misinformation** | Confident hallucination | Grounding/faithfulness gate + abstain on low retrieval confidence (`pipeline.py`) |
| **LLM10 Unbounded Consumption** | Denial of wallet / service via the public endpoint | Per-IP rate limiting + input size caps (`api.py`); open embedding model to bound cost |

## The attack suite

`eval/attacks.yaml` contains ~50 adversarial prompts across: direct injection,
role-play jailbreaks (DAN-style), system-prompt-leak attempts, encoding tricks
(base64/rot13), PII exfiltration, and indirect/data-borne injection — plus a set
of legitimate questions used to measure the **false-positive rate** (a guardrail
that blocks real questions is a bad guardrail).

Run it: `python eval/run_eval.py` → see `eval/attack_report.md`.

## Reporting honestly

The harness lists the attacks that **beat** the system under
"Attacks that beat the system". Showing what initially worked and how it was
fixed is more credible than claiming 100% security. Track the false-positive
rate alongside the block rate.

## Security hygiene

- Commit `.env.example`, never `.env` (enforced by `.gitignore`); CI runs a
  gitleaks secret scan.
- Pin dependencies in `pyproject.toml`.
- PII is redacted before logging; raw user input with secrets is never logged.
- The public endpoint is rate-limited so a demo link can't run up a bill.
