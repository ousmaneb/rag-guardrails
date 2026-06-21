# Architecture

Two flows are kept separate — an **offline ingestion pipeline** that builds the
index, and an **online query pipeline** that serves answers through the
guardrail layer.

## Request path

```
User ──HTTP──> FastAPI /chat
                  │
                  ▼
          [Input Guardrails]   ← injection / jailbreak / PII detection (LLM01/LLM02)
                  │ (pass)
                  ▼
        Retriever (vector search)  ──> Vector DB (pgvector / memory)
                  │  top-k chunks + scores
                  ▼
        Prompt Builder (context + citations + system policy, instruction isolation)
                  │
                  ▼
              LLM (Claude via provider-agnostic interface)
                  │  draft answer
                  ▼
         [Output Guardrails]   ← PII redaction, grounding (cite-or-refuse), policy (LLM02/LLM05/LLM09)
                  │
                  ▼
        Response  { answer, citations, latency, guardrail_flags }
                  │
                  └──> eval harness / monitoring
```

## Ingestion path

```
data/corpus/*.md ──> load ──> chunk (~600 tok, ~13% overlap) ──> embed (bge-small)
                                                                       │
                                                                       ▼
                                                          upsert ──> Vector DB
```

> Replace this file with an Excalidraw export (`docs/architecture.png`) for the
> README hero image.
