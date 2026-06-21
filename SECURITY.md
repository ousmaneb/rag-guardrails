# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's **Report a vulnerability**
button (Security tab → Advisories), or by opening a minimal issue that does not
disclose exploit details. Do not post working exploits publicly.

## Repository protections

This repository is configured for defense-in-depth:

- **Read-only to the public.** It is a public repository: anyone may view, clone,
  and fork, but only the owner has write access. External contributions are only
  possible via pull requests, which require owner approval before merge.
- **Branch protection (`main`).** Force-pushes and branch deletion are blocked by
  an active ruleset, so history cannot be rewritten or the branch removed.
- **Secret scanning + push protection.** Commits containing detected secrets are
  blocked before they reach the repository.
- **Dependabot.** Vulnerability alerts and automated security-update pull requests
  are enabled; dependencies are watched weekly (`.github/dependabot.yml`).
- **CodeQL.** Static security analysis (`security-extended` queries) runs on every
  push/PR and weekly (`.github/workflows/codeql.yml`).
- **Secret hygiene.** Secrets live only in `.env` (git-ignored); only the blank
  `.env.example` template is committed. CI runs a `gitleaks` scan on every push.

## Application-level security

The application itself is hardened against the OWASP LLM Top 10 — input/output
guardrails, instruction isolation, PII redaction, a grounding gate, and rate
limiting. See [`docs/security.md`](docs/security.md).
