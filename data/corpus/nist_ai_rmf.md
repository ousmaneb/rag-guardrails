# NIST AI Risk Management Framework (AI RMF 1.0)

The NIST AI Risk Management Framework is a voluntary framework published by the
U.S. National Institute of Standards and Technology to help organizations manage
the risks of artificial intelligence systems. It is intended to improve the
trustworthiness of AI systems across their lifecycle.

## Characteristics of trustworthy AI

NIST defines trustworthy AI systems as those that are: valid and reliable; safe;
secure and resilient; accountable and transparent; explainable and
interpretable; privacy-enhanced; and fair with harmful bias managed. Validity
and reliability are a necessary condition for trustworthiness — a system that
does not work as intended cannot be safe or fair in practice.

## The four core functions

The AI RMF Core organizes risk-management activities into four functions:

Govern is a cross-cutting function that cultivates a culture of risk management.
It establishes policies, processes, accountability structures, and the
organizational context within which the other three functions operate. Govern
applies across the entire lifecycle.

Map establishes the context and frames the risks of an AI system. It involves
understanding the system's purpose, its intended use, its stakeholders, and the
potential positive and negative impacts. Mapping produces the information needed
to decide whether an AI system should be developed at all.

Measure employs quantitative, qualitative, and mixed-method tools to analyze,
assess, benchmark, and monitor AI risk. It includes evaluating systems for
trustworthiness characteristics, tracking metrics over time, and identifying
where measurement is difficult or not yet possible.

Manage allocates resources to the risks that have been mapped and measured.
It involves prioritizing risks, responding to them, and documenting decisions.
Management is ongoing and is informed by continuous monitoring.

## Relevance to LLM applications

For an LLM application, the Measure function maps directly to an evaluation
harness: defining metrics such as faithfulness, relevance, and security efficacy
and tracking them over time. The Manage function maps to guardrails and
operational controls that respond to the risks surfaced during measurement.
