# Open Event Intel

A personal research project exploring structured event extraction, temporal reasoning, and agentic workflows built on top of a continuously updated corpus of publicly available industry news. The project follows principles from the OpenAI Temporal Knowledge Graph framework while extending it with tooling for provenance, contradiction detection, and temporal claim management.

Live version: [Open Event Intel](https://vsevolodnedora.github.io/open-event-intel/)
---

## Motivation

Industry domains with fast-moving regulatory, technical, and infrastructure developments generate a large volume of unstructured information. Understanding what changed, when it changed, and how claims evolve over time is difficult without automation.

This project investigates how modern LLMs, structured extraction, and agent-based pipelines can be applied to:

- normalize heterogeneous news and regulatory updates,
- extract entities, events, and time-scoped claims,
- maintain a temporally aware knowledge graph,
- detect updates, contradictions, and invalidations,
- provide grounded explanations based on structured information.

The goal is to demonstrate capabilities in LLM tooling, agent-oriented architectures, and temporal knowledge systems that generalize beyond the current domain focus.

---

## Current Stage

The project is under active development and includes:

- automated daily ingestion of selected public news sources,
- standardized cleaning and normalization pipelines,
- structured extraction of entities and events,
- early components for temporal claim tracking and provenance-aware storage.

The system is evolving toward an event-intelligence stack built on temporal representations, modular agents, and continuous evaluation.

---

## Planned Direction

The next development stages explore an extensible framework tentatively referred to as **Grid Intel Lab** — a research-oriented platform for change detection and temporal reasoning over public information. Planned components include:

- a “what changed” feed powered by temporal diffing of claims,
- contradiction and supersession detection across sources,
- an event explainer agent grounded in the knowledge graph,
- evaluation suites for extraction accuracy and temporal consistency,
- a minimal web interface for querying and browsing time-scoped events,
- clean APIs and small OSS-style libraries for temporal claim models and extraction utilities.

These features aim to provide a demonstration of modern LLM operations, agentic orchestration, and structured reasoning workflows in a compact, well-engineered system suitable as a public portfolio project.

---

## Notes on local use

Due to copyright and compliance, full text of publications cannot be provided on a public repository.  
Databases with raw publications as well as with components of the temporally-aware knowledge graph will be 
built in `/database/` automatically when running `run_scrape.py`.  

However, in order to run `run_tkg.py` prompts and definitions are required which can be provided by the project author upon reasonable request.

---

## License

This project is for personal, educational, and non-commercial use. All rights to original source documents remain with their respective publishers.
