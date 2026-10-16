# Phase 4 — Aspect + Category Intelligence

Category validation macro-F1:
- Text only: 0.2539
- Aspect + text: 0.3879

Final held-out dev/test category macro-F1: 0.3732.

The category benchmark uses the gold aspect as an input, so it is a second-stage benchmark, not end-to-end.

Aspect lexical baseline:
- Candidate vocabulary: top 2,000 training aspect phrases
- Annotation-level exact hit rate: 0.6085
- Review-level precision: 0.1989
- Review-level recall: 0.5698
- Review-level F1: 0.2948

This is a transparent lexical baseline; a later stronger phase can replace it with span extraction.
