# Phase 15 — Severity & Priority Intelligence — FIXED

This is the corrected Phase 15 script.

The fix prevents a pandas `KeyError: 'rank'` caused by a rank-column collision
when Phase 14 evidence is merged with the Phase 15 ranking.

Run from the project root:

```powershell
..\.venv\Scripts\python.exe -u scripts/phase15_severity_priority_intelligence.py
```
