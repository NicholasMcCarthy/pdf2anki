Summary

Remove legacy preprocess and consolidate functionality under generate.
Implement generate modes:
--plan: real first-chunk prompt render (no LLM call).
--sample: first-chunk LLM call; display parsed cards; no writes.
default: full generation orchestrated via documents.yaml pipelines.
Add chunking.mode: entire with auto-splitting and trimming of References/Appendix.
Add optional reviewer step with min_score gating via a second LLM call using prompts/reviewer_special.j2.
Unify config into generate with pipelines; documents.yaml supports multiple pipelines per document.
Fix enum/path YAML serialization (scalars, not single-item lists).
pdf2anki init now also copies notes/ and creates samples/ with a generator script.
Add CI to run tests on PRs (Python 3.11).

Key changes

CLI: remove preprocess; all docs updated to use generate.
Implement generate --plan, --sample, and full run.
TextChunker: new entire mode with limits and trimming of References/Bibliography/Appendix.
Reviewer: optional LLM reviewer filters cards by min_score; edited content retained.
Config: new generate.default_pipeline and generate.pipelines; legacy ingestion/strategies translated with deprecation warning.
YAML: fix writing of enums and paths as scalars.
init: copies notes/ and creates samples/; runs sample generator if reportlab is present.
CI: GitHub Actions runs pytest on PRs (Python 3.11).

Acceptance

generate --plan shows a real prompt preview with first-chunk content (truncated).
generate --sample calls LLM on first chunk and displays parsed cards only.
Default generate persists CSV/media/manifest.
entire chunking mode works and trims terminal references sections.
Reviewer step filters cards under min_score when enabled.
Example YAMLs validate successfully.
CI tests run on PRs and pass on Python 3.11.

Migration

preprocess removed; use generate. Legacy ingestion/strategies supported via translation with a deprecation warning for one release.
