# Generation Pipelines and Chunking

A pipeline defines:
- chunking: how the document is split
- prompt: which template to guide card production
- note_type: which card schema/rendering to use
- params: free-form tuning knobs passed to templates/postprocessors
- reviewer: optional LLM-based scoring/cleanup; `min_score` gate
- dedup: de-duplication policy and scope

Chunking modes (`generate.*.chunking.mode`)
- pages: one chunk per page (simple; coarse)
- sections: heading-aware splits (chapters/sections)
- paragraphs: paragraph-level chunks (fine-grained)
- smart: adaptive merges within token limits for coherent chunks
- figures: isolate figure captions and neighboring text
- highlights: use annotations/highlights if available
- entire: prefer one chunk for entire document; auto-split when limits exceeded; trims terminal sections like References/Appendix

Prompts (examples in `prompts/`)
- cloze_generation: cloze deletions with explanations
- high_concept_cloze_generation: broader concept clozes on large context
- key_points: short Q/A fact extraction
- figure_explanations: explain figures/diagrams grounded in captions + nearby text
- reviewer_special: reviewer model instructs to score/edit cards and return numeric scores

Note types (examples in `notes/`)
- Basic: Front/Back
- Cloze: Cloze deletions and Extra
- CustomCloze: custom schema with provenance fields

Reviewer and min_score
- When `reviewer.enabled: true`, a second LLM call reviews cards.
- Cards with score < `min_score` are dropped; edited content may be kept.
- Defaults: disabled; `min_score: 0.6`.

Params
- Free-form dict accessible by templates. Common keys:
  - target_card_count, difficulty, forbid_lists, enable_citations
- Extend as needed; templates can read via Jinja variables.

Dedup
- scope: per_pipeline | global
- policy: or | and

Migration
- Legacy `ingestion.*` and `strategies.*` map into `generate.*` pipelines at load time with a deprecation warning.
