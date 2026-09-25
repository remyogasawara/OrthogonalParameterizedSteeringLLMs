# Generation protocols

These are the original generation instructions, with protocol names and internal
references standardized to Generation Protocols 1-4. Dataset row
numbers below are one-based and refer to `../data/warmth_families.jsonl`.

| Rows | Families | Protocol | Recorded generator |
|---|---:|---|---|
| 1-100 | 100 | [Generation Protocol 1](generation_protocol_1.md) | ChatGPT Sol 5.6 Pro |
| 101-200 | 100 | [Generation Protocol 2](generation_protocol_2.md) | Claude Opus 5 Max |
| 201-300 | 100 | [Generation Protocol 3](generation_protocol_3.md) | Specific model not recorded |
| 301-1000 | 700 | [Generation Protocol 4](generation_protocol_4.md) | Codex GPT-5.6 Sol Ultra |

`../data/provenance.csv` records family-level provenance using `protocol_1`
through `protocol_4`. These numbers identify generation protocols, not warmth labels.

The first three protocols take the corresponding user messages as input. Generation
Protocol 4 takes each complete family record without its `responses` field: prompts
and factual specifications were retained while responses were regenerated. The
700 Protocol 4 inputs were processed in their recorded order in 35 batches of 20 families.
`generation_batch_instructions.txt` preserves the original generic batch wrapper;
its protocol filename reference has been updated. The wrapper does
not identify a generation platform. Protocol texts document the intended generation
constraints, not a claim that every generated response satisfies them.
