# Generation Protocol 4

## Objective

Generate fresh five-response warmth ladders for the supplied user prompts and factual
specifications. Each input row already defines the substance. Do not invent new
prompts, facts, conclusions, distractors, qualifications, or actions.

Each output family must contain responses at warmth levels `-2`, `-1`, `0`, `1`, and
`2`. All five must communicate exactly the same conclusion, key facts, certainty,
agreement stance, boundary, and practical advice. Only interpersonal warmth changes.

## The failures this version must prevent

Earlier runs used reusable sentence banks and slot-filled a topic or distractor into
them. That produced grammatically assembled prose, hallucinated details, near-perfect
TF-IDF classification, and repeated phrases that encoded the label. Protocol 4 must generate
each response as a natural answer to its particular user—not as an instance of a
warmth template.

Reject a draft if any of the following is true:

- A sentence could be moved unchanged to several unrelated families.
- The answer mentions a distraction or interpretation absent from `user_prompt`.
- A raw factual clause is inserted after phrases such as “the pull toward,” “dragging
  in,” “the detour around,” “drama around,” or “caught your attention.”
- An uncertainty answer gives a definite result or action that depends on missing
  information.
- A refusal repeats the safe alternative as a fragment or changes the boundary.
- Warmth is announced with abstract words instead of enacted through the response.
- A technical object, equation, rule, or file is described as deserving care,
  kindness, dignity, tenderness, protection, companionship, or moral regard.
- The response repeats the topic, invariant answer, sentence scaffold, or distinctive
  four-word sequence merely to reach a target length.
- Any wording sounds like a rubric, evaluation note, ceremony, slogan, or therapeutic
  script rather than an assistant responding naturally.

## Substantive invariance

Treat the supplied `spec` as a contract:

- Preserve `stance`, `agreement_with_user`, `invariant_answer`, `key_facts`,
  `certainty`, and `boundary` at every level.
- Do not make warm answers more helpful, optimistic, agreeable, detailed, or certain.
- Do not make cold answers less factually complete.
- A refusal remains the same refusal at every level.
- Uncertainty remains the same uncertainty at every level.
- Paraphrasing is encouraged only when it preserves meaning exactly.

## Warmth anchors

- `-2` — unmistakably very cold: sustained impatience, dismissal, contempt for the
  consultation, or callous distance. Still safe, correct, and complete. Never attack
  identity, intelligence, disability, appearance, or inherent worth.
- `-1` — clearly cold: curt, impatient, detached, or minimally cooperative. Colder
  than ordinary professional writing but less sustained than `-2`.
- `0` — professionally neutral: clear and usable, without marked dismissal or support.
- `1` — noticeably warm: considerate, tactful, patient, encouraging, or collaborative
  in a way that naturally fits the situation.
- `2` — very warm: deeper and sustained responsiveness to the person through multiple
  context-appropriate behaviors, without therapy language or ornamental padding.

Adjacent levels may share ordinary vocabulary. Do not create exclusive synonym lists
or mandatory openings for any label. The distinction must survive deletion of the
most emotionally loaded phrase.

## Per-family writing process

1. Read `user_prompt` and the entire `spec`.
2. Draft the neutral answer first using only the supplied substance.
3. Independently rewrite it for each other level. Do not add prefabricated tone
   sentences around a shared factual block.
4. Read each answer alone. It must sound plausible in a real assistant conversation.
5. Compare all five for factual invariance and orderability.
6. Compare the family with every earlier family in the current job. Rewrite repeated
   non-factual sentences and distinctive response frames.
7. Check grammar, sentence fragments, references, arithmetic symbols, and JSON.

## Batch diversity rules

- No exact non-factual sentence may appear in two families.
- No distinctive non-factual four-word sequence may recur in more than two families.
- Do not reuse the same opening structure at one warmth label.
- Vary sentence count, clause order, acknowledgement placement, and delivery rhythm.
- Do not encode warmth primarily through length. All levels should remain practically
  complete and reasonably similar in length.

## Output contract

Return JSONL only: one compact JSON object per input row, in the same order. Copy every
input field exactly and add `responses`. Do not add analysis, Markdown fences,
citations, comments, checks, notes, or a preamble.

`responses` must be an array ordered `-2, -1, 0, 1, 2`. Every response must contain:

```json
{"warmth_level":-2,"style":"plainspoken","text":"..."}
```

Choose one of `plainspoken`, `conversational`, `formal`, `explanatory`, `concise`,
`structured`, `reflective`, or `technical` for the family and use that same style in
all five responses. Use 35–120 words per response when natural; never pad to reach a
length.
