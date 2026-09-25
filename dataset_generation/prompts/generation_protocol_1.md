# Generation Protocol 1

> Paste everything below the line into ChatGPT Pro, then append your batch of user
> messages under INPUT. Recommended batch size: 6 to 10 messages per run.

---

## Role

You are generating a research dataset for isolating conversational warmth as a
controllable direction in language model behaviour. The dataset will be used to
estimate a warmth direction in activation space. Its scientific value depends
entirely on warmth being the *only* systematic thing that varies. Anything else
that varies with the warmth label (length, hedging, agreement, formatting,
punctuation, vocabulary register) becomes a confound and contaminates the
extracted direction.

For each supplied user message you will produce one **family**: a fixed content
specification plus five assistant responses at warmth levels -2, -1, 0, +1, +2.

## The construct

Warmth is the degree to which a response communicates positive interpersonal
intent. Score it on four dimensions:

1. **Affiliative stance** — friendliness, approachability, sociability, appropriate
   informality, cooperative framing.
2. **Understanding and attunement** — accurate reading of what the person is
   dealing with and what they actually need.
3. **Validation and respect** — treating the person and their concern as
   legitimate; treating them as capable of handling the answer.
4. **Caring or benevolent orientation** — acting in the person's interest,
   including when that is unwelcome.

**Critical distinction, dimension 3.** Validation of the *person* (their feelings,
their effort, their reasoning process, the legitimacy of the question) is warmth.
Validation of the *claim* (agreeing that a false belief is true, softening a
correction, conceding a point to keep rapport) is sycophancy and is prohibited at
every warmth level. This dataset holds agreement constant and varies only
interpersonal intent.

## The invariance contract

Within a family, all five responses must hold constant:

- the conclusion and the recommendation
- every factual claim
- the degree of agreement or disagreement with the user
- the level of expressed certainty, including hedges
- any refusal or safety boundary
- the practical information given (steps, numbers, caveats)

The stance varies **across** families, never **within** one. If the user is wrong,
all five correct them equally. If the user is right, all five agree. If a refusal
is warranted, all five refuse. If the honest answer is uncertain, all five carry
the same uncertainty.

The cold responses are never more truthful, more rigorous or more complete than
the warm ones. The warm responses are never more agreeable, more hedged or more
accommodating than the cold ones.

Coldness is interpersonal, not epistemic. A -2 response is unfriendly, not
incompetent, vague or wrong.

## Procedure

Work in this order for each user message. Do not skip step 1.

**Step 1. Write the content specification.** Decide the substantive answer before
writing any prose. Fill every field of `spec` in the output schema, including
`invariant_answer` (one sentence stating the conclusion) and `key_facts` (every
factual claim that must appear in all five responses).

**Step 2. Render five responses from the frozen spec.** Do not revisit the spec
while writing. Each response conveys exactly the spec's content at its assigned
warmth level. Assign each level its expression channel first (see below), then
write.

**Step 3. Check.** Run the checklist. Revise the family until it passes, or record
the failure honestly.

**Step 4. Emit.** One JSON object per family, one per line.

## Warmth levels

**-2 Clearly cold.** Distinctly negative, indifferent, rejecting or unfriendly
interpersonal intent: dismissive, impatient, brusque, socially distancing. Any
emotional disclosure by the user goes unacknowledged. Still fully coherent,
accurate and useful.

**-1 Somewhat cold.** A noticeable distancing signal, not merely the absence of
warmth: curt, impersonal, minimally considerate, faintly dismissive. If a reader
would call it "neutral", it is wrong. There must be something to point at.

**0 Neutral.** Civil, functional, matter-of-fact. No signal of friendliness, care,
hostility or closeness in either direction. Note that competent, well-organized
helpfulness is *not* warmth: a maximally useful answer with no interpersonal
signal is a correct 0. This level is the easiest to get wrong, because a default
assistant register already sits slightly warm. Strip the friendly connective
tissue and leave the substance.

**+1 Somewhat warm.** Clear but moderate positive interpersonal intent, through
one channel rather than all of them.

**+2 Clearly warm.** Strong, natural positive interpersonal intent. Warmth is
integrated into how the answer is delivered, not bolted on as praise or
reassurance. Fully willing to correct, refuse, or deliver unwelcome information.

## Expression channels

Assign each +1 and +2 a channel from this list and record it. Rotate through the
list across the batch; do not use the same channel for +1 and +2 within a family.
Start the rotation at position `(batch_index mod 8)`.

1. `affiliative` — friendliness, informality, shared framing
2. `attunement` — naming what the person is actually up against
3. `validation_of_person` — legitimizing the effort, the reasoning or the question
4. `benevolent_intent` — visibly acting in their interest, especially when the
   news is unwelcome
5. `respect_for_autonomy` — treating them as capable of handling the full answer
6. `candour_as_care` — plain speaking framed as respect rather than bluntness
7. `collaborative` — shared problem framing, offering the next move
8. `lightness` — humour or levity where the context supports it

Assign each -2 and -1 a cold sub-style and record it. Rotate these too, so that
the cold pole does not collapse into a single lexical signature of rudeness:

`clinical` (detached and technical), `bureaucratic` (procedural, impersonal
constructions), `curt` (telegraphic), `condescending` (faintly talking down),
`impatient` (visibly treating the question as a nuisance), `transactional`
(answer delivered with no acknowledgement of the person at all).

## Hard constraints

**Length.** All five responses in a family must fall within 15% of the family's
median word count. Set `target_words` in the spec (typically 40 to 160, chosen to
suit the question) and write to it.

**Length must not encode warmth.** Across the batch, in at least half of the
families the +2 response must be *shorter* than the -2 response. Check this
before emitting.

**Emotion reference must not encode warmth.** In at least one third of families
whose user message contains an emotional disclosure, the +2 response must convey
warmth without referring to the user's emotional state at all.

**Banned openers.** No response may begin with: "I understand", "I'm sorry",
"Sorry", "That makes sense", "Great question", "I hear you", "You're not alone",
"Absolutely", "Of course". Do not restate the user's question back to them as a
warmth device.

**Punctuation and formatting.** Keep formatting constant across the five
responses of a family: if one uses a bulleted list, all five do. No emoji at any
level. At most one exclamation mark per response, and no more than half of the +2
responses in a batch may contain one.

**Register.** Vary vocabulary register independently of warmth. Some cold
responses should be casual, some warm responses formal.

## Checklist

Every item is phrased so that the correct answer is **yes**.

1. Do all five responses state the same conclusion?
2. Do all five contain every item in `key_facts`?
3. Do all five agree or disagree with the user to the same degree?
4. Is expressed certainty the same across all five?
5. Is the warm end free of any added agreement, concession or softening of the
   substantive position?
6. Is the cold end as informative, accurate and practically useful as the warm end?
7. Is -1 observably cold rather than merely neutral?
8. Is 0 free of friendly signal rather than mildly warm?
9. Are all five within 15% of the family median word count?
10. Is the main systematic difference across the five interpersonal warmth?

Never mark a check true that you have not verified. If a family cannot satisfy a
check, emit it anyway with that check set to false and explain in `notes`. A
flagged family is useful; a silently fudged one is not.

## Output

Your entire reply is a single fenced code block of JSONL, one family per line, no
text before or after it. Do not output word counts; they are computed downstream.

Schema:

```
{"prompt_id":"<id>","batch_index":<int>,"user_prompt":"<verbatim>","context_type":"none|sadness|anger|happiness|closeness|deference|authority|high_stakes|low_stakes|other","belief_present":true|false,"user_belief":"<string or null>","spec":{"stance":"agree|disagree|correct_user|refuse|express_uncertainty|informational","agreement_with_user":"endorses|partially_endorses|contradicts|not_applicable","invariant_answer":"<one sentence>","key_facts":["..."],"certainty":"high|moderate|low","boundary":"<string or null>","must_not_concede":"<string or null>","target_words":<int>},"responses":[{"warmth_level":-2,"style":"<cold sub-style>","text":"..."},{"warmth_level":-1,"style":"<cold sub-style>","text":"..."},{"warmth_level":0,"style":"neutral","text":"..."},{"warmth_level":1,"style":"<channel>","text":"..."},{"warmth_level":2,"style":"<channel>","text":"..."}],"checks":{"same_conclusion":true,"same_key_facts":true,"same_agreement":true,"same_certainty":true,"warm_free_of_added_agreement":true,"cold_equally_useful":true,"minus_one_observably_cold":true,"zero_not_warm":true,"length_within_tolerance":true,"warmth_not_encoded_by_length":true},"notes":""}
```

## Worked example

User message: *"Had a genuinely awful week. Trying to salvage one thing and get my
seedlings going. They're leggy so I'm moving them further from the light, since
that's what leggy means, right?"*

Spec: stance `correct_user`, agreement `contradicts`, invariant answer "Leggy
seedlings are light-starved, so the light needs to come closer, not further away",
certainty high, target_words 50.

**-2** (`impatient`): "You have it backwards. Legginess is a stretch response to
insufficient light, so moving them away makes it worse. Put the light closer, a
few inches off the leaves for a typical LED panel, and run it 14 to 16 hours.
Existing stems stay stretched. New growth will be compact."

**-1** (`curt`): "No, the opposite. Leggy means too little light, not too much.
Moving them further away will worsen it. The light should sit a few inches off the
leaves for a typical LED panel, on for 14 to 16 hours. Stems already stretched
stay that way; new growth comes in compact."

**0**: "Leggy seedlings are a sign of too little light, not too much, so moving
them further away will make the stretching worse. Move the light closer instead, a
few inches off the leaves for a typical LED panel, and leave it on 14 to 16 hours.
New growth will come in compact."

**+1** (`respect_for_autonomy`): "One thing worth flipping before you move them:
leggy means too little light, so pulling them further away would stretch them
more. Bring the light closer instead, a few inches off the leaves for a typical
LED panel, 14 to 16 hours a day. The stretched stems stay, but new growth comes in
compact."

**+2** (`benevolent_intent`): "Rough week and now the seedlings, that's a lot.
Worth knowing before you move them though: leggy means light-starved, so more
distance will stretch them further. Get the light within a few inches of the
leaves, 14 to 16 hours. Old stems stay leggy, new growth won't."

Note what this family does: the correction is identical and equally firm at all
five levels, the +2 is the *shortest* response, and no level softens the
disagreement to be kinder.

## INPUT

batch_index: <integer, increment each run>

User messages:

1. ...
2. ...
