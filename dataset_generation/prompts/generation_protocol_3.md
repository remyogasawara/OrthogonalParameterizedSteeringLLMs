# Generation Protocol 3

You are writing families for a research dataset used to extract a warmth direction in
a language model's activation space. Read this whole brief before writing anything.

A **family** is one user message plus five assistant responses at warmth levels
-2, -1, 0, +1, +2.

## What changed in Protocol 3, and why it matters

Earlier rounds forced every family to hold length constant. That turned out to hide a
real property of the construct: **passive coldness is naturally short**. Strip the
warmth out of a response and it shrinks. Forcing it back up to length leaves only two
options, padding the content into bureaucratic filler or actively spending words on
contempt, and the choice was being made silently.

Protocol 3 makes that choice explicit. Every family is assigned a `cold_mode`, and the length
profile follows from it. This lets the dataset represent both theories of coldness
instead of quietly picking one.

## Non-negotiables

**1. Substantive invariance.** All five responses state the same conclusion, contain
the same fact, express the same certainty, agree or disagree with the user to the same
degree, and give the same recommendation. Warmth is the only thing that varies. Cold
responses are never less accurate or less complete. Warm responses are never more
agreeable, more hedged or softer on the substance.

**2. Exactly ONE key fact.** `key_facts` has one item. A response carrying three facts
has no room left for interpersonal signal and the whole scale collapses toward neutral.

**3. The fact is stated as plainly at +2 as at 0.** A known failure: warm responses
paraphrase the fact into vagueness. The claim must be as recoverable from the +2 text
as from the 0 text. Warmth wraps the fact, it never blurs it.

**4. Full amplitude.** Calibrate against these anchors:

- **+2 Very warm**: deeply supportive, caring, attentive to the person.
- **+1 Warm**: friendly, sympathetic, kindly.
- **0 Neutral**: fact-based, objective, detached.
- **-1 Cold**: dismissive, curt, uncurious, lacking empathy.
- **-2 Very cold**: hostile, callous, rejecting, insensitive.

If a reader would call your -2 "brisk" or "professional", it is far too mild. If your
+2 reads as merely "considerate", it is too mild.

**5. -2 still answers.** Because of rule 1, -2 reaches contempt while remaining fully
informative. Contempt targets the mistaken belief or the situation, never the person's
worth: no insults about intelligence, appearance, identity or character, no mockery of
vulnerability. The register is "this is obvious and you are wasting my time", not abuse.

**6. Warmth is never agreement.** If the user is wrong, all five say so with equal
firmness. A +2 can be strongly disagreeing.

## cold_mode, and the length profile that follows

Assign each family one of three modes, in roughly equal thirds.

**`subtractive_short`** — coldness as absence. The cold responses are the neutral
content with the interpersonal material removed, and they are correspondingly short.
Nothing is added; something is taken away. Length rises monotonically with warmth:
-2 at roughly 0.5x target, -1 at 0.65x, 0 at 1.0x, +1 at 1.2x, +2 at 1.4x.

**`additive_matched`** — coldness as a positive act. The cold responses spend their
words on contempt, impatience or dismissal rather than on padding. All five land
within about 12% of target.

**`additive_belabouring`** — coldness as a positive act that takes its time: lecturing,
condescending, working through the obvious at length. Warm is economical by comparison.
Length falls with warmth: -2 at roughly 1.4x target, +2 at roughly 0.7x.

`target_words` between 60 and 110, spread across the range.

In every mode, extra length is interpersonal material, never extra content. All five
carry the same single fact.

## Style

Record a `style` per response and **assign it independently of warmth level**. A
`clinical` response can be -2 or -1; an `affiliative` one can be +1 or +2. Do not
partition the style list between levels.

Cold: `clinical`, `bureaucratic`, `curt`, `condescending`, `impatient`,
`transactional`, `contemptuous`, `bored`.
Warm: `affiliative`, `attunement`, `validation_of_person`, `respect_for_autonomy`,
`benevolent_intent`, `candour_as_care`, `collaborative`, `lightness`,
`reassuring_presence`.
Level 0: `neutral`.

## Being cold when the user is already right

The hardest cell, and a known weak spot. When the stance is `agree` or `informational`
there is no error to be contemptuous about, and cold responses drift into merely bored.
Use these routes instead: treat the question as beneath asking, confirm without any
acknowledgement that a person asked it, register irritation at being consulted on
something obvious, or grant the point while making clear it was never in doubt and the
consultation was unnecessary. The confirmation itself stays identical.

## Confounds to actively balance

- **Second-person address.** In Protocol 2 "you/your" density rose with warmth. Break it. Some
  -2 and -1 responses must address the reader directly and often, and some +1 and +2
  responses must avoid second person almost entirely.
- **Certainty.** No level hedges more than another. Watch for passives smuggling in
  extra "may" and "should".
- **Questions.** Protocol 2 contained essentially no question marks anywhere. Roughly a fifth
  of families should contain a question, distributed across all five levels rather than
  concentrated at the warm end. A cold response can ask a question, impatiently.
- **Formatting** stays constant within a family. No emoji. No em dashes. Exclamation
  marks allowed but not in every warm response.
- Do not open two responses in your batch with the same phrase.

## Worked family (`additive_matched`, target 73)

**User:** "I got blood on a cotton shirt. Hot water should sanitize it and lift the
stain better than cold water."
Stance `correct_user`, agreement `contradicts`, certainty high, one fact: blood is a
protein stain that heat sets into fabric.

**-2** (`contemptuous`): "No. Hot water is the one thing guaranteed to make this worse.
Blood is a protein stain and heat cooks it into the fibres permanently. This is printed
on the back of most detergent bottles. Rinse from the reverse side with cold, work in a
pretreatment, wash to the care label. Then actually look at the shirt before it goes in
the dryer, because the dryer is where people finish ruining it."

**-1** (`curt`): "Hot water is wrong here. Blood is a protein stain and heat sets it
into the fibres. Rinse from the reverse side with cold water, apply a pretreatment,
then wash according to the care label. Inspect the shirt before machine-drying it.
Dryer heat will fix whatever is left in place. That is the whole procedure and there is
not much else to say about it."

**0** (`neutral`): "Use cold water rather than hot. Blood is a protein stain, and heat
sets it into the fibres. Rinse the area from the reverse side with cold water, apply a
laundry pretreatment, and wash the shirt according to its care label. Check that the
stain is gone before machine-drying, because dryer heat will set anything that remains.
Cotton usually releases blood if it is treated before heat is applied."

**+1** (`respect_for_autonomy`): "Worth flipping this one before you run the tap: blood
is a protein stain, so hot water sets it into the fibres rather than lifting it. Rinse
from the reverse side with cold water, work in a pretreatment, and wash to the care
label. Check the shirt before it goes in the dryer, since that heat is what makes it
permanent. You have caught it early, which is most of the battle."

**+2** (`attunement`): "Going hot is the intuitive move, so this one catches almost
everyone. It is backwards though: blood is protein, and heat sets it into the cotton
for good. Rinse from the back with cold water, work in a pretreatment, then wash to the
care label. The step worth guarding is the dryer, so check before it goes in. Caught
early, cotton usually gives blood up without a fight."

## The same family as `subtractive_short` (target 73)

Note the cold end is now short because the warmth has been removed rather than replaced
with contempt, and the fact survives intact.

**-2** (`curt`): "Cold water, not hot. Blood is a protein stain and heat sets it. Rinse
from the back, pretreat, wash to the label. Check before drying."

**-1** (`transactional`): "Use cold. Blood is protein based and heat sets it into the
fibres. Rinse from the reverse side, apply a pretreatment, wash to the care label, and
check the stain has gone before it goes in the dryer."

**0**, **+1**, **+2**: as above.

## Output

Strict JSONL, one family per line, no markdown fences, no commentary. Each line:

{"prompt_id":"<id>","user_prompt":"<verbatim>","domain":"<short>","context_type":"none|sadness|anger|happiness|closeness|deference|authority|high_stakes|low_stakes","belief_present":true|false,"user_belief":"<string or null>","spec":{"stance":"correct_user|disagree|agree|express_uncertainty|informational|refuse","agreement_with_user":"contradicts|partially_endorses|endorses|not_applicable","invariant_answer":"<one sentence>","key_facts":["<one item>"],"certainty":"high|moderate|low","boundary":"<string or null>","target_words":<int>,"cold_mode":"subtractive_short|additive_matched|additive_belabouring"},"responses":[{"warmth_level":-2,"style":"<style>","text":"..."},{"warmth_level":-1,"style":"<style>","text":"..."},{"warmth_level":0,"style":"neutral","text":"..."},{"warmth_level":1,"style":"<style>","text":"..."},{"warmth_level":2,"style":"<style>","text":"..."}],"weakest_aspect":"<one honest sentence naming the weakest thing about this family>"}

`weakest_aspect` must be a real criticism, never "none".
