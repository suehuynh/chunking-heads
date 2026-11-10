# How do lexical task heads (LTH) verbalize the task? Hypothesis and findings

Llama-3.2-1B-Instruct, 5-shot `input: output;` prompts, single-token items. Category task: country-capital.
Relation task: antonym. Last updated 2026-11-09.

## Hypothesis

> LTH name the task by **averaging what sits at the demo-output positions**. What they can name depends on
> what those positions hold: a **shared category** (category tasks), or **relation information written in by
> context** (relation tasks).

| part | claim | status |
|---|---|---|
| **H-a, structure** | Symbolic induction heads (L7H5, L7H28, L7H6) send LTH attention to output positions by role, not content | Supported for antonym: ablating them halves LTH attention to outputs (0.28 → 0.15) and accuracy (0.43 → 0.21). Country-capital doesn't need them. |
| **H-b, content (revised)** | Category LTH output the average of their output-position values; each category member maps toward the category word, and averaging strengthens the signal and grades it by coherence | Mechanism supported (output ≈ output-slot mean, cosine 0.97; only the mean matters). "Emergence" not supported: a single city already decodes to "city". |
| **H-c, relation enrichment** | In relation tasks, context writes the relation into each output position, for correct pairs only, and relation LTH average that | Supported: antonym outputs give "antonym" in L8H31 only in clean context (0.68 → 1.00 with averaging), never alone or in shuffled context. Output-slot values necessary (0.64 → 0.02) and partly sufficient (0.00 → 0.23). |

## Types of heads

| type | heads | reads from | depends on | evidence |
|---|---|---|---|---|
| Symbolic induction | L7H5, L7H28, L7H6 | demo outputs, by role (0.94 attention, unchanged with fillers) | — | L7H5 averages outputs (cosine 0.99) but never names a category |
| Role-bound output averager (category LTH) | L9H4 (+ L8H13) | demo outputs, plus an informative query | L9H4: nothing upstream; L8H13: induction heads (animal 0.91 → 0.45 when ablated) | output ≈ mean of output slots; never reads inputs (animals at inputs → 0.00); in country-capital, "animal" rate linear in # animal outputs (see correction below) |
| Content-seeking (category LTH) | L11H22, L11H17, L12H27, L12H24 | most salient category content: query by default, outputs when pairs mismatch, inputs when outputs are uninformative | — | attention shifts query → outputs as mismatched pairs grow 0 → 5; low cosine with output mean (0.35–0.46) |
| Relation LTH | L8H31 | output positions enriched by context | binding + induction heads | never says "animal"; dies with any pairing break; binder ablation: accuracy 0.43 → 0.06 |
| Binding heads | L5H4, L6H17 (cc); L5H22, L6H14, L5H6 (antonym) | each output → its own input | — | pair-binding > 95th percentile and ≥ 2× control |

## Category vs relation task

| | category (country-capital) | relation (antonym) |
|---|---|---|
| LTH need | output positions sharing a category | correct pairs (relation in output positions) |
| Shuffled pairs | still verbalized (0.93) | collapses (0.09) |
| Animal outputs | says "animal" | L8H31 never does |
| Induction-head ablation | no effect on clean | verbalization and accuracy drop |
| Binder ablation | small (swapped roles only) | accuracy 0.43 → 0.06 |

## Anomalies resolved
1. **Swapped roles still give "city"**: the query was a city. With a filler query, L9H4 drops 1.00 → 0.11; content-seeking heads still read cities at inputs.
2. **Clean (0.56) < shuffled (0.93)** (maps_top_lth): content-seeking heads read the query in clean prompts and move to the outputs as pairs mismatch, rising steadily with # mismatched pairs. L9H4 stays at 1.00.
3. **Term list**: the country-capital match is mostly "city" — L9H4 names the output category, not "capital-of".

## Coherence curve (k of 5 demo outputs are animals, rest fillers)
Head output ≈ Σⱼ aⱼ · m(itemⱼ) (attention to output slot j × item's mapping toward "animal"). Shape tells
whether attention a is fixed (linear) or grows with content (convex/threshold).

| head (task) | "animal" rate k=0→5 | attention to outputs k=0→5 | reading |
|---|---|---|---|
| L9H4 (country-capital) | .00 .09 .25 .39 .55 .70 | 0.58 → 0.75 (high, near fixed) | linear: fixed-attention averager |
| L8H13 (country-capital) | .00 .02 .07 .09 .48 .91 | 0.18 → 0.39 (query 0.21 → 0.14) | threshold: content and attention both grow |
| L12H27 (country-capital) | .00 .00 .02 .14 .41 .84 | 0.15 → 0.38 | threshold |
| L9H4 (antonym) | .00 .02 .06 .08 .27 .55 | 0.36 → 0.52 (query 0.26 → 0.20) | convex |
| L8H31 (antonym) | ≤ .01 | flat ~0.28 | relation head, no category |

**Correction (2026-11-09):** earlier notes called L9H4's curve "roughly linear" in both tasks. In antonym it is
**convex**: its attention to outputs starts lower (0.36, with 0.26 on the query) and rises with k. So L9H4 is
role-bound (fixed attention on outputs) only when the outputs already form a category; otherwise it behaves
more like a content-seeking head. Caveat: the top-10 rate exaggerates thresholds; the continuous score is
smoother (L9H4 cc: 3.9 → 8.3, rising fast then levelling off). Attention values are role-group sums, not per position.

**Related (2026-11-09):** L9H4's *total* attention on outputs is content-gated even though it rarely moves to
inputs: clean 0.52, unrelated outputs 0.58, random both 0.20, swapped roles 0.28, swapped + filler query 0.08.
L7H5 by contrast stays ≈0.94 on output slots in every condition with a real query (pure role selection).

## Other findings
- Naming the task is separable from doing it: value patches on L9H4 change what it says, not accuracy.
- L7H5 → L9H4 (path-patching edge) likely matters for the answer logit, not task naming.
- Retracted: "induction heads override L9H4's query default" (4/44 prompts; refuted by the 12-head ablation).

## Open questions
- What drives the content-seeking heads' attention shift (unexplained outputs draw attention?) — needs per-position attention.
- L11H22 reads animals at inputs only when real countries are at outputs.
- Do binding heads write the relation into antonym output positions? (binder-output patching)
- Why does L9H4 work without induction heads while L8H13 doesn't?

## Caveats
One model, one seed, 44 country-capital items; controls vary across draws (only effects outside the control
range count); ablations show necessity, not flow; one task per type so far.

## Next steps
1. Second task of each type (park-country, present-past).
2. Binder-output patching at output positions (H-c).
3. Per-position attention for the mismatch effect.
4. Replicate coherence curve, mean swap, and antonym in-context-only result on Llama-3.1-8B.
