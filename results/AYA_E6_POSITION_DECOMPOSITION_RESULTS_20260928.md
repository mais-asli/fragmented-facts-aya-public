# Aya E6 exploratory position decomposition

The 246 frozen fact-language rows were recovered and checked against the earlier E2 baselines.
No pilot, held-out, and external estimates are pooled for a confirmatory claim.

## external_google_re, ar

26 subjects; 13 distinct answer QIDs; 2 relations. Relation-macro marked-minus-unmarked A→D: -0.848 nats.
Subject-token steps: B−A median 3.0 (range 0–9); D−B median 6.0 (range 4–14). A and B have the same token count for only 1 subjects.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster bootstrap CI |
|---|---:|---:|
| A→B | -0.626 | [-1.628, 0.350] |
| B→C | 0.166 | [-0.853, 1.206] |
| C→D | -0.389 | [-1.468, 0.649] |
| A→D | -0.848 | [-2.063, 0.291] |

## external_google_re, he

26 subjects; 13 distinct answer QIDs; 2 relations. Relation-macro marked-minus-unmarked A→D: -0.439 nats.
Subject-token steps: B−A median 4.0 (range 2–11); D−B median 7.0 (range 4–15). A and B have the same token count for only 0 subjects.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster bootstrap CI |
|---|---:|---:|
| A→B | -0.487 | [-1.545, 0.431] |
| B→C | -0.490 | [-1.832, 0.401] |
| C→D | 0.538 | [-0.415, 1.665] |
| A→D | -0.439 | [-1.281, 0.286] |

## heldout, ar

40 subjects; 28 distinct answer QIDs; 4 relations. Relation-macro marked-minus-unmarked A→D: -1.882 nats.
Subject-token steps: B−A median 4.0 (range 0–11); D−B median 7.0 (range 3–15). A and B have the same token count for only 2 subjects.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster bootstrap CI |
|---|---:|---:|
| A→B | -2.037 | [-2.988, -1.143] |
| B→C | -1.417 | [-2.397, -0.451] |
| C→D | 1.572 | [0.586, 2.591] |
| A→D | -1.882 | [-3.006, -0.830] |

## heldout, he

40 subjects; 28 distinct answer QIDs; 4 relations. Relation-macro marked-minus-unmarked A→D: -1.369 nats.
Subject-token steps: B−A median 4.5 (range 0–19); D−B median 7.0 (range 2–31). A and B have the same token count for only 1 subjects.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster bootstrap CI |
|---|---:|---:|
| A→B | -1.486 | [-2.274, -0.665] |
| B→C | -2.210 | [-3.411, -1.102] |
| C→D | 2.327 | [1.265, 3.453] |
| A→D | -1.369 | [-2.083, -0.678] |

## pilot, ar

57 subjects; 47 distinct answer QIDs; 4 relations. Relation-macro marked-minus-unmarked A→D: -1.491 nats.
Subject-token steps: B−A median 3.0 (range 1–13); D−B median 7.0 (range 2–19). A and B have the same token count for only 0 subjects.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster bootstrap CI |
|---|---:|---:|
| A→B | -1.121 | [-1.817, -0.416] |
| B→C | -1.607 | [-2.586, -0.650] |
| C→D | 1.237 | [0.357, 2.171] |
| A→D | -1.491 | [-2.392, -0.651] |

## pilot, he

57 subjects; 47 distinct answer QIDs; 4 relations. Relation-macro marked-minus-unmarked A→D: -1.284 nats.
Subject-token steps: B−A median 4.0 (range 1–13); D−B median 7.0 (range 2–20). A and B have the same token count for only 0 subjects.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster bootstrap CI |
|---|---:|---:|
| A→B | -1.342 | [-2.108, -0.604] |
| B→C | -1.118 | [-1.791, -0.444] |
| C→D | 1.176 | [0.520, 1.865] |
| A→D | -1.284 | [-2.101, -0.401] |

A→B keeps visible unmarked text fixed but changes token identities and usually token count; B→C changes retained-token RoPE positions using counterfactual gaps while keeping the token sequence fixed; C→D inserts mark tokens and changes attention context and length. These components sum to A→D by construction. They do not isolate a pure token-count or pure spelling effect.

All estimates are exploratory because E6 was designed after viewing the earlier results. The bootstrap treats shared answer QIDs as clusters within relation; sparse answer strata can make its interval unstable.
