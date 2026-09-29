# Aya E7 same-visible-text, matched-count controls

Exploratory follow-up designed after viewing E6. All 246 E6 fact-language pairs were included. The new forced-split sequences were frozen before the E7 Aya output. Every candidate decodes to the same visible unmarked prompt; the early and late versions have exactly the marked prompt's subject-token count.

Outcome: total log likelihood of the same canonical gold answer, in nats. B and D come from the previously verified E6 archive. Each cell is summarized separately, with answer-QID cluster bootstrap within relation and relation-macro aggregation. No confirmatory p-values are claimed.

## external_google_re, ar

26 subjects; 13 answer QIDs; 2 relations.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster CI |
|---|---:|---:|
| D minus full mean | 0.430 | [-0.603, 1.512] |
| early minus late | -0.730 | [-1.812, 0.482] |
| mid minus B | -0.426 | [-1.170, 0.256] |
| early minus B | -1.017 | [-2.003, -0.094] |
| late minus B | -0.287 | [-1.697, 0.972] |
| full mean minus mid | -0.226 | [-0.956, 0.446] |
| D minus B | -0.222 | [-0.947, 0.509] |
| D minus A | -0.848 | [-1.988, 0.273] |

## external_google_re, he

26 subjects; 13 answer QIDs; 2 relations.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster CI |
|---|---:|---:|
| D minus full mean | 0.355 | [-0.471, 1.090] |
| early minus late | -0.097 | [-0.915, 0.783] |
| mid minus B | -0.224 | [-0.736, 0.179] |
| early minus B | -0.355 | [-1.105, 0.432] |
| late minus B | -0.258 | [-0.944, 0.361] |
| full mean minus mid | -0.083 | [-0.599, 0.501] |
| D minus B | 0.048 | [-0.676, 0.780] |
| D minus A | -0.439 | [-1.248, 0.289] |

## heldout, ar

40 subjects; 28 answer QIDs; 4 relations.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster CI |
|---|---:|---:|
| D minus full mean | 2.273 | [1.277, 3.239] |
| early minus late | 0.198 | [-1.005, 1.462] |
| mid minus B | -0.376 | [-1.328, 0.432] |
| early minus B | -2.019 | [-3.236, -0.782] |
| late minus B | -2.217 | [-3.236, -1.225] |
| full mean minus mid | -1.742 | [-2.605, -0.894] |
| D minus B | 0.155 | [-0.445, 0.734] |
| D minus A | -1.882 | [-3.011, -0.829] |

## heldout, he

40 subjects; 28 answer QIDs; 4 relations.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster CI |
|---|---:|---:|
| D minus full mean | 1.648 | [0.751, 2.603] |
| early minus late | 0.784 | [-0.938, 2.456] |
| mid minus B | -0.915 | [-2.080, -0.054] |
| early minus B | -1.139 | [-2.575, -0.034] |
| late minus B | -1.923 | [-3.088, -0.878] |
| full mean minus mid | -0.617 | [-1.435, 0.151] |
| D minus B | 0.117 | [-0.515, 0.687] |
| D minus A | -1.369 | [-2.097, -0.677] |

## pilot, ar

57 subjects; 47 answer QIDs; 4 relations.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster CI |
|---|---:|---:|
| D minus full mean | 1.855 | [1.036, 2.699] |
| early minus late | 0.462 | [-0.598, 1.588] |
| mid minus B | -1.315 | [-2.030, -0.603] |
| early minus B | -1.994 | [-2.896, -1.106] |
| late minus B | -2.455 | [-3.576, -1.363] |
| full mean minus mid | -0.910 | [-1.613, -0.262] |
| D minus B | -0.370 | [-0.964, 0.257] |
| D minus A | -1.491 | [-2.357, -0.677] |

## pilot, he

57 subjects; 47 answer QIDs; 4 relations.

| Contrast | Relation-macro mean (nats) | 95% answer-cluster CI |
|---|---:|---:|
| D minus full mean | 1.035 | [0.449, 1.652] |
| early minus late | 0.085 | [-0.649, 0.832] |
| mid minus B | -0.528 | [-1.099, 0.014] |
| early minus B | -0.934 | [-1.696, -0.203] |
| late minus B | -1.019 | [-1.828, -0.186] |
| full mean minus mid | -0.449 | [-1.043, 0.147] |
| D minus B | 0.058 | [-0.506, 0.664] |
| D minus A | -1.284 | [-2.115, -0.415] |

B→mid→full compares the same visible unmarked prompt at increasing subject-token counts, but forced splitting necessarily changes token identities. Early and late are different same-count split patterns. D versus their mean matches the subject-token count but changes marked spelling, token identities, and segmentation. Therefore E7 can test robustness of a simple length explanation; it cannot identify a pure causal count effect. All forced splits are outside ordinary tokenizer output and may themselves impair Aya.

This likelihood-only diagnostic does not establish generated-answer accuracy, knowledge loss, or a unique neural mechanism.
