# Aya source-disjoint activation-patching replication

The patch site (layers 12–13, last subject token) was fixed from the earlier Aya work. This run uses new source-disjoint place subjects and frozen, different-answer donors.

## arabic_geonames

- 96 subjects, 72 answer strata.
- Same-entity U→D patch minus unpatched D, gold/distractor margin: 1.327 nats; hierarchical 95% CI [0.640, 2.058]; two-sided stratum sign-flip p=0.00022; Holm p=0.00044.
- Same-entity minus unrelated-donor margin: 2.607 nats; 95% CI [1.636, 3.615] (control contrast, descriptive).
- Identity-patch maximum absolute margin deviation: 0.0345 nats. E2 baseline maximum score deviation: 0.0000 nats.

## hebrew_cbs

- 96 subjects, 14 answer strata.
- Same-entity U→D patch minus unpatched D, gold/distractor margin: 0.357 nats; hierarchical 95% CI [-0.185, 0.960]; two-sided stratum sign-flip p=0.19702; Holm p=0.19702.
- Same-entity minus unrelated-donor margin: 0.067 nats; 95% CI [-0.654, 0.839] (control contrast, descriptive).
- Identity-patch maximum absolute margin deviation: 0.0271 nats. E2 baseline maximum score deviation: 0.0000 nats.

All controls were fixed before these E5 outputs. The site came from prior Aya experiments, and the new relations differ by language; the result tests transfer of the patch effect to new sources rather than independent discovery of a circuit. Whole-residual intervention does not separate token count from mark identity or spelling familiarity.
