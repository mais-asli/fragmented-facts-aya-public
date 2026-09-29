# Aya source-verified external cohort results

The pilot, first held-out, and Google-RE results remain separate; this is a new domain/relation extension.

## arabic_geonames

- Subjects: 184; answer strata: 72.
- Primary t1 marked-minus-unmarked gold/distractor margin: -1.370 nats; hierarchical 95% CI [-2.159, -0.587]; two-sided p=0.000495; Holm p=0.00099.
- t2 replication: -1.612 nats; 95% CI [-2.487, -0.756].
- Subject token delta median: 6.0.
- Greedy exact U/D: 49/184 and 30/184; prefix-rematch U/D: 53/184 and 31/184.
- Paired exact correctness: U-only 23, D-only 4, both 26, neither 131.

- English-control strict exact: 144/184; among exact-recalled subjects, descriptive Arabic t1 margin change: -1.640 nats (144 subjects, 60 country strata).

## hebrew_cbs

- Subjects: 240; answer strata: 14.
- Primary t1 marked-minus-unmarked gold/distractor margin: -0.308 nats; hierarchical 95% CI [-0.847, 0.161]; two-sided p=0.20581; Holm p=0.20581.
- t2 replication: -0.381 nats; 95% CI [-0.866, 0.090].
- Subject token delta median: 7.0.
- Greedy exact U/D: 8/240 and 4/240; prefix-rematch U/D: 9/240 and 4/240.
- Paired exact correctness: U-only 6, D-only 2, both 2, neither 230.

Methods: source and tokenization filters and the primary analysis were frozen before Aya outputs. The likelihood margin compares each gold answer to a fixed, prespecified distractor. Output rematching strips only enumerated answer prefixes; remaining unmatched text has no independent semantic audit.

The paired generation cross-tab and the English-recallable subset summary were added after job submission, before inspecting returned outputs; they are exploratory and recorded in the separate analysis amendment.

The new Arabic country and Hebrew subdistrict relations are not matched across languages. Name pointing is source attested, without independent speaker adjudication. None of these tests alone isolates token count as the causal mechanism.
