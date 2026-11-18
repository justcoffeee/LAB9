# Error analysis — BERT NER on CoNLL-2003 test

AI-assisted technical evidence, not the student's personal error analysis. W9 section 28 requires the student to add their own interpretation. The descriptions below compare spans mechanically, not causal explanations.

Test sentences: **3,453**. Entity-level disagreements: **531 false positives**, **458 false negatives**, **169 overlapping-but-not-identical span pairs**.

Error categories are assigned mechanically from span overlap: an unmatched prediction that overlaps a gold entity of another type is `wrong entity type`; an unmatched span that overlaps a gold entity of the same type is a boundary error; an unmatched span with no overlap is `spurious` / `missing`. Exact matches are never listed.

## False positives (5)

| # | Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|---|
| 1 | SOCCER - JAPAN GET LUCKY WIN, CHINA IN SURPRISE DEFEAT. | JAPAN | JAPAN (PER) | wrong entity type | The span matches a gold LOC entity ('JAPAN') but the model labelled it PER. |
| 2 | Burmese troops and riot police moved in to disperse a student street protest at a suburban road junction near the Ranyon ( Rangoon) University early on Saturday, witnesses said. | Ranyon ( Rangoon ) University | Ranyon (ORG) | wrong boundary | Gold 'Ranyon ( Rangoon ) University' (ORG, 5 tokens); prediction 'Ranyon' differs by +0 token(s) at the start and -4 at the end. |
| 3 | SOCCER - JAPAN GET LUCKY WIN, CHINA IN SURPRISE DEFEAT. | O | LUCKY (LOC) | spurious entity | No gold entity overlaps this span; the model turned 'LUCKY' into a LOC entity. |
| 4 | SOCCER - JAPAN GET LUCKY WIN, CHINA IN SURPRISE DEFEAT. | CHINA | CHINA (ORG) | wrong entity type | The span matches a gold PER entity ('CHINA') but the model labelled it ORG. |
| 5 | Defender Hassan Abbas rose to intercept a long ball into the area in the 84th minute but only managed to divert it into the top corner of Bitar's goal. | Bitar | Bitar (ORG) | wrong entity type | The span matches a gold PER entity ('Bitar') but the model labelled it ORG. |

## False negatives (5)

| # | Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|---|
| 1 | SOCCER - JAPAN GET LUCKY WIN, CHINA IN SURPRISE DEFEAT. | JAPAN (LOC) | JAPAN (PER) | wrong entity type | Gold LOC 'JAPAN' overlaps a predicted PER span 'JAPAN'. |
| 2 | SOCCER - JAPAN GET LUCKY WIN, CHINA IN SURPRISE DEFEAT. | CHINA (PER) | CHINA (ORG) | wrong entity type | Gold PER 'CHINA' overlaps a predicted ORG span 'CHINA'. |
| 3 | Cuttitta announced his retirement after the 1995 World Cup, where he took issue with being dropped from the Italy side that faced England in the pool stages. | 1995 World Cup (MISC) | World Cup (MISC) | wrong boundary | Gold span [6,9) vs predicted [7,9); overlap 2/3 gold tokens. |
| 4 | FIFA's players' status committee, meeting in Barcelona, decided that although the Udinese document was basically valid, it could not be legally protected. | Udinese (ORG) | Udinese (MISC) | wrong entity type | Gold ORG 'Udinese' overlaps a predicted MISC span 'Udinese'. |
| 5 | Blinker was fined 75,000 Swiss francs ( $ 57,600) for failing to inform the Engllsh club of his previous commitment to Udinese. | Engllsh (MISC) | Engllsh (ORG) | wrong entity type | Gold MISC 'Engllsh' overlaps a predicted ORG span 'Engllsh'. |

## Boundary errors (5)

| # | Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|---|
| 1 | Cuttitta announced his retirement after the 1995 World Cup, where he took issue with being dropped from the Italy side that faced England in the pool stages. | 1995 World Cup (MISC) | World Cup (MISC) | wrong boundary | Gold span [6,9) vs predicted [7,9); overlap 2/3 tokens, types MISC -> MISC. |
| 2 | Stefanel Milan ( Italy) 9 6 3 15 | Stefanel Milan (ORG) | Stefanel (PER) | wrong boundary + type | Gold span [0,2) vs predicted [0,1); overlap 1/2 tokens, types ORG -> PER. |
| 3 | CRICKET - AUSTRALIA BEAT WEST INDIES BY FIVE WICKETS. | WEST INDIES (LOC) | INDIES (LOC) | wrong boundary | Gold span [4,6) vs predicted [5,6); overlap 1/2 tokens, types LOC -> LOC. |
| 4 | All-conquering Juventus field their most recent signing, Portuguese defender Dimas, while Alessandro Del Piero and Croat Alen Boksic lead the attack. | Alen Boksic (PER) | Croat Alen Boksic (PER) | wrong boundary | Gold span [18,20) vs predicted [17,20); overlap 2/3 tokens, types PER -> PER. |
| 5 | SQUASH - MAHINDRA INTERNATIONAL SEMIFINAL RESULTS. | MAHINDRA INTERNATIONAL (MISC) | MAHINDRA (LOC) | wrong boundary + type | Gold span [2,4) vs predicted [2,3); overlap 1/2 tokens, types MISC -> LOC. |

## Context analysis — same surface form, different entity type

The three forms below were found automatically by grouping gold entities by their exact surface form and keeping the forms with at least two types.

### `ARIZONA` -> LOC, ORG

- *ARIZONA 6 7 0 248 332*  
  gold **ORG**, predicted **ORG** (tags: B-ORG).
- *DALLAS AT ARIZONA*  
  gold **LOC**, predicted **LOC** (tags: B-LOC).

### `ATLANTA` -> LOC, ORG

- *ATLANTA 10 8 .556 7*  
  gold **ORG**, predicted **ORG** (tags: B-ORG).
- *ATLANTA 2 11 0 234 393*  
  gold **ORG**, predicted **ORG** (tags: B-ORG).
- *ATLANTA AT NEW ORLEANS*  
  gold **ORG**, predicted **ORG** (tags: B-ORG).
- *ATLANTA 10 8 .556 7*  
  gold **ORG**, predicted **ORG** (tags: B-ORG).
- *TORONTO AT ATLANTA*  
  gold **LOC**, predicted **LOC** (tags: B-LOC).

### `American` -> MISC, ORG

- *In a statement Havel, who is recovering from cancer surgery, said: " Madeleine Albright is a distinguished friend, a tested diplomat, and a true American of fine origins. "*  
  gold **MISC**, predicted **MISC** (tags: B-MISC).
- *Britain sets conditions to clear American alliance.*  
  gold **MISC**, predicted **MISC** (tags: B-MISC).
- *The conditions for clearance of the alliance were that British Airways and American drop 168 slots at London Heathrow airport, the busiest in Europe.*  
  gold **ORG**, predicted **ORG** (tags: B-ORG).
- *American's parent, AMR Corp., said it did not view the terms as a " deal breaker. "*  
  gold **ORG**, predicted **ORG** (tags: B-ORG).
- *The government also wants British Airways to drop a clause in its agreement with USAir that bars it from competing on trans-Atlantic routes, and said both British Airways and American should be prepared to reduce services on the London to Dallas-Fort Worth route in the event that a new entrant wishes to enter.*  
  gold **ORG**, predicted **ORG** (tags: B-ORG).
- *The Office of Fair Trade called for British Airways / American to allow third-party access to their joint frequent flyer programme where the applicant does not have access to an equivalent programme.*  
  gold **ORG**, predicted **O** (tags: B-ORG).
- *The chance of material heading north, talked earlier this week, may be in jeopardy now since American fuel oil is expected to head transatlantic following outages at two coking units in the U.S.*  
  gold **MISC**, predicted **MISC** (tags: B-MISC).
- *It was the second attack by skinheads in two years on Meyers, an American.*  
  gold **MISC**, predicted **MISC** (tags: B-MISC).
- *Half of dog bites provoked, says American vet.*  
  gold **MISC**, predicted **MISC** (tags: B-MISC).
- *In a stand some analysts linked to controversy over Singapore's caning of an American teenager for vandalism, then-U.S. Trade Representative Mickey Kantor had said the meeting ought to be held where the WTO was going to be headquartered.*  
  gold **MISC**, predicted **MISC** (tags: B-MISC).

A static embedding lookup gives the same vector to the same surface form, while BERT conditions token states on the sentence. The baseline here also uses neighbouring tokens and gold POS features, so it is not context-free. These examples document annotation ambiguity; aggregate F1 alone does not prove which contextual mechanism caused a particular prediction.

## Tokenization / alignment errors

WordPiece expands the test set to 1.3667 tokens per word and splits 17.85% of all words. With Strategy A only the first subword carries a label; the continuation pieces are trained with `ignore_index=-100`. Word-level decoding uses the first piece, while all unmasked pieces still participate in attention. Gold entity boundaries are at word boundaries, not inside a word. `training_environment.json` / `transformer_main_metrics.json` record the exact number of split words and truncated sentences for this run.
