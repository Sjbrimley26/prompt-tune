---
type: "concept"
title: "Reading a run: what to learn from the prompt diffs"
description: "How to use tune.py show to see which prompt changes moved a small model, and what not to conclude."
tags: ["prompt-tune", "prompting", "small-models"]
resource: "runs"
generated: { by: "hermes-agent/claude-opus-5-5", at: "2026-11-10T23:07:00+00:00" }
sources:
  - { id: "tune", resource: "tune.py", title: "cmd_show and the round log" }
  - { id: "nb99", resource: "(private lab notebook, section 99: GAL - facts in the prompt beat instructions)", title: "GAL: facts in the prompt beat instructions" }
---
# Reading a run

`python3 -I tune.py show <run-id>` prints every round: the hypothesis the cloud wrote, the full instruction text,
and its score. The run log (`runs/<id>/rounds.jsonl`) holds only prompts and gateway reports, so it is safe to
read anywhere, paste into a chat, or hand to a cloud model.

**What to look at.**

- **Accepted rounds next to their confusion lines.** If round 2 added "Dining Out = food prepared for you" and the
  `Dining Out -> Groceries` confusion dropped from 9 to 3, that sentence did the work. That is the lesson to
  carry into your own prompts: a concrete definition of a confusable pair.
- **Unsure going up while wrong goes down.** That is the abstain rule working. Check the right count did not
  fall by more than the wrong count; if it did, the rule made the model timid, not careful.
- **Rejected rounds.** A rejected idea is as informative as an accepted one. "Think step by step" style
  additions usually do nothing here, because the schema forces a one-word answer with no room to reason.
- **Tune vs holdout.** A big tune gain and a flat holdout means the cloud fitted the tune keys. Trust the
  holdout line and the sign test, not the round scores.

**What not to conclude.**

- A gain of 2-3 keys on ~90 holdout keys is noise (the run prints this when p > 0.05).
- A prompt tuned on gemma4 (`vision-text`) is not tuned for bolt or granite. Each seat reads instructions
  differently; run the loop per seat if the arena is the production path.
- Categories that depend on facts only you know (who a payee actually is: initials, one-off local shops) cannot be fixed by
  any prompt. Their confusions will stay; leave them to rules.

**The general rule this demonstrates.** The cloud model cannot see a single payee, yet it can improve the prompt,
because it is told the goal, the exact format the small model sees, how the score is computed, and where the
model is confused. The same four things are what a small model needs from you: the goal, the exact input shape,
how it will be judged, and the known traps.
