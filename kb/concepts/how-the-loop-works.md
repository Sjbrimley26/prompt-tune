---
type: "concept"
title: "How the prompt-tune loop works"
description: "A local model answers on private payees, code turns the answers into counts, a cloud model rewrites the prompt from the counts; repeat, then score once on held-out keys."
tags: ["prompt-tune", "privacy", "local-llm", "cloud", "categorize"]
resource: "tune.py"
generated: { by: "hermes-agent/claude-opus-5-5", at: "2026-11-10T23:07:00+00:00" }
sources:
  - { id: "tune", resource: "tune.py", title: "loop driver" }
  - { id: "gateway", resource: "gateway.py", title: "the trust boundary" }
  - { id: "categorize", resource: "~/bin/categorize.py (author's machine; not in this repo)", title: "the real classifier (ask(), INSTRUCTIONS hook)" }
---
# How the prompt-tune loop works

**The shape.** Two models that never talk directly, joined by code:

    local model (vision-text) --answers, private file--> gateway.py --counts only--> cloud model (glm-5.3)
         ^                                                                                  |
         +------------------------- new instruction text ----------------------------------+

1. **Local step.** `tune.py` sets the candidate instructions on `categorize.py` and calls its real `ask()` for
   every key in the *tune* split. That is the same request production makes (json_schema enum, temperature 0),
   so a prompt that wins here wins in `categorize.py` too. Answers go to
   `~/.config/<categorizer-config>/prompt-tune/answers-*.json` (the author's machine): private, never read by an agent.
2. **Gateway.** `gateway.py` turns the answers into a fixed report: totals, score, outcomes per gold category,
   the top (gold -> predicted) confusions, outcomes by source (hand / rule). Every string in it must be a
   category name or a field name, or it raises. Payee text is not even an input.
3. **Cloud step.** The cloud model gets a fixed brief, the category names, and every earlier round (its own
   prompt, its one-line hypothesis, the report). It answers with a hypothesis and new instructions.
4. **Accept or reject.** A candidate is kept only if it beats the best score so far. The next round builds on the
   best, not on the last.
5. **Holdout.** After the last round the baseline and the winner are scored once on the *holdout* keys, which the
   cloud never saw. A paired sign test says whether the gain is more than noise.

**Score = right - 2 x wrong; UNSURE = 0.** A blank costs you one manual fill; a wrong label can slip past review,
so it counts double. `--wrong-penalty` changes the ratio. This is the single most important design choice: it
tells the cloud what "better" means, and a different number produces a different prompt.

**Why a split.** About 130 tune keys and 90 holdout keys. With five or six rounds the cloud can overfit the tune
keys (it is choosing among prompts by their tune score), so only the holdout number counts as a result.
Rule-sourced keys are easy (your keyword rules already cover them); the `hand` rows are the hard tail the model
meets in production, so read the `hand` lines first.

**Approval.** `tune.py run` refuses to start unless `gateway.py`'s sha256 equals `approved-gateway.sha256`. You
create that file by running `baseline`, reading the whole report it prints (the exact shape the cloud sees),
then `approve-gateway`. Any edit to gateway.py changes the hash and blocks the loop until you approve again.

**GPU.** The run takes the relay GPU lease (`GpuSwap.window`), loads the seat once, and restores `aux` and the
Hermes aux lane at the end, even on a crash. If someone takes the card mid-run it stops with "evicted";
answers so far are cached, so a rerun resumes.

**Cost.** Each cloud call is about 1-4k prompt tokens (it grows with history) and up to ~7k completion tokens,
because glm-5.3 thinks before answering (measured: one call, 34 s, 965 in / 6,788 out). Six rounds is a few
minutes of cloud time. Local time per round is one pass over ~130 keys: about 3-4 min on vision-text at the arena's measured ~1.6 s per key (29 s for 18 keys). A 6-round run is roughly 30-40 min including the holdout.
