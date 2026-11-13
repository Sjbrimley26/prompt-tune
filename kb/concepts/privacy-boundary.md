---
type: "concept"
title: "The privacy boundary: what can and cannot reach the cloud"
description: "Exactly what the cloud model receives, why payee text cannot get through, and what the approval covers."
tags: ["prompt-tune", "privacy", "gateway", "approval"]
resource: "gateway.py"
generated: { by: "hermes-agent/claude-opus-5-5", at: "2026-11-10T23:07:00+00:00" }
sources:
  - { id: "gateway", resource: "gateway.py", title: "report() and verify()" }
  - { id: "tests", resource: "tests/test_tune.py", title: "leak test over every cloud request body" }
  - { id: "policy", resource: "(sibling private repo: reconcile-tools)", title: "reconcile-tools approval rule" }
---
# The privacy boundary

**What the cloud receives, all of it:** the fixed brief in `tune.py` (`BRIEF`), your category names (you approved
sending these on 2026-11-10), its own earlier prompts and hypotheses, and gateway reports. A report contains only
integers, category names, `UNSURE`, `OTHER`, the source labels `hand` / `rule` / `other`, and field names.

**Why payee text cannot get through.**

1. Payee text never enters the gateway. Answer rows passed to `report()` carry `gold`, `pred` and `src`; the
   key is in the private answers file but is not read by `report()`.
2. A predicted label that is not a category (a model echoing a payee, say) becomes `OTHER` before counting.
3. `verify()` walks the whole report, keys and values, and raises `LeakError` on any string outside the
   whitelist. `build_messages()` runs it again on every report right before a request is sent.
4. The tests plant payee-shaped strings, run the full loop against a fake cloud, and assert that no request
   body and no run-log line contains any of them.

**What can still leak, honestly.** Counts can carry a little information: "1 key with gold Medical predicted
OTHER" tells the cloud that such a key exists. With your category names already approved this reveals nothing
about who you pay. The cloud's *own* prompts are logged and sent back; it never had private text to put in them.

**What the approval covers.** The one-time approval is for `gateway.py` at a given sha256, not for `tune.py`.
That is deliberate: the loop driver can change (new rounds logic, a new cloud model) without re-approval, because
everything it sends to the cloud passes through `gateway.py`'s `verify()`. A change to `BRIEF` changes what
the cloud is *told*, not what private data it sees. If you want the brief covered too, say so and the hash check
can include it.

**Relation to the reconcile-tools rule.** There, every new script needs your vet because its stdout is the
boundary. Here the boundary is one small file that does not change between runs, which is what lets the two
models loop without you in the middle.
