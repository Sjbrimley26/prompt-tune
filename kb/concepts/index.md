# concepts

How the loop works, what the cloud sees, and how to read a run.

* [How the prompt-tune loop works](/concepts/how-the-loop-works.md) - A local model answers on private payees, code turns the answers into counts, a cloud model rewrites the prompt from the counts; repeat, then score once on held-out keys.
* [The privacy boundary: what can and cannot reach the cloud](/concepts/privacy-boundary.md) - Exactly what the cloud model receives, why payee text cannot get through, and what the approval covers.
* [Reading a run: what to learn from the prompt diffs](/concepts/reading-a-run.md) - How to use tune.py show to see which prompt changes moved a small model, and what not to conclude.
