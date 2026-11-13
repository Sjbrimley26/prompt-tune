# prompt-tune

A cloud model improves the prompt of a local model that categorizes your private bank payees, without seeing any
payee. Explainers: kb/concepts/ (start with how-the-loop-works.md).

First real run (GPU back in fedora; takes the GPU lease, restores aux):

    cd ~/chunk/code/prompt-tune
    python3 -I tune.py baseline            # prints the exact report the cloud will see: read all of it
    python3 -I tune.py approve-gateway     # only if nothing in it is private
    python3 -I tune.py run --rounds 6      # ~30-40 min (about 130 tune keys x 7 passes plus the holdout, ~1.6 s per key on vision-text); prints round scores, then the holdout result
    python3 -I tune.py show <run-id>       # the prompts and what each changed

Use a winner:  python3 -I ~/bin/categorize.py in.csv out.csv --instructions runs/<id>/best.txt --fresh
Tests:         python3 -I -m unittest discover -s tests

The local seat answers through any OpenAI-compatible server: `--url` sets the root, and the `BASE_URL`
environment variable wins over the built-in default (`http://localhost:8080/v1`).

## First real runs (2026-11-12)

Same local seat (a 4 B model on a 4 GB GPU), 6 rounds each, the last ~40% of the labelled keys (91) held out
and scored once. Cloud writers ran on Ollama Cloud; token counts are the writer's own usage.

| writer | tune 34 -> | holdout 11 -> | reading |
|---|---|---|---|
| glm-5.3-flash | 44 | 26 | wrong 20 -> 12, right flat (51 -> 50); sign test p = 1.00 |
| deepseek-v4.1-flash | 40 | 20 | wrong 20 -> 14, but the local seat produced no answer on 33 of 91 keys (vs 3); p = 0.55 |

Honest reading: neither gain is established — both tuned prompts score higher but **abstain more**, so the
score moved without the model getting more answers right. The transferable finding is the failure pattern:
deepseek's best prompt is longer, and the 4 B seat failed or mis-parsed on 33 of 91 holdout keys with it
(against 3 with glm's) — per-round failures grew with prompt length. Longer prompts are not free on a small
model even when the text is better.

The full aggregate table, the abort of a first arm on a too-small completion cap, and caveats sit in the
lab-excerpt of the sibling repo: [small-gpu-bench, `docs/excerpts/prompt-tune.md`](https://github.com/Sjbrimley26/small-gpu-bench/blob/master/docs/excerpts/prompt-tune.md).
Run records stay out of this repo (`runs/` is git-ignored); the same loop with a fake local model, a fake
cloud and synthetic payees is covered by the tests, including the leak assertions.

## License

MIT; see `LICENSE`.
