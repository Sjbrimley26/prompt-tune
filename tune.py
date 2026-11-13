#!/usr/bin/env python3
"""Prompt-tune loop: a CLOUD model rewrites the instructions of a LOCAL model that categorizes private payees,
seeing only gateway.py's aggregate report. Nobody carries output between the two.

    local model  --answers (private file)-->  gateway.py  --counts only-->  cloud model  --new prompt-->  local model

Commands (default seat vision-text = gemma4:e4b, categorize.py's default):
  tune.py baseline [--seat S]         score categorize.py's current prompt on the tune split and PRINT THE
                                      GATEWAY REPORT. This is exactly what the cloud will see each round:
                                      read it before approving.
  tune.py approve-gateway             record gateway.py's sha256 as approved (you run this, once)
  tune.py run [--rounds 6] [--seat S] [--cloud-model glm-5.3]
                                      the loop; refuses to start unless gateway.py matches the approved hash.
                                      Ends by scoring the baseline and the best prompt ONCE on the held-out keys.
  tune.py show RUN_ID                 print the run log (prompts, hypotheses, reports): the training material
  tune.py selftest                    gateway selftest (the full suite: python3 -I -m unittest discover -s tests)

Data: gold keys from the author's categorizer (its config dir holds arena/keys.json, "direction|cleaned payee" ->
gold category + source). Keys are split by a hash of the key into tune / holdout (--holdout-pct, default 40),
so the split is stable across runs and the cloud never sees a holdout score until the end.
Private files (answers, answer cache) live in the author's categorizer config dir, under prompt-tune/. The run log in
runs/<id>/ holds only prompts written by the cloud and gateway reports, so it is safe to read anywhere.
Takes the GPU lease (relay GpuSwap window) for the whole run and restores aux at the end.
"""
import argparse
import hashlib
import json
import math
import os
import pwd
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
REAL_HOME = pwd.getpwuid(os.getuid()).pw_dir
BIN = os.environ.get("CATEGORIZE_BIN", os.path.join(REAL_HOME, "bin"))
RELAY = os.environ.get("RELAY_ROOT", os.path.join(REAL_HOME, "chunk", "ai", "relay"))
sys.path.insert(0, HERE)
import gateway as G  # noqa: E402
sys.path.insert(0, BIN)
import categorize as C  # noqa: E402

MAX_CHARS = 1500
CLOUD_URL = "https://ollama.com/v1"
CLOUD_KEY_ENV = "OLLAMA_API_KEY"
MAX_COMPLETION = 32000                 # reasoning models spend most of this before the visible reply

BRIEF = """GOAL
Improve the instruction text of a small LOCAL model (4-8B parameters, on a 4 GB GPU) that assigns household bank
transactions to budget categories. You write the instructions; the owner's local machine scores them on real
transactions you will never see.

WHAT THE LOCAL MODEL SEES, PER CALL
  system: <your instruction text>
          Categories: <the category list, appended automatically>
  user:   Transaction (money out): <cleaned merchant text, at most 4 words; store numbers and boilerplate removed>
Its reply is forced by a JSON schema to {{"category": <one listed category or "UNSURE">}}, temperature 0. It sees no
amount, date, account or history. It cannot look anything up.

HOW A PROMPT IS SCORED
score = right - {penalty:g} x wrong. UNSURE scores 0: the owner fills blanks by hand, and a wrong answer is worse
than a blank because it can slip past review. So a prompt that makes the model abstain on merchants it cannot
place can beat one that answers everything.

WHAT YOU CAN SEE
Only aggregate counts from a fixed-shape report: totals, per-gold-category outcomes, the most common
(gold -> predicted) confusions, and outcomes by source ("hand" = payees the owner categorized by hand, the hard
tail the model meets in production; "rule" = payees the owner's keyword rules already cover). "OTHER" means
a label outside the list. You will never see a merchant name from the data; do not ask for one.

RULES FOR YOUR CANDIDATE
- Plain text, at most {maxc} characters. Do not repeat the category list; it is appended for you.
- Change one or two ideas per round, so the score change can be attributed to them. Build on the best prompt
  so far, not on a rejected one.
- Prefer short concrete definitions of categories that get confused, and a clear rule for when to answer
  UNSURE, over general advice such as "be careful" or "think step by step" (the model cannot show work).
- One sentence about the reply format is enough; the schema enforces it.

REPLY
JSON only, no code fence:
{{"hypothesis": "<one sentence: what you changed and which confusion or failure it targets>",
 "instructions": "<the complete new instruction text>"}}"""


# ---------------------------------------------------------------------------------------------- data
def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def private_dir():
    d = os.path.join(C.CFG, "prompt-tune")
    os.makedirs(d, exist_ok=True)
    return d


def split_of(key, holdout_pct):
    return "holdout" if int(sha(key)[:8], 16) % 100 < holdout_pct else "tune"


def load_keys(path, holdout_pct):
    keys = json.load(open(path, encoding="utf-8"))
    out = {"tune": [], "holdout": []}
    for k in sorted(keys):
        out[split_of(k, holdout_pct)].append((k, keys[k]["gold"], keys[k].get("src", "other")))
    return out


def jload(p, default):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else default


def jsave(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(tmp, p)


# ---------------------------------------------------------------------------------------------- local side
class Evicted(Exception):
    pass


def evaluate(instructions, keys, seat, url, tag, window=None):
    """Run categorize.ask (the real classifier code) with `instructions` over keys. Answers go to a PRIVATE file;
    the caller gets back only the gateway report."""
    C.INSTRUCTIONS = instructions
    cats = C.categories()
    cache_p = os.path.join(private_dir(), "answer-cache.json")
    cache = jload(cache_p, {})
    ph = sha(instructions)[:16]
    rows, t0, f0, asked = [], time.time(), sum(C.FAILS.values()), 0
    for key, gold, src in keys:
        ck = "%s|%s|%s" % (seat, ph, key)
        if ck not in cache:
            if window is not None and asked % 20 == 0 and not window.seat_loaded(seat):
                jsave(cache_p, cache)
                raise Evicted("seat %s is no longer loaded (someone took the card)" % seat)
            direction, text = key.split("|", 1)
            cache[ck] = C.ask(url, seat, cats, text, direction)   # ServerError propagates: abort, cache kept
            asked += 1
            if asked % 25 == 0:
                jsave(cache_p, cache)
        rows.append({"key": key, "gold": gold, "pred": cache[ck], "src": src})
    jsave(cache_p, cache)
    secs, fails = time.time() - t0, sum(C.FAILS.values()) - f0
    jsave(os.path.join(private_dir(), "answers-%s.json" % tag), {"rows": rows, "seconds": secs, "failures": fails})
    return rows, G.report(rows, cats, secs, fails, PENALTY)


# ---------------------------------------------------------------------------------------------- cloud side
def cloud_key():
    k = os.environ.get("PROMPT_TUNE_API_KEY")
    if k:
        return k
    sys.path.insert(0, RELAY)
    from relay.env import get_secret
    k = get_secret(CLOUD_KEY_ENV)
    if not k:
        sys.exit("%s is not set (env or ~/.hermes/.env)" % CLOUD_KEY_ENV)
    return k


def cloud_chat(messages, model, url, key, log):
    body = {"model": model, "messages": messages, "max_tokens": MAX_COMPLETION, "temperature": 0.5, "stream": False}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    log.append(body)                        # every outgoing body is kept for the leak test and the run log
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    ch = out["choices"][0]
    m = ch.get("message") or {}
    info = {"finish_reason": ch.get("finish_reason"), "content_chars": len(m.get("content") or ""),
            "reasoning_chars": len(m.get("reasoning_content") or m.get("reasoning") or ""),
            "content": m.get("content"), "reasoning": m.get("reasoning_content") or m.get("reasoning")}
    return (m.get("content") or ""), out.get("usage", {}), info


def parse_candidate(text):
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        d = json.loads(text[s:e + 1])
    except ValueError:
        return None
    if not isinstance(d, dict) or not isinstance(d.get("instructions"), str) or not d["instructions"].strip():
        return None
    return {"hypothesis": str(d.get("hypothesis", "")).strip()[:400], "instructions": d["instructions"].strip()}


def build_messages(history, cats, best_round):
    """The cloud's whole view: fixed brief + category names + its own earlier prompts + verified reports."""
    hist = []
    for h in history:
        G.verify(h["report"], cats)          # re-check right before anything leaves the box
        hist.append({"round": h["round"], "hypothesis": h["hypothesis"], "instructions": h["instructions"],
                     "accepted": h["accepted"], "report": h["report"]})
    user = ("CATEGORIES\n%s\n\nHISTORY (round 0 = the current production prompt; best so far = round %d)\n%s\n\n"
            "Write the next candidate." % ("; ".join(cats), best_round, json.dumps(hist, indent=1)))
    return [{"role": "system", "content": BRIEF.format(penalty=PENALTY, maxc=MAX_CHARS)},
            {"role": "user", "content": user}]


# ---------------------------------------------------------------------------------------------- gpu
class _NoWindow:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def gpu_window(seat, url, no_lease):
    if no_lease:
        return _NoWindow()
    sys.path.insert(0, RELAY)
    from relay.gpuswap import GpuSwap
    host = url[:-3] if url.rstrip("/").endswith("/v1") else url
    sw = GpuSwap(host.rstrip("/"), "", owner="prompt-tune", purpose="prompt-tune on %s" % seat)

    class W:
        def __enter__(self):
            self.cm = sw.window(["aux"])
            w = self.cm.__enter__()
            w.switch(seat)
            return w

        def __exit__(self, *a):
            return self.cm.__exit__(*a)
    return W()


# ---------------------------------------------------------------------------------------------- stats
def sign_test(a, b):
    """Two-sided exact sign test on discordant pairs (a = only A right, b = only B right)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def line(name, r):
    return "%-10s n=%3d  right %3d  wrong %3d  unsure %3d  score %6.1f  failures %d  %.0f s" % (
        name, r["n"], r["right"], r["wrong"], r["unsure"], r["score"], r["failures"], r["seconds"])


# ---------------------------------------------------------------------------------------------- commands
def cmd_baseline(a):
    ks = load_keys(a.keys, a.holdout_pct)
    print("keys: tune %d, holdout %d (holdout not touched)" % (len(ks["tune"]), len(ks["holdout"])))
    with gpu_window(a.seat, a.url, a.no_lease) as w:
        _, rep = evaluate(C.DEFAULT_INSTRUCTIONS, ks["tune"], a.seat, a.url, "baseline-tune", w)
    print("gateway.py sha256 %s" % G.sha256())
    print("--- this report is what the cloud model sees each round (plus prompts it wrote itself) ---")
    print(json.dumps(rep, indent=1))
    print("--- if nothing above is private, run: tune.py approve-gateway")


def cmd_approve(a):
    with open(a.approved, "w") as f:
        f.write(G.sha256() + "\n")
    print("approved gateway.py sha256 %s" % G.sha256())


def cmd_run(a):
    approved = open(a.approved).read().strip() if os.path.exists(a.approved) else ""
    if approved != G.sha256():
        sys.exit("gateway.py is not approved at its current hash. Run 'tune.py baseline', read the whole report, "
                 "then 'tune.py approve-gateway'.")
    ks = load_keys(a.keys, a.holdout_pct)
    cats = C.categories()
    rid = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    rdir = os.path.join(a.runs_dir, rid)
    os.makedirs(rdir)
    logp = os.path.join(rdir, "rounds.jsonl")
    key = cloud_key()
    sent, usage = [], {"prompt_tokens": 0, "completion_tokens": 0}
    meta = {"run": rid, "seat": a.seat, "cloud_model": a.cloud_model, "rounds": a.rounds, "penalty": PENALTY,
            "gateway_sha256": G.sha256(), "tune_keys": len(ks["tune"]), "holdout_keys": len(ks["holdout"])}
    print("run %s: seat %s, cloud %s, tune %d keys, holdout %d" % (rid, a.seat, a.cloud_model,
                                                                   len(ks["tune"]), len(ks["holdout"])))

    def log(rec):
        with open(logp, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    with gpu_window(a.seat, a.url, a.no_lease) as w:
        _, rep = evaluate(C.DEFAULT_INSTRUCTIONS, ks["tune"], a.seat, a.url, rid + "-r0", w)
        history = [{"round": 0, "hypothesis": "production prompt", "instructions": C.DEFAULT_INSTRUCTIONS,
                    "accepted": True, "report": rep}]
        best = 0
        log({**meta, **history[0]})
        print(line("round 0", rep))
        for rnd in range(1, a.rounds + 1):
            cand = None
            for _ in range(2):                                    # one retry on an unusable reply
                text, u, info = cloud_chat(build_messages(history, cats, best), a.cloud_model, a.cloud_url, key, sent)
                for k in usage:
                    usage[k] += int(u.get(k) or 0)
                cand = parse_candidate(text)
                if cand:
                    break
                log({"round": rnd, "unusable_reply": info, "usage": u})
                print("round %d: unusable reply (finish=%s, content %d chars, reasoning %d chars, %s completion tokens)"
                      % (rnd, info["finish_reason"], info["content_chars"], info["reasoning_chars"],
                         u.get("completion_tokens")))
            if not cand:
                print("round %d: cloud reply unusable twice; stopping" % rnd)
                break
            rec = {"round": rnd, **cand, "chars": len(cand["instructions"])}
            if len(cand["instructions"]) > MAX_CHARS:
                rec.update(accepted=False, report=None, rejected="longer than %d chars" % MAX_CHARS)
                log(rec)
                print("round %d: rejected, %d chars" % (rnd, rec["chars"]))
                history.append({**rec, "instructions": "(rejected: %d chars, over the limit)" % rec["chars"],
                                "report": {"n": 0}})
                continue
            _, rep = evaluate(cand["instructions"], ks["tune"], a.seat, a.url, "%s-r%d" % (rid, rnd), w)
            rec["report"] = rep
            rec["accepted"] = rep["score"] > history[best]["report"]["score"]
            history.append(rec)
            if rec["accepted"]:
                best = len(history) - 1
            rec["best_round"] = history[best]["round"]
            log(rec)
            print(line("round %d" % rnd, rep) + ("  ACCEPTED" if rec["accepted"] else "") + "  | " + cand["hypothesis"])
        # the holdout is scored once, at the end, for the baseline and the winner only
        bi = history[best]["instructions"]
        rows0, h0 = evaluate(C.DEFAULT_INSTRUCTIONS, ks["holdout"], a.seat, a.url, rid + "-holdout-baseline", w)
        rows1, h1 = (rows0, h0) if best == 0 else \
            evaluate(bi, ks["holdout"], a.seat, a.url, rid + "-holdout-best", w)
    pr = G.paired(rows0, rows1, cats)
    hand0 = G.report([r for r in rows0 if r["src"] == "hand"], cats, 0, 0, PENALTY)
    hand1 = G.report([r for r in rows1 if r["src"] == "hand"], cats, 0, 0, PENALTY)
    p = sign_test(pr["only_a_right"], pr["only_b_right"])
    with open(os.path.join(rdir, "best.txt"), "w", encoding="utf-8") as f:
        f.write(bi + "\n")
    final = {"final": True, "best_round": history[best]["round"], "holdout_baseline": h0, "holdout_best": h1,
             "holdout_hand_baseline": hand0, "holdout_hand_best": hand1, "paired": pr, "sign_test_p": p,
             "cloud_usage": usage, "cloud_calls": len(sent)}
    log(final)
    print("\nHOLDOUT (scored once; the cloud never saw these keys' results)")
    print(line("baseline", h0))
    print(line("best r%d" % history[best]["round"], h1))
    print(line("hand base", hand0))
    print(line("hand best", hand1))
    print("paired: both right %d, only baseline right %d, only best right %d, neither %d; sign test p = %.2f"
          % (pr["both_right"], pr["only_a_right"], pr["only_b_right"], pr["neither_right"], p))
    if p > 0.05:
        print("p > 0.05: the holdout does not show a real difference; treat the gain as noise until another month.")
    print("cloud: %d calls, %d prompt + %d completion tokens" % (len(sent), usage["prompt_tokens"],
                                                               usage["completion_tokens"]))
    print("log: %s\nbest prompt: %s\nto use it: categorize.py in.csv out.csv --instructions %s"
          % (logp, os.path.join(rdir, "best.txt"), os.path.join(rdir, "best.txt")))
    return final


def cmd_show(a):
    for l in open(os.path.join(a.runs_dir, a.run_id, "rounds.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if r.get("final"):
            print("\nFINAL best round %d | holdout baseline score %s, best %s | sign test p %.2f"
                  % (r["best_round"], r["holdout_baseline"]["score"], r["holdout_best"]["score"], r["sign_test_p"]))
            continue
        rep = r.get("report") or {}
        print("\n=== round %d %s| score %s (right %s, wrong %s, unsure %s)\nhypothesis: %s\n%s"
              % (r["round"], "ACCEPTED " if r.get("accepted") and r["round"] else "", rep.get("score"),
                 rep.get("right"), rep.get("wrong"), rep.get("unsure"), r.get("hypothesis"), r["instructions"]))


PENALTY = 2.0


def main(argv=None):
    global PENALTY, MAX_CHARS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["baseline", "approve-gateway", "run", "show", "selftest"])
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--seat", default="vision-text")
    # BASE_URL wins over the default so a run works against any OpenAI-compatible server without a flag
    ap.add_argument("--url", default=os.environ.get("BASE_URL", "http://localhost:8080/v1"))
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--cloud-model", default="glm-5.3")
    ap.add_argument("--cloud-url", default=CLOUD_URL)
    ap.add_argument("--keys", default=None, help="gold keys (default: arena keys.json)")
    ap.add_argument("--holdout-pct", type=int, default=40)
    ap.add_argument("--wrong-penalty", type=float, default=2.0)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--cfg", default=None, help="categorize config dir (tests)")
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--approved", default=os.path.join(HERE, "approved-gateway.sha256"))
    ap.add_argument("--no-lease", action="store_true", help="do not take the GPU lease (tests, fake servers)")
    a = ap.parse_args(argv)
    if a.cfg:
        C.CFG = a.cfg
    a.keys = a.keys or os.path.join(C.CFG, "arena", "keys.json")
    PENALTY, MAX_CHARS = a.wrong_penalty, a.max_chars
    if a.cmd == "selftest":
        return 0 if G.selftest() else 1
    if a.cmd == "show":
        if not a.run_id:
            sys.exit("show RUN_ID")
        return cmd_show(a)
    return {"baseline": cmd_baseline, "approve-gateway": cmd_approve, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    try:
        r = main()
    except C.ServerError as e:
        sys.exit("ABORTED (answers so far are cached; rerun to resume): %s" % e)
    except Evicted as e:
        sys.exit("ABORTED: %s" % e)
    sys.exit(r if isinstance(r, int) else 0)
