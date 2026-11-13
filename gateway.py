#!/usr/bin/env python3
"""The trust boundary of the prompt-tune loop.

This is the ONLY code that turns private per-payee results into something a cloud model (or a cloud-model
agent reading stdout) may see. It reads the answers file that tune.py writes into the private directory and
returns a fixed-shape report built from integers and whitelisted strings only:

    category names (from categories.txt), "UNSURE", "OTHER", source labels ("hand", "rule", "other"),
    and this file's own field names.

Any label that is not in that vocabulary becomes "OTHER" before it is counted, and verify() walks the
finished report and raises LeakError on any string outside the vocabulary. So even if a prompt somehow made
the local model echo a payee, the payee could not reach the output. Payee text is never an input here:
answer rows carry only gold, pred and src.

Approve this file once (read it, run it on the real answers, read the whole output). Each run records this
file's sha256, so a later change shows up as a different hash and needs a new approval.

Usage:
    gateway.py ANSWERS.json CATEGORIES.txt     print the report for one answers file
    gateway.py --selftest                      prove the whitelist turns planted text into OTHER and raises
"""
import hashlib
import json
import os
import sys

SOURCES = ("hand", "rule", "other")
FIELDS = {"n", "right", "wrong", "unsure", "failures", "seconds", "score", "wrong_penalty", "by_source",
          "by_gold", "gold", "pred", "confusions", "both_right", "only_a_right", "only_b_right",
          "neither_right", "a", "b", "paired"}
SPECIAL = {"UNSURE", "OTHER"}
TOP_CONFUSIONS = 25


class LeakError(Exception):
    """A string outside the whitelist reached a report. The caller must stop and send nothing."""


def sha256():
    with open(os.path.realpath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_categories(path):
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")]


def _label(x, cats):
    return x if (x in cats or x == "UNSURE") else "OTHER"


def _outcome(gold, pred):
    if pred == "UNSURE":
        return "unsure"
    return "right" if pred == gold else "wrong"


def verify(obj, vocab):
    """Raise LeakError unless every string in obj (keys and values) is in vocab or a field name."""
    allowed = set(vocab) | SPECIAL | set(SOURCES) | FIELDS
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in allowed:
                raise LeakError("key outside the whitelist")
            verify(v, vocab)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            verify(v, vocab)
    elif isinstance(obj, str):
        if obj not in allowed:
            raise LeakError("string outside the whitelist")
    elif isinstance(obj, bool) or not isinstance(obj, (int, float)) and obj is not None:
        raise LeakError("value of type %s is not allowed" % type(obj).__name__)
    return obj


def report(rows, cats, seconds=0.0, failures=0, wrong_penalty=2.0):
    """rows: [{"gold": str, "pred": str, "src": str}, ...] -> the fixed-shape report (already verified)."""
    cats = list(cats)
    tot = {"right": 0, "wrong": 0, "unsure": 0}
    by_src = {s: {"right": 0, "wrong": 0, "unsure": 0} for s in SOURCES}
    by_gold = {}
    conf = {}
    for r in rows:
        g, p = _label(r.get("gold"), cats), _label(r.get("pred"), cats)
        s = r.get("src") if r.get("src") in SOURCES else "other"
        o = _outcome(g, p)
        tot[o] += 1
        by_src[s][o] += 1
        bg = by_gold.setdefault(g, {"right": 0, "wrong": 0, "unsure": 0})
        bg[o] += 1
        if o == "wrong":
            conf[(g, p)] = conf.get((g, p), 0) + 1
    out = {
        "n": len(rows), **tot,
        "score": round(tot["right"] - wrong_penalty * tot["wrong"], 2),
        "wrong_penalty": float(wrong_penalty),
        "failures": int(failures), "seconds": round(float(seconds), 1),
        "by_source": {s: v for s, v in by_src.items() if sum(v.values())},
        "by_gold": [{"gold": g, "n": sum(v.values()), **v}
                    for g, v in sorted(by_gold.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))],
        "confusions": [{"gold": g, "pred": p, "n": n}
                       for (g, p), n in sorted(conf.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_CONFUSIONS]],
    }
    return verify(out, cats)


def paired(rows_a, rows_b, cats):
    """Two answer lists over the SAME keys, in the same order -> counts of who got which key right."""
    if len(rows_a) != len(rows_b):
        raise ValueError("answer lists differ in length")
    c = {"both_right": 0, "only_a_right": 0, "only_b_right": 0, "neither_right": 0}
    for a, b in zip(rows_a, rows_b):
        if a.get("gold") != b.get("gold"):
            raise ValueError("answer lists are not over the same keys")
        ra, rb = a.get("pred") == a.get("gold"), b.get("pred") == b.get("gold")
        c["both_right" if ra and rb else "only_a_right" if ra else "only_b_right" if rb else "neither_right"] += 1
    return verify({"n": len(rows_a), **c}, cats)


def selftest():
    cats = ["Groceries", "Gas", "Medical"]
    rows = [{"gold": "Groceries", "pred": "Groceries", "src": "hand"},
            {"gold": "Medical", "pred": "Groceries", "src": "rule"},
            {"gold": "Gas", "pred": "UNSURE", "src": "hand"},
            {"gold": "Gas", "pred": "KROGER #123 SPRINGFIELD", "src": "hand"},     # a payee echoed by the model
            {"gold": "ACME DENTAL", "pred": "Medical", "src": "SECRET-SRC"}]         # bad gold and src
    rep = report(rows, cats, seconds=1.23, failures=1)
    txt = json.dumps(rep)
    checks = [
        ("planted payee text absent", "KROGER" not in txt and "ACME" not in txt and "SECRET" not in txt),
        ("echoed payee counted as OTHER", {"gold": "Gas", "pred": "OTHER", "n": 1} in rep["confusions"]),
        ("totals", (rep["n"], rep["right"], rep["wrong"], rep["unsure"]) == (5, 1, 3, 1)),
        ("score = right - 2*wrong", rep["score"] == 1 - 2 * 3),
        ("bad src -> other", rep["by_source"].get("other") == {"right": 0, "wrong": 1, "unsure": 0}),
    ]
    for name, bad in (("verify rejects a payee string", {"gold": "KROGER"}),
                      ("verify rejects an unknown key", {"KROGER": 1}),
                      ("verify rejects a nested string", {"by_gold": [{"gold": "x y z"}]})):
        try:
            verify(bad, cats)
            checks.append((name, False))
        except LeakError:
            checks.append((name, True))
    p = paired(rows[:3], [dict(rows[0], pred="UNSURE"), dict(rows[1], pred="Medical"), rows[2]], cats)
    checks.append(("paired counts", (p["only_a_right"], p["only_b_right"], p["neither_right"]) == (1, 1, 1)))
    ok = all(v for _, v in checks)
    for name, v in checks:
        print("%s  %s" % ("PASS" if v else "FAIL", name))
    print("gateway selftest:", "OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(0 if selftest() else 1)
    if len(sys.argv) != 3 or sys.argv[1].startswith("--"):
        sys.exit(__doc__)
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    print(json.dumps(report(data["rows"], load_categories(sys.argv[2]), data.get("seconds", 0),
                            data.get("failures", 0), data.get("wrong_penalty", 2.0)), indent=1))
