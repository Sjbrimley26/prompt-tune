"""End-to-end test of the prompt-tune loop on SYNTHETIC payees, with a fake local model and a fake cloud.

Run: python3 -I -m unittest discover -s tests   (from ~/chunk/code/prompt-tune)

The fake local model is prompt-sensitive on purpose: it calls restaurants "Groceries" unless the system
prompt defines Dining Out, and answers UNSURE on cryptic payees only if the prompt says to abstain on
unknown merchants. So a better prompt really scores better, and the loop has something to find.
The fake cloud records every request body; the leak test asserts no synthetic payee string is in any of them.
"""
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
import gateway as G  # noqa: E402
import tune as T  # noqa: E402

CATS = ["Groceries", "Dining Out", "Gas", "Medical", "Utilities"]
# cleaned payee text (as categorize sends it) -> (gold, kind)
PAYEES = {}
for i in range(12):
    PAYEES["ZQ BISTRO %d" % i] = ("Dining Out", "restaurant")
    PAYEES["ZQ MARKET %d" % i] = ("Groceries", "grocer")
    PAYEES["ZQ FUEL %d" % i] = ("Gas", "fuel")
    PAYEES["XKCDQ %d" % i] = ("Medical", "cryptic")     # unguessable: the right move is UNSURE


def fake_local_answer(system, text):
    kind = PAYEES[text][1]
    if kind == "restaurant":
        return "Dining Out" if "restaurant" in system.lower() else "Groceries"
    if kind == "grocer":
        return "Groceries"
    if kind == "fuel":
        return "Gas"
    return "UNSURE" if "unknown merchant" in system.lower() else "Utilities"


class Local(http.server.BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Local.calls += 1
        system = b["messages"][0]["content"]
        text = b["messages"][1]["content"].split(": ", 1)[1]
        out = {"choices": [{"message": {"content": json.dumps({"category": fake_local_answer(system, text)})},
                            "finish_reason": "stop"}]}
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


CANDIDATES = [
    "not json at all",                                                       # unusable -> retry
    json.dumps({"hypothesis": "define Dining Out", "instructions":
                "Categorize household bank transactions. Dining Out means any restaurant, bistro or cafe."}),
    json.dumps({"hypothesis": "too long", "instructions": "x" * 5000}),     # rejected by length
    json.dumps({"hypothesis": "abstain on unknown merchants", "instructions":
                "Categorize household bank transactions. Dining Out means any restaurant, bistro or cafe. "
                "If it is an unknown merchant you cannot place, answer UNSURE."}),
    json.dumps({"hypothesis": "worse: drop the definition", "instructions": "Categorize transactions."}),
]


class Cloud(http.server.BaseHTTPRequestHandler):
    bodies, auth, i = [], [], 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        raw_in = self.rfile.read(int(self.headers["Content-Length"]))
        Cloud.bodies.append(raw_in.decode())
        Cloud.auth.append(self.headers.get("Authorization"))
        content = CANDIDATES[min(Cloud.i, len(CANDIDATES) - 1)]
        Cloud.i += 1
        raw = json.dumps({"choices": [{"message": {"content": content}}],
                          "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve(handler):
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


class LoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="prompt-tune-test-")
        self.cfg = os.path.join(self.tmp, "cfg")
        os.makedirs(os.path.join(self.cfg, "arena"))
        with open(os.path.join(self.cfg, "categories.txt"), "w") as f:
            f.write("\n".join(CATS) + "\n")
        keys = {"money out|" + p: {"gold": g, "src": "hand" if i % 3 else "rule"}
                for i, (p, (g, _)) in enumerate(sorted(PAYEES.items()))}
        with open(os.path.join(self.cfg, "arena", "keys.json"), "w") as f:
            json.dump(keys, f)
        self.local, self.cloud = serve(Local), serve(Cloud)
        Cloud.bodies, Cloud.auth, Cloud.i, Local.calls = [], [], 0, 0
        os.environ["PROMPT_TUNE_API_KEY"] = "test-key"
        self.common = ["--cfg", self.cfg, "--no-lease", "--seat", "fake",
                       "--url", "http://127.0.0.1:%d/v1" % self.local.server_port,
                       "--cloud-url", "http://127.0.0.1:%d/v1" % self.cloud.server_port,
                       "--runs-dir", os.path.join(self.tmp, "runs"),
                       "--approved", os.path.join(self.tmp, "approved.sha256")]

    def tearDown(self):
        self.local.shutdown()
        self.cloud.shutdown()
        T.C.INSTRUCTIONS = None
        os.environ.pop("PROMPT_TUNE_API_KEY", None)
        shutil.rmtree(self.tmp)

    def test_refuses_without_approval(self):
        with self.assertRaises(SystemExit):
            T.main(["run"] + self.common)
        self.assertEqual(Cloud.bodies, [])

    def test_refuses_after_gateway_changes(self):
        with open(os.path.join(self.tmp, "approved.sha256"), "w") as f:
            f.write("0" * 64 + "\n")
        with self.assertRaises(SystemExit):
            T.main(["run"] + self.common)
        self.assertEqual(Cloud.bodies, [])

    def test_full_loop(self):
        T.main(["approve-gateway"] + self.common)
        final = T.main(["run", "--rounds", "4"] + self.common)

        # 1. nothing private reached the cloud: no payee text, no payee number pattern
        self.assertGreater(len(Cloud.bodies), 0)
        for body in Cloud.bodies:
            for p in PAYEES:
                self.assertNotIn(p, body)
            for marker in ("ZQ ", "XKCDQ", "BISTRO", "MARKET", "FUEL"):
                self.assertNotIn(marker, body)
        self.assertTrue(all(a == "Bearer test-key" for a in Cloud.auth))

        # 2. the loop found the better prompts and kept the best one
        rdir = os.path.join(self.tmp, "runs", os.listdir(os.path.join(self.tmp, "runs"))[0])
        rounds = [json.loads(l) for l in open(os.path.join(rdir, "rounds.jsonl"))]
        by = {r["round"]: r for r in rounds if not r.get("final")}
        self.assertTrue(by[1]["accepted"])                      # Dining Out definition
        self.assertEqual(by[2].get("rejected"), "longer than 1500 chars")
        self.assertTrue(by[3]["accepted"])                      # abstain rule
        self.assertFalse(by[4]["accepted"])                     # the worse prompt
        self.assertGreater(by[3]["report"]["score"], by[1]["report"]["score"])
        self.assertGreater(by[1]["report"]["score"], by[0]["report"]["score"])
        self.assertEqual(final["best_round"], 3)
        self.assertIn("unknown merchant", open(os.path.join(rdir, "best.txt")).read())

        # 3. the holdout was scored only at the end, and the best prompt wins on it
        self.assertGreater(final["holdout_best"]["score"], final["holdout_baseline"]["score"])
        self.assertGreater(final["paired"]["only_b_right"], 0)
        # holdout results never went to the cloud: the last cloud call happened before them
        self.assertNotIn("holdout_best", "".join(Cloud.bodies))

        # 4. the run log holds no payee text either (it is meant to be readable anywhere)
        log = open(os.path.join(rdir, "rounds.jsonl")).read()
        for p in PAYEES:
            self.assertNotIn(p, log)

        # 5. the private answers file DOES hold keys, and lives under the private cfg dir
        priv = os.path.join(self.cfg, "prompt-tune")
        self.assertTrue(any(f.startswith("answers-") for f in os.listdir(priv)))

    def test_cache_skips_repeat_calls(self):
        T.main(["approve-gateway"] + self.common)
        T.main(["baseline"] + self.common)
        n = Local.calls
        T.main(["baseline"] + self.common)
        self.assertEqual(Local.calls, n)

    def test_split_is_stable_and_disjoint(self):
        ks = T.load_keys(os.path.join(self.cfg, "arena", "keys.json"), 40)
        ks2 = T.load_keys(os.path.join(self.cfg, "arena", "keys.json"), 40)
        self.assertEqual(ks, ks2)
        self.assertFalse({k for k, _, _ in ks["tune"]} & {k for k, _, _ in ks["holdout"]})
        self.assertEqual(len(ks["tune"]) + len(ks["holdout"]), len(PAYEES))

    def test_default_prompt_unchanged(self):
        self.assertEqual(T.C.DEFAULT_INSTRUCTIONS,
                         "You categorize bank transactions for a household budget. Reply with JSON only. "
                         "Choose the single best category from the allowed list. If the merchant is cryptic "
                         "or could plausibly be several categories, answer UNSURE rather than guessing.")

    def test_sign_test(self):
        self.assertEqual(T.sign_test(0, 0), 1.0)
        self.assertAlmostEqual(T.sign_test(0, 6), 0.03125)
        self.assertGreater(T.sign_test(3, 5), 0.5)


class GatewayTest(unittest.TestCase):
    def test_selftest(self):
        self.assertTrue(G.selftest())


if __name__ == "__main__":
    unittest.main()
