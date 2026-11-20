"""Report-metric server: the reward of GRPO on report generation (`report_radcliq`, `mixed_report_vqa`).

Reward = -RadCliQ-v1 of each sampled report against the reference report of its study, with the same RadCliQ-v1 as
the evaluation (report_score.py). The launcher starts this file in the metrics environment ($MEDRES_METRICS_PYTHON)
on the student's first GPU and waits for /health; the reward plugin posts to /score.

    POST /score  {"refs": [...], "hyps": [...]}  ->  {"rewards": [...]}
    GET  /health                                  ->  {"ok": true, "calls": n}   (answers once the metric is loaded)

Listens on 127.0.0.1 only and logs counts only (the reports it scores may be credentialed data).
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_score  # noqa: E402  (the sibling file: this one is run by path, outside the package)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, required=True)
    a = p.parse_args()
    metric = report_score.radcliq_metric()
    lock = threading.Lock()  # one scoring at a time: the metric models share one GPU with the student
    calls = [0]

    def score(refs, hyps):
        hyps = [report_score.norm(h) or "." for h in hyps]
        return [float(-x) for x in report_score.composite(metric, refs, hyps)["radcliq"]]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"ok": True, "calls": calls[0]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                rewards = score(req["refs"], req["hyps"])
                calls[0] += 1
                if calls[0] % 200 == 1:
                    print(f"call {calls[0]}: {len(rewards)} reports, mean reward {np.mean(rewards):.3f}", flush=True)
            self._send({"rewards": rewards})

    print(f"reward server ready on {a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
