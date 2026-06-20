#!/usr/bin/env python3
"""Bake the demo trace + reliability data into a single self-contained HTML file.

Produces web/index.html with the event traces embedded inline, so it opens with no
server and no network (also valid as a claude.ai Artifact). Re-run after regenerating
the traces:

    python demo/run_demo.py --json-out web/trace.json
    python demo/reliability.py --trials 40 --json-out web/reliability.json
    python demo/build_web.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def main() -> None:
    trace = json.loads((WEB / "trace.json").read_text())
    reliability = json.loads((WEB / "reliability.json").read_text())
    template = (WEB / "_template.html").read_text()
    data_blob = json.dumps({"trace": trace, "reliability": reliability})
    html = template.replace("/*__DATA__*/", f"window.__QUORUM__ = {data_blob};")
    (WEB / "index.html").write_text(html)
    print(f"wrote {WEB / 'index.html'} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
