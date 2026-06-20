.PHONY: install demo slow live reliability test web all clean

# Editable install of the package (Python 3.10+ recommended). One-time setup.
install:
	python3 -m pip install -e ./packages/quorum-py[dev]

# Offline, deterministic baseline-vs-QUORUM demo.
demo:
	python3 demo/run_demo.py

# Slowed down for a live audience / screen recording.
slow:
	python3 demo/run_demo.py --slow 0.08

# Live run against the Claude API (needs ANTHROPIC_API_KEY).
live:
	python3 demo/run_demo.py --live

# The pass^k reliability benchmark.
reliability:
	python3 demo/reliability.py --trials 40

# Offline tests.
test:
	python3 -m pytest tests/ -q

# Regenerate traces and rebuild the self-contained web UI.
web:
	python3 demo/run_demo.py    --json-out web/trace.json >/dev/null
	python3 demo/reliability.py --trials 40 --json-out web/reliability.json >/dev/null
	python3 demo/build_web.py

# Everything: tests, then rebuild the web UI.
all: test web

clean:
	rm -rf **/__pycache__ .pytest_cache
