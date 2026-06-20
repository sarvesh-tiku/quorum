"""Setuptools shim for editable installs on older pip (<21.3).

All real metadata lives in pyproject.toml — modern pip reads that directly.
"""

from setuptools import setup

setup()
