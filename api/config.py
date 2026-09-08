"""
api/config.py
-------------
Environment-variable driven configuration for the production API.

All summarizer and classifier settings continue to be controlled via their
own env vars (SUMMARIZER_*, CLASSIFIER_*) — this file only holds settings
that are specific to the API process itself.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiConfig:
    # PORT is the standard env var used by AWS ECS, Elastic Beanstalk, and
    # most container orchestration platforms to tell the process which port
    # to listen on.
    port: int = int(os.getenv("PORT", "8000"))


cfg = ApiConfig()
