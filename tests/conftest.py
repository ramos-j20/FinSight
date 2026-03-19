"""
Shared pytest configuration for FinSight tests.

Unit tests (tests/unit/) use mock retrievers — no infra needed.
Integration tests (tests/integration/) hit the LIVE API at localhost:8000.
"""
