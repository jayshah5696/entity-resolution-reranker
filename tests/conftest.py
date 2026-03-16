import pytest
import os
from pathlib import Path

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow to run"
    )

def pytest_collection_modifyitems(config, items):
    for item in items:
        # Check if test uses a file fixture or expects a downloaded file
        # We can look for @pytest.mark.slow and check if files exist
        if "slow" in item.keywords:
            if "census" in item.name and not Path("data/raw/census_surnames.csv").exists():
                item.add_marker(pytest.mark.skip(reason="Missing census data"))
            if "ssa" in item.name and not Path("data/raw/ssa_names").exists():
                item.add_marker(pytest.mark.skip(reason="Missing SSA data"))
            if "gleif" in item.name and not Path("data/raw/gleif_golden_copy.csv").exists():
                item.add_marker(pytest.mark.skip(reason="Missing GLEIF data"))
            if "onet_alternates" in item.name and not Path("data/raw/onet_alternate_titles.txt").exists():
                item.add_marker(pytest.mark.skip(reason="Missing ONET alternates data"))
            if "onet_reported" in item.name and not Path("data/raw/onet_reported_titles.txt").exists():
                item.add_marker(pytest.mark.skip(reason="Missing ONET reported data"))
            if "edgar" in item.name and not Path("data/raw/company_tickers_exchange.json").exists():
                item.add_marker(pytest.mark.skip(reason="Missing EDGAR data"))
