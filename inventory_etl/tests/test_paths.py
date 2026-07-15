"""Regression tests for repo/ETL path resolution after the move into the
larger Inventory-Planning-Agent repository.

These are offline: they never open a DB connection and never read or print the
real password from `.env`.
"""
from pathlib import Path


def test_import_etl_and_config_from_repo_root():
    import etl
    assert etl.__version__            # package imports
    from etl import config            # config imports from repo root
    assert hasattr(config, "REPO_ROOT")
    assert hasattr(config, "ETL_ROOT")


def test_repo_and_etl_roots_resolve():
    from etl import config
    assert config.ETL_ROOT.name == "inventory_etl"
    assert config.REPO_ROOT.name == "Inventory-Planning-Agent"
    assert config.ETL_ROOT.parent == config.REPO_ROOT
    assert config.ETL_ROOT.is_dir()


def test_env_path_is_repo_root_dotenv():
    from etl import config
    assert config.ENV_PATH == config.REPO_ROOT / ".env"


def test_config_path_exists_under_inventory_etl():
    from etl import config
    assert config.CONFIG_PATH == config.ETL_ROOT / "config" / "config.yaml"
    assert config.CONFIG_PATH.exists(), "config.yaml must live at inventory_etl/config/"


def test_default_sqlite_target(monkeypatch):
    from etl import config
    monkeypatch.delenv("TARGET_SQLITE_PATH", raising=False)
    assert config.target_sqlite_path() == config.ETL_ROOT / "output" / "inventory.db"


def test_relative_target_resolves_against_repo_root(monkeypatch):
    from etl import config
    monkeypatch.setenv("TARGET_SQLITE_PATH", "inventory_etl/output/inventory.db")
    assert config.target_sqlite_path() == config.ETL_ROOT / "output" / "inventory.db"


def test_settings_loads_and_is_mapping():
    from etl import config
    s = config.settings()
    assert isinstance(s, dict) and "channels" in s
