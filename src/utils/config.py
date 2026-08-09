"""
Configuration and path utilities
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parents[2]


def setup_project_path():
    """Add project root to sys.path if not already present"""
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def load_env_config():
    """Load environment variables from .env file"""
    project_root = get_project_root()
    env_path = project_root / ".env"
    load_dotenv(env_path)
    return env_path.exists()
