
import logging
import logging.config
import os
from typing import Optional

import yaml


def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration for the BCI Compression Toolkit.
    If config_path is None, loads from config/defaults.yaml.
    Returns a dictionary of configuration parameters.
    """
    if config_path is None:
        # Correctly locate the project root and then the config file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        config_path = os.path.join(project_root, 'config', 'defaults.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(logging_config_path: Optional[str] = None):
    """
    Set up logging for the BCI Compression Toolkit using YAML config.
    Search order:
    1. BCI_LOGGING_CONFIG environment variable
    2. config/logging.yaml relative to project root
    3. config/logging.yaml relative to current working directory
    Raises a clear error if not found.
    """
    # 1. Environment variable
    env_path = os.environ.get('BCI_LOGGING_CONFIG')
    if env_path and os.path.isfile(env_path):
        logging_config_path = env_path
    # 2. Project root (from __file__)
    elif logging_config_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        candidate = os.path.join(project_root, 'config', 'logging.yaml')
        if os.path.isfile(candidate):
            logging_config_path = candidate
        else:
            # 3. CWD
            cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), 'config', 'logging.yaml'))
            if os.path.isfile(cwd_candidate):
                logging_config_path = cwd_candidate
            else:
                # Fallback to a basic config if no file is found
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logging.warning("logging.yaml not found. Falling back to basic logging config.")
                return

    try:
        with open(logging_config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Error loading logging configuration from {logging_config_path}: {e}", exc_info=True)


import logging
import logging.config
import os
from typing import Optional

import yaml


def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration for the BCI Compression Toolkit.
    If config_path is None, loads from config/defaults.yaml.
    Returns a dictionary of configuration parameters.
    """
    if config_path is None:
        # Correctly locate the project root and then the config file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        config_path = os.path.join(project_root, 'config', 'defaults.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(logging_config_path: Optional[str] = None):
    """
    Set up logging for the BCI Compression Toolkit using YAML config.
    Search order:
    1. BCI_LOGGING_CONFIG environment variable
    2. config/logging.yaml relative to project root
    3. config/logging.yaml relative to current working directory
    Raises a clear error if not found.
    """
    # 1. Environment variable
    env_path = os.environ.get('BCI_LOGGING_CONFIG')
    if env_path and os.path.isfile(env_path):
        logging_config_path = env_path
    # 2. Project root (from __file__)
    elif logging_config_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        candidate = os.path.join(project_root, 'config', 'logging.yaml')
        if os.path.isfile(candidate):
            logging_config_path = candidate
        else:
            # 3. CWD
            cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), 'config', 'logging.yaml'))
            if os.path.isfile(cwd_candidate):
                logging_config_path = cwd_candidate
            else:
                # Fallback to a basic config if no file is found
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logging.warning("logging.yaml not found. Falling back to basic logging config.")
                return

    try:
        with open(logging_config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Error loading logging configuration from {logging_config_path}: {e}", exc_info=True)


import logging
import logging.config
import os
from typing import Optional

import yaml


def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration for the BCI Compression Toolkit.
    If config_path is None, loads from config/defaults.yaml.
    Returns a dictionary of configuration parameters.
    """
    if config_path is None:
        # Correctly locate the project root and then the config file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        config_path = os.path.join(project_root, 'config', 'defaults.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(logging_config_path: Optional[str] = None):
    """
    Set up logging for the BCI Compression Toolkit using YAML config.
    Search order:
    1. BCI_LOGGING_CONFIG environment variable
    2. config/logging.yaml relative to project root
    3. config/logging.yaml relative to current working directory
    Raises a clear error if not found.
    """
    # 1. Environment variable
    env_path = os.environ.get('BCI_LOGGING_CONFIG')
    if env_path and os.path.isfile(env_path):
        logging_config_path = env_path
    # 2. Project root (from __file__)
    elif logging_config_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        candidate = os.path.join(project_root, 'config', 'logging.yaml')
        if os.path.isfile(candidate):
            logging_config_path = candidate
        else:
            # 3. CWD
            cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), 'config', 'logging.yaml'))
            if os.path.isfile(cwd_candidate):
                logging_config_path = cwd_candidate
            else:
                # Fallback to a basic config if no file is found
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logging.warning("logging.yaml not found. Falling back to basic logging config.")
                return

    try:
        with open(logging_config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Error loading logging configuration from {logging_config_path}: {e}", exc_info=True)


import logging
import logging.config
import os
from typing import Optional

import yaml


def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration for the BCI Compression Toolkit.
    If config_path is None, loads from config/defaults.yaml.
    Returns a dictionary of configuration parameters.
    """
    if config_path is None:
        # Correctly locate the project root and then the config file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        config_path = os.path.join(project_root, 'config', 'defaults.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(logging_config_path: Optional[str] = None):
    """
    Set up logging for the BCI Compression Toolkit using YAML config.
    Search order:
    1. BCI_LOGGING_CONFIG environment variable
    2. config/logging.yaml relative to project root
    3. config/logging.yaml relative to current working directory
    Raises a clear error if not found.
    """
    # 1. Environment variable
    env_path = os.environ.get('BCI_LOGGING_CONFIG')
    if env_path and os.path.isfile(env_path):
        logging_config_path = env_path
    # 2. Project root (from __file__)
    elif logging_config_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        candidate = os.path.join(project_root, 'config', 'logging.yaml')
        if os.path.isfile(candidate):
            logging_config_path = candidate
        else:
            # 3. CWD
            cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), 'config', 'logging.yaml'))
            if os.path.isfile(cwd_candidate):
                logging_config_path = cwd_candidate
            else:
                # Fallback to a basic config if no file is found
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logging.warning("logging.yaml not found. Falling back to basic logging config.")
                return

    try:
        with open(logging_config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Error loading logging configuration from {logging_config_path}: {e}", exc_info=True)


import logging
import logging.config
import os
from typing import Optional

import yaml


def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration for the BCI Compression Toolkit.
    If config_path is None, loads from config/defaults.yaml.
    Returns a dictionary of configuration parameters.
    """
    if config_path is None:
        # Correctly locate the project root and then the config file
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        config_path = os.path.join(project_root, 'config', 'defaults.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(logging_config_path: Optional[str] = None):
    """
    Set up logging for the BCI Compression Toolkit using YAML config.
    Search order:
    1. BCI_LOGGING_CONFIG environment variable
    2. config/logging.yaml relative to project root
    3. config/logging.yaml relative to current working directory
    Raises a clear error if not found.
    """
    # 1. Environment variable
    env_path = os.environ.get('BCI_LOGGING_CONFIG')
    if env_path and os.path.isfile(env_path):
        logging_config_path = env_path
    # 2. Project root (from __file__)
    elif logging_config_path is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        candidate = os.path.join(project_root, 'config', 'logging.yaml')
        if os.path.isfile(candidate):
            logging_config_path = candidate
        else:
            # 3. CWD
            cwd_candidate = os.path.abspath(os.path.join(os.getcwd(), 'config', 'logging.yaml'))
            if os.path.isfile(cwd_candidate):
                logging_config_path = cwd_candidate
            else:
                # Fallback to a basic config if no file is found
                logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                logging.warning("logging.yaml not found. Falling back to basic logging config.")
                return

    try:
        with open(logging_config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Error loading logging configuration from {logging_config_path}: {e}", exc_info=True)
