# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import logging
import subprocess

import pytest
import requests

cpu_url = "http://localhost:8889"
# gpu_url = "http://localhost:8888" - will be added as soon as self-hosted gpu runner is available

logger = logging.getLogger(__name__)


def _docker_container_running() -> bool:
    """Return True if any Docker container is currently running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-q"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

    return bool(result.stdout.strip())


pytestmark = pytest.mark.skipif(
    not _docker_container_running(),
    reason="requires Docker with a running container exposing JupyterLab",
)


def test_jupyterlab_running():
    """Check that the JupyterLab instance inside the container is reachable."""
    try:
        response = requests.get(cpu_url, timeout=5)  # timeout prevents hanging
        logger.info("✅ Connected to %s, status code: %s", cpu_url, response.status_code)
        assert response.status_code in [200, 302], f"Unexpected status code: {response.status_code}"

    except requests.exceptions.ConnectionError:
        pytest.fail(f"❌ Could not connect to {cpu_url} (connection refused or server not running)")

    except requests.exceptions.Timeout:
        pytest.fail(f"⏰ Connection to {cpu_url} timed out")

    except requests.exceptions.RequestException as e:
        pytest.fail(f"⚠️ General error connecting to {cpu_url}: {e}")
