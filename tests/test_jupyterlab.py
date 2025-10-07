# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import requests
import pytest

cpu_url = "http://localhost:8889"
# gpu_url = "http://localhost:8888" - will be added as soon as self-hosted gpu runner is available

def test_jupyterlab_running():
    """Check that the JupyterLab instance inside the container is reachable."""
    try:
        response = requests.get(cpu_url, timeout=5)  # timeout prevents hanging

        print(f"✅ Connected to {cpu_url}")
        print("Status Code:", response.status_code)

        # Basic validation
        assert response.status_code in [200, 302], f"Unexpected status code: {response.status_code}"

    except requests.exceptions.ConnectionError:
        pytest.fail(f"❌ Could not connect to {cpu_url} (connection refused or server not running)")

    except requests.exceptions.Timeout:
        pytest.fail(f"⏰ Connection to {cpu_url} timed out")

    except requests.exceptions.RequestException as e:
        pytest.fail(f"⚠️ General error connecting to {cpu_url}: {e}")