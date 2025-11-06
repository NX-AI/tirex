# Copyright (c) NXAI GmbH.
# This software may be used and distributed according to the terms of the NXAI Community License Agreement.

import json
import re

import pytest
from conftest import assert_default_prediction_correct, get_default_context
from fastmcp import Client


@pytest.fixture
async def main_mcp_client(api_server):
    async with Client(f"{api_server}/mcp") as mcp_client:
        yield mcp_client


async def test_list_tools(main_mcp_client: Client):
    list_tools = await main_mcp_client.list_tools()

    assert len(list_tools) == 1


async def test_mcp(main_mcp_client):
    context, prediction_length = get_default_context()
    params = {"context": context[0], "prediction_length": prediction_length}

    result = await main_mcp_client.call_tool("tirex_model", params)

    assert result.data is not None
    assert "TiRex Forecast Results" in result.data

    result_list = json.loads(re.search(r"Forecasted values:\s*(\[.*?\])", result.data, re.DOTALL).group(1))

    assert_default_prediction_correct([result_list])
