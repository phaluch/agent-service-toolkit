"""Todoist MCP tools for the personal assistant.

Connects to the Todoist hosted MCP server (https://ai.todoist.net/mcp) using a
TODOIST_API_KEY bearer token. The connection is initialized lazily on first use
and kept alive for the lifetime of the service.
"""

import asyncio
import logging

from langchain_core.tools import BaseTool

from core import settings

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_tools: list[BaseTool] | None = None


async def get_todoist_tools() -> list[BaseTool]:
    """Return Todoist tools via MCP, initializing the connection lazily.

    Returns an empty list if TODOIST_API_KEY is not set or if the connection fails.
    """
    global _tools
    if _tools is not None:
        return _tools

    if not settings.TODOIST_API_KEY:
        return []

    async with _lock:
        if _tools is not None:
            return _tools

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    "todoist": {
                        "transport": "streamable_http",
                        "url": "https://ai.todoist.net/mcp",
                        "headers": {
                            "Authorization": f"Bearer {settings.TODOIST_API_KEY.get_secret_value()}"
                        },
                    }
                }
            )
            _tools = await client.get_tools()
            logger.info(f"Initialized {len(_tools)} Todoist MCP tools")
        except Exception as e:
            logger.error(f"Failed to initialize Todoist MCP tools: {e}")
            _tools = []

    return _tools
