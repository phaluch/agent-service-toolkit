"""Perplexity MCP tools for the personal assistant.

Spawns the Perplexity MCP server as a local stdio subprocess via npx.
The connection is initialized lazily on first use and kept alive for the
lifetime of the service.
"""

import asyncio
import logging

from langchain_core.tools import BaseTool

from core import settings

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()
_tools: list[BaseTool] | None = None


async def get_perplexity_tools() -> list[BaseTool]:
    """Return Perplexity search tools via MCP, initializing the connection lazily.

    Returns an empty list if PERPLEXITY_API_KEY is not set or if the connection fails.
    """
    global _tools
    if _tools is not None:
        return _tools

    if not settings.PERPLEXITY_API_KEY:
        return []

    async with _lock:
        if _tools is not None:
            return _tools

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient

            client = MultiServerMCPClient(
                {
                    "perplexity": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "@perplexity-ai/mcp-server"],
                        "env": {
                            "PERPLEXITY_API_KEY": settings.PERPLEXITY_API_KEY.get_secret_value()
                        },
                    }
                }
            )
            _tools = await client.get_tools()
            logger.info(f"Initialized {len(_tools)} Perplexity MCP tools")
        except Exception as e:
            logger.error(f"Failed to initialize Perplexity MCP tools: {e}")
            _tools = []

    return _tools
