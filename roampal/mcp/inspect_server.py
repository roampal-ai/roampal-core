"""
Minimal MCP server for Glama inspection.

Zero backend dependencies — just exposes tool definitions over stdio.
Used when the full server can't run (e.g., Docker inspection).
"""
import asyncio
import sys
import os

# Force unbuffered output so mcp-proxy sees it immediately
os.environ["PYTHONUNBUFFERED"] = "1"

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types


def run():
    server = Server("roampal")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_memory",
                description="Search across memory collections for relevant context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="add_to_memory_bank",
                description="Store permanent facts about the user",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="update_memory",
                description="Update an existing memory",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["memory_id", "content"],
                },
            ),
            types.Tool(
                name="delete_memory",
                description="Delete a memory by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                    },
                    "required": ["memory_id"],
                },
            ),
            types.Tool(
                name="score_memories",
                description="Score previous exchange outcomes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_scores": {"type": "object"},
                    },
                    "required": ["memory_scores"],
                },
            ),
            types.Tool(
                name="record_response",
                description="Store key takeaways from significant exchanges",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key_takeaway": {"type": "string"},
                    },
                    "required": ["key_takeaway"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        return [
            types.TextContent(
                type="text",
                text=f"Tool '{name}' is available. Backend not running in inspect mode.",
            )
        ]

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    asyncio.run(main())


if __name__ == "__main__":
    run()
