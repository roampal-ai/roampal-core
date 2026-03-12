"""
Minimal MCP server for Glama inspection.

Serves MCP over streamable HTTP on port 8080 — no mcp-proxy needed.
Zero backend dependencies — just exposes tool definitions.
"""
import asyncio
import os

os.environ["PYTHONUNBUFFERED"] = "1"

from mcp.server import Server
from mcp.server.streamable_http import StreamableHTTPServerTransport
from mcp import types
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.responses import PlainTextResponse


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

    transport = StreamableHTTPServerTransport(
        mcp_session_id=None,
        is_json_response_enabled=True,
    )

    async def run_server():
        async with transport.connect() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    # Ping endpoint for health checks
    async def ping(request):
        return PlainTextResponse("pong")

    app = Starlette(
        routes=[
            Mount("/mcp", app=transport.handle_request),
            Mount("/ping", app=ping),
        ],
    )

    import uvicorn

    # Run MCP server handler in background, uvicorn in foreground
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(run_server())
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    run()
