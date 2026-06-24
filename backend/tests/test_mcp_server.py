"""
Test MCP Server

Basic tests to verify MCP server functionality without requiring an MCP client.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
# Add backend to path (parent of tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import pytest
from mcp_server import app, list_tools, list_resources


@pytest.mark.asyncio
async def test_mcp_server():
    """Test MCP server components."""
    print("\n" + "="*60)
    print("MCP SERVER TESTS")
    print("="*60)
    
    # Test 1: List Tools
    print("\n" + "="*60)
    print("TEST 1: List Available Tools")
    print("="*60)
    
    tools = await list_tools()
    print(f"\nFound {len(tools)} tools:")
    for tool in tools:
        print(f"\n  Tool: {tool.name}")
        print(f"  Description: {tool.description}")
        print(f"  Required params: {tool.inputSchema.get('required', [])}")
    
    # Test 2: List Resources
    print("\n" + "="*60)
    print("TEST 2: List Available Resources")
    print("="*60)
    
    resources = await list_resources()
    print(f"\nFound {len(resources)} resources:")
    for resource in resources:
        print(f"\n  URI: {resource.uri}")
        print(f"  Name: {resource.name}")
        print(f"  Type: {resource.mimeType}")
    
    # Test 3: Server Info
    print("\n" + "="*60)
    print("TEST 3: Server Information")
    print("="*60)
    
    from config import get_settings
    settings = get_settings()
    
    print(f"\n  Server Name: {settings.mcp_server_name}")
    print(f"  Server Version: {settings.mcp_server_version}")
    print(f"  LLM Mode: {settings.llm_mode}")
    print(f"  Reranking: {'enabled' if settings.enable_reranking else 'disabled'}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
    print("\nMCP Server is ready!")
    print("\nNext steps:")
    print("  1. Configure MCP client (Claude Desktop, Cline, etc.)")
    print("  2. Add this server to client configuration")
    print("  3. Test with: npx @modelcontextprotocol/inspector python run_mcp_server.py")
    print("\nSee README_MCP.md for detailed setup instructions.")


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
