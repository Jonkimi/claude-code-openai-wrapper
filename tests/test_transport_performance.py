import asyncio
import os
import tempfile
import time
import psutil
from pathlib import Path
import shutil
import traceback

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    ClaudeSDKClient,
)
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport


# --- Configuration ---
# Attempt to get CLAUDE_CLI_PATH from environment, fallback to a common path
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", "claude")
# Define CLAUDE_TOOLS (assuming it's a list of tool names)
# CLAUDE_TOOLS = ["Read", "Write", "Bash"]  # Example tools, adjust if actual constants are different
from src.constants import CLAUDE_TOOLS


async def test_subprocess_cli_transport_performance():
    """
    Tests the startup time and memory usage of SubprocessCLITransport.
    """
    print("\n🧪 Testing SubprocessCLITransport performance (startup time and memory usage)...")

    temp_dir = None
    transport = None
    try:
        # Create a temporary working directory for the agent
        temp_dir = tempfile.mkdtemp(prefix="claude_cli_transport_test_")
        cwd = Path(temp_dir)
        print(f"  Using temporary isolated workspace: {cwd}")

        # Define ClaudeAgentOptions
        options = ClaudeAgentOptions(
            stderr=lambda msg: print(f"CLI stderr: {msg}"),
            cli_path=CLAUDE_CLI_PATH,
            max_turns=1,
            cwd=cwd,
            permission_mode="bypassPermissions",
            disallowed_tools=CLAUDE_TOOLS,
            system_prompt="You are a helpful assistant.",
            # system_prompt={
            #     "type": "preset",
            #     "preset": "claude_code",
            # }",
        )

        # Create a dummy prompt and configured_options for SubprocessCLITransport
        # The actual prompt content might not matter for just measuring startup
        prompt = "Hello"
        configured_options = {"max_tokens": 50}  # Example, adjust if needed

        # Instantiate SubprocessCLITransport
        transport = SubprocessCLITransport(
            prompt="1+1",
            options=options,
        )

        # --- Measure startup time and memory ---
        start_time = time.time()
        # await transport.connect()
        end_time = time.time()

        # Get process details
        # if transport._process is None:
        #     raise RuntimeError("Transport process was not started.")
        # proc = psutil.Process(transport._process.pid)
        # process_create_time = proc.create_time()
        # memory_info = proc.memory_info()

        # print(f"\n📊 Performance Metrics:")
        # print(f"  Process ID: {transport._process.pid}")
        # print(f"  Process created at: {time.ctime(process_create_time)}")
        # print(f"  Transport connect duration: {end_time - start_time:.4f} seconds")
        # print(f"  Memory usage (RSS): {memory_info.rss / (1024 * 1024):.2f} MB")
        # print(f"  Memory usage (VMS): {memory_info.vms / (1024 * 1024):.2f} MB")

        # wait 20s for process initialization
        # await asyncio.sleep(20)

        # Get process details after 20s
        # proc = psutil.Process(transport._process.pid)
        # memory_info = proc.memory_info()
        # print(f"  Memory usage (RSS): {memory_info.rss / (1024 * 1024):.2f} MB")
        # print(f"  Memory usage (VMS): {memory_info.vms / (1024 * 1024):.2f} MB")

        try:

            # 创建客户端时传入预先初始化的 transport
            # async with ClaudeSDKClient(options=options) as client:
            #     # 客户端会跳过 transport 创建和连接
            #     await client.query("Hello")
            #     async for msg in client.receive_response():
            #         print(msg)
            async for msg in query(prompt="Hello", options=options, transport=transport):
                print(msg)
        finally:
            # memory_info = proc.memory_info()
            # print(f"  Memory usage (RSS): {memory_info.rss / (1024 * 1024):.2f} MB")
            # print(f"  Memory usage (VMS): {memory_info.vms / (1024 * 1024):.2f} MB")
            # 确保清理资源
            # await transport.close()
            pass

        # Basic assertion to ensure it connected
        assert (
            transport._process.returncode is None
        ), "SubprocessCLITransport process should be running"
        print("  ✓ SubprocessCLITransport connected successfully.")
        return True

    except Exception as e:
        traceback.print_exc()
        print(f"  ❌ Test failed with exception: {e}")
        return False
    finally:
        if transport is None:
            return
        # Disconnect transport (if connected)
        if "transport" in locals() and getattr(transport, "_process", None):
            # Only try to terminate if the process is still running (returncode is None)
            if transport._process.returncode is None:
                print("  Terminating SubprocessCLITransport...")
                transport._process.terminate()
                print("  SubprocessCLITransport terminated.")
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"  Cleaned up temporary workspace: {temp_dir}")
            except Exception as e:
                print(f"  ⚠️  Failed to clean up temp directory {temp_dir}: {e}")


async def main():
    """Run the performance test."""
    print("=" * 60)
    print("SubprocessCLITransport Performance Test")
    print("=" * 60)

    success = await test_subprocess_cli_transport_performance()

    print("\n" + "=" * 60)
    if success:
        print("🎉 SubprocessCLITransport performance test PASSED!")
    else:
        print("❌ SubprocessCLILibrary performance test FAILED.")
        print(
            "⚠️  Check the output above for details and ensure 'claude' CLI is installed and accessible."
        )
    print("=" * 60)
    return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
