import asyncio
import time
import os
import tempfile
import shutil
import logging
from pathlib import Path
from typing import AsyncGenerator, Dict, Any

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    ResultMessage,
)
from src.constants import SYSTEM_PROMPT, CLAUDE_TOOLS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
# PROMPT = "Generate a simple Python function to add two numbers."
PROMPTS = [
    "Hello Claude",
    "Write a one-line Python function to reverse a string.",
    "What is 2+2?",
    "Name a color.",
    "Say 'Hello, World!' in Python.",
    "Generate a simple Python function to add two numbers.",
    "List all questions and answers above.",
    "/clear",
    "List all questions and answers above.",
    "/context",
]
NUM_ITERATIONS = len(PROMPTS)  # Number of times to run each query method

# Attempt to get CLAUDE_CLI_PATH from environment, fallback to a common path
# This path needs to be valid for the test to run
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", "claude")  # Adjust this default if necessary


async def run_benchmark():
    """
    Runs a benchmark comparing the performance of claude_agent_sdk.query
    and ClaudeSDKClient.query over multiple requests.
    """
    temp_dir = None

    try:
        # Create a temporary working directory for the agent
        temp_dir = tempfile.mkdtemp(prefix="claude_sdk_benchmark_")
        cwd = Path(temp_dir)
        logger.info(f"Using temporary isolated workspace: {cwd}")

        # Common ClaudeAgentOptions for both methods
        options = ClaudeAgentOptions(
            cli_path=CLAUDE_CLI_PATH,
            max_turns=1,
            cwd=cwd,
            permission_mode="bypassPermissions",
            disallowed_tools=CLAUDE_TOOLS,  # disable all tools for simpler benchmark
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["user", "project", "local"],
        )

        client = None
        try:

            logger.info("Connecting ClaudeSDKClient...")
            client = ClaudeSDKClient(options=options)
            await client.connect()
            logger.info("ClaudeSDKClient connected.")
            stateful_start_time = time.perf_counter()
            for i, prompt in enumerate(PROMPTS):
                logger.info(f"Stateful query iteration {i+1}/{NUM_ITERATIONS}...")
                # Ensure all messages are consumed for proper generator cleanup
                await client.query(prompt=prompt, session_id=f"chatcmpl-{os.urandom(8).hex()}")
                async for message in client.receive_response():
                    # For ClaudeSDKClient.query, we expect a single message directly
                    logger.info(message)
                    if isinstance(message, ResultMessage):
                        logger.info(f"API耗时: {message.duration_ms}ms")

            stateful_end_time = time.perf_counter()
            stateful_total_time = stateful_end_time - stateful_start_time
            stateful_avg_time = stateful_total_time / NUM_ITERATIONS
            logger.info(
                f"Stateful `ClaudeSDKClient.query()` Total Time: {stateful_total_time:.4f} seconds"
            )
            logger.info(
                f"Stateful `ClaudeSDKClient.query()` Average Time per request: {stateful_avg_time:.4f} seconds"
            )

        finally:
            if client:
                logger.info("Disconnecting ClaudeSDKClient...")
                await client.disconnect()
                logger.info("ClaudeSDKClient disconnected.")

        # --- Summary ---
        logger.info("\n--- Benchmark Summary ---")
        logger.info(f"Stateless `query()` Average Time: {stateless_avg_time:.4f} seconds")
        logger.info(
            f"Stateful `ClaudeSDKClient.query()` Average Time: {stateful_avg_time:.4f} seconds"
        )

        if stateful_avg_time < stateless_avg_time:
            logger.info("ClaudeSDKClient.query is faster on average for multiple requests.")
        elif stateful_avg_time > stateless_avg_time:
            logger.info(
                "query() function is faster on average for multiple requests (unexpected, check setup)."
            )
        else:
            logger.info("Both methods have similar average performance.")

    except Exception as e:
        logger.error(f"An error occurred during benchmarking: {e}")
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary workspace: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
