import os
import tempfile
import atexit
import shutil
from typing import AsyncGenerator, Dict, Any, Optional, List
from pathlib import Path
import logging
import time
from claude_agent_sdk import (
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    ClaudeSDKClient,
)
from src.constants import CLAUDE_TOOLS

logger = logging.getLogger(__name__)


CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", "claude")


class ClaudeCodeCLI:
    def __init__(self, timeout: int = 600000, cwd: Optional[str] = None):
        self.timeout = timeout / 1000  # Convert ms to seconds
        self.temp_dir = None

        # If cwd is provided (from CLAUDE_CWD env var), use it
        # Otherwise create an isolated temp directory
        if cwd:
            self.cwd = Path(cwd)
            # Check if the directory exists
            if not self.cwd.exists():
                logger.error(f"ERROR: Specified working directory does not exist: {self.cwd}")
                logger.error(
                    "Please create the directory first or unset CLAUDE_CWD to use a temporary directory"
                )
                raise ValueError(f"Working directory does not exist: {self.cwd}")
            else:
                logger.info(f"Using CLAUDE_CWD: {self.cwd}")
        else:
            # Create isolated temp directory (cross-platform)
            self.temp_dir = tempfile.mkdtemp(prefix="claude_code_workspace_")
            self.cwd = Path(self.temp_dir)
            logger.info(f"Using temporary isolated workspace: {self.cwd}")

            # Register cleanup function to remove temp dir on exit
            atexit.register(self._cleanup_temp_dir)

        # Import auth manager
        from src.auth import auth_manager, validate_claude_code_auth

        # Validate authentication
        is_valid, auth_info = validate_claude_code_auth()
        if not is_valid:
            logger.warning(f"Claude Code authentication issues detected: {auth_info['errors']}")
        else:
            logger.info(f"Claude Code authentication method: {auth_info.get('method', 'unknown')}")

        # Store auth environment variables for SDK
        self.claude_env_vars = auth_manager.get_claude_code_env_vars()
        self.client = None
        self.client_options = ClaudeAgentOptions(
            cli_path=CLAUDE_CLI_PATH,
            max_turns=1,
            cwd=self.cwd,
            permission_mode="bypassPermissions",
            disallowed_tools=CLAUDE_TOOLS,
            system_prompt={"type": "text", "text": "You are a helpful assistant."},
        )
        atexit.register(self._disconnect_client)

    def _random_session_id(self):
        return f"session-{os.urandom(8).hex()}"

    async def _connect_client(self):
        if not self.client:
            logger.info("Connecting ClaudeSDKClient...")
            self.client = ClaudeSDKClient(options=self.client_options)
            await self.client.connect()
            logger.info("ClaudeSDKClient connected.")

    def _disconnect_client(self):
        if self.client:
            logger.info("Disconnecting ClaudeSDKClient...")
            # Run async disconnect in a sync-compatible way
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.create_task(self.client.disconnect())
                else:
                    loop.run_until_complete(self.client.disconnect())
            except RuntimeError:  # No running loop
                asyncio.run(self.client.disconnect())
            self.client = None
            logger.info("ClaudeSDKClient disconnected.")

    async def verify_cli(self) -> bool:
        """Verify Claude Agent SDK is working and authenticated."""
        try:
            await self._connect_client()
            assert self.client, "Client should be initialized"
            logger.info("Testing Claude Agent SDK...")
            start_time = time.time()

            await self.client.query(prompt="Hello", session_id=self._random_session_id())

            # message = await self.client.receive_response().__anext__()

            messages = []
            async for message in self.client.receive_response():
                messages.append(message)
                if isinstance(message, AssistantMessage):
                    logger.debug(f"首条消息延迟: {time.time() - start_time:.2f}s")
                elif isinstance(message, ResultMessage):
                    logger.debug(f"总耗时: {time.time() - start_time:.2f}s")
                    logger.debug(f"API耗时: {message.duration_api_ms}ms")
                # Break early on first response to speed up verification
                # Handle both dict and object types
                msg_type = (
                    getattr(message, "type", None)
                    if hasattr(message, "type")
                    else message.get("type") if isinstance(message, dict) else None
                )
                if msg_type == "assistant":
                    break

            if messages:
                logger.info("✅ Claude Agent SDK verified successfully")
                return True
            else:
                logger.warning("⚠️ Claude Agent SDK test returned no messages")
                return False

        except Exception as e:
            logger.error(f"Claude Agent SDK verification failed: {e}")
            logger.warning("Please ensure Claude Code is installed and authenticated:")
            logger.warning("  1. Install: npm install -g @anthropic-ai/claude-code")
            logger.warning("  2. Set ANTHROPIC_API_KEY environment variable")
            logger.warning("  3. Test: claude --print 'Hello'")
            return False

    async def run_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        stream: bool = True,
        max_turns: int = 10,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        continue_session: bool = False,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run Claude Agent using the Python SDK and yield response chunks."""
        try:
            await self._connect_client()
            assert self.client, "Client should be initialized"

            # Set authentication environment variables
            original_env = {key: os.environ.get(key) for key in self.claude_env_vars}
            os.environ.update(self.claude_env_vars)

            try:
                # Run the query and yield messages
                await self.client.query(prompt=prompt, session_id=self._random_session_id())

                async for message in self.client.receive_response():
                    # logger.debug(f"Raw SDK message type: {type(message)}")
                    logger.debug(f"Raw SDK message: {message}")

                    if hasattr(message, "__dict__") and not isinstance(message, dict):
                        message_dict = {
                            attr: getattr(message, attr)
                            for attr in dir(message)
                            if not attr.startswith("_") and not callable(getattr(message, attr))
                        }
                        logger.debug(f"Converted message dict: {message_dict}")
                        yield message_dict
                    elif isinstance(message, dict):
                        yield message

            finally:
                # Restore original environment
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        except Exception as e:
            logger.error(f"Claude Agent SDK error: {e}")
            yield {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "error_message": str(e),
            }

    def parse_claude_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the assistant message from Claude Agent SDK messages."""
        for message in messages:
            # Look for AssistantMessage type (new SDK format)
            if "content" in message and isinstance(message["content"], list):
                text_parts = []
                for block in message["content"]:
                    # Handle TextBlock objects
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)

                if text_parts:
                    return "\n".join(text_parts)

            # Fallback: look for old format
            elif message.get("type") == "assistant" and "message" in message:
                sdk_message = message["message"]
                if isinstance(sdk_message, dict) and "content" in sdk_message:
                    content = sdk_message["content"]
                    if isinstance(content, list) and len(content) > 0:
                        # Handle content blocks (Anthropic SDK format)
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        return "\n".join(text_parts) if text_parts else None
                    elif isinstance(content, str):
                        return content

        return None

    def extract_metadata(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata like costs, tokens, and session info from SDK messages."""
        metadata = {
            "session_id": None,
            "total_cost_usd": 0.0,
            "duration_ms": 0,
            "num_turns": 0,
            "model": None,
        }

        for message in messages:
            # New SDK format - ResultMessage
            if message.get("subtype") == "success" and "total_cost_usd" in message:
                metadata.update(
                    {
                        "total_cost_usd": message.get("total_cost_usd", 0.0),
                        "duration_ms": message.get("duration_ms", 0),
                        "num_turns": message.get("num_turns", 0),
                        "session_id": message.get("session_id"),
                    }
                )
            # New SDK format - SystemMessage
            elif message.get("subtype") == "init" and "data" in message:
                data = message["data"]
                metadata.update({"session_id": data.get("session_id"), "model": data.get("model")})
            # Old format fallback
            elif message.get("type") == "result":
                metadata.update(
                    {
                        "total_cost_usd": message.get("total_cost_usd", 0.0),
                        "duration_ms": message.get("duration_ms", 0),
                        "num_turns": message.get("num_turns", 0),
                        "session_id": message.get("session_id"),
                    }
                )
            elif message.get("type") == "system" and message.get("subtype") == "init":
                metadata.update(
                    {"session_id": message.get("session_id"), "model": message.get("model")}
                )

        return metadata

    def estimate_token_usage(
        self, prompt: str, completion: str, model: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Estimate token usage based on character count.

        Uses rough approximation: ~4 characters per token for English text.
        This is approximate and may not match actual tokenization.
        """
        # Rough approximation: 1 token ≈ 4 characters
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(completion) // 4)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _cleanup_temp_dir(self):
        """Clean up temporary directory and disconnect client on exit."""
        self._disconnect_client()
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary workspace: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {self.temp_dir}: {e}")
