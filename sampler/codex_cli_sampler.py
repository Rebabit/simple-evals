import json
import os
import shutil
import subprocess
from typing import Any

from ..types import MessageList, SamplerBase, SamplerResponse


class CodexCLISampler(SamplerBase):
    """
    Sample from the Codex CLI.
    
    Uses non-interactive mode: `codex exec --json "<prompt>"`
    Requires OPENAI_API_KEY environment variable to be set.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        system_message: str | None = None,
        binary: str = "codex",
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.model = model
        self.system_message = system_message
        self.binary = binary
        self.image_format = "base64"

        if shutil.which(self.binary) is None:
            raise FileNotFoundError(
                f"Codex CLI binary '{self.binary}' not found on PATH. "
                "Install with: npm install -g @openai/codex or brew install --cask codex"
            )
        
        # Check for authentication: OPENAI_API_KEY is required
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                f"{self.api_key_name} must be set. "
                f"Set it with: export {self.api_key_name}='your-api-key'\n"
                "Get your API key from: https://platform.openai.com/api-keys"
            )

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ) -> dict[str, Any]:
        raise NotImplementedError("Codex CLI sampler currently supports text only.")

    def _handle_text(self, text: str) -> str:
        return text

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _format_prompt(self, message_list: MessageList) -> str:
        """Convert role/content messages into a plain-text prompt."""
        lines: list[str] = []
        if self.system_message:
            lines.append(f"system: {self.system_message}")

        for message in message_list:
            role = message.get("role", "user")
            content = message.get("content", "")

            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                content = "\n".join(parts)

            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        prompt = self._format_prompt(message_list)

        cmd: list[str] = [
            self.binary,
            "exec",
            "--json",
            "--skip-git-repo-check",
        ]

        if self.model:
            cmd.extend(["--model", self.model])
        
        cmd.append(prompt)

        completed = subprocess.run(cmd, text=True, capture_output=True)

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if completed.returncode != 0:
            error_msg = f"Codex CLI error: {stderr}" if stderr else f"returncode {completed.returncode}"
            # Check for authentication errors
            stderr_lower = stderr.lower()
            if (
                "authentication" in stderr_lower
                or "login" in stderr_lower
                or "unauthorized" in stderr_lower
                or "api_key" in stderr_lower
                or "api key" in stderr_lower
            ):
                raise RuntimeError(
                    f"Codex CLI authentication failed: {error_msg}\n"
                    f"Please set {self.api_key_name} environment variable:\n"
                    f"  export {self.api_key_name}='your-api-key'\n"
                    "Get your API key from: https://platform.openai.com/api-keys"
                )
            raise RuntimeError(f"Codex CLI failed: {error_msg}")

        if not stdout:
            raise RuntimeError("Codex CLI returned empty response")

        # Parse JSONL output (newline-delimited JSON)
        response_text = None
        last_agent_message = None
        
        for line in stdout.split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type", "")
                
                # Look for agent_message items (Codex uses "agent_message" type)
                if event_type == "item.completed":
                    item = event.get("item", {})
                    # The item has a "type" field (not "item_type")
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            last_agent_message = text
            except json.JSONDecodeError:
                continue

        # Use the last agent message, or fallback to raw stdout
        response_text = last_agent_message or stdout

        return SamplerResponse(
            response_text=response_text,
            response_metadata={
                "returncode": completed.returncode,
                "stderr": stderr,
                "raw_stdout": stdout,
            },
            actual_queried_message_list=message_list,
        )




