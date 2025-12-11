import json
import os
import shutil
import subprocess
from typing import Any

from ..types import MessageList, SamplerBase, SamplerResponse


class GeminiCLISampler(SamplerBase):
    """
    Sample from the Gemini CLI.
    
    Uses non-interactive mode: `gemini -p "<prompt>" -m <model> --output-format json`
    Requires GEMINI_API_KEY to be set.
    """

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        system_message: str | None = None,
        binary: str = "gemini",
    ):
        self.api_key_name = "GEMINI_API_KEY"
        self.model = model
        self.system_message = system_message
        self.binary = binary
        self.image_format = "base64"

        if shutil.which(self.binary) is None:
            raise FileNotFoundError(
                f"Gemini CLI binary '{self.binary}' not found on PATH. "
                "Install with: npm install -g @google/gemini-cli"
            )
        
        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError(
                f"{self.api_key_name} or GOOGLE_API_KEY must be set. "
                "Get your API key from https://aistudio.google.com/apikey"
            )

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ) -> dict[str, Any]:
        raise NotImplementedError("Gemini CLI sampler currently supports text only.")

    def _handle_text(self, text: str) -> str:
        return text

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _format_prompt(self, message_list: MessageList) -> str:
        """Convert role/content messages into a plain-text transcript."""
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
            "-p",
            prompt,
            "-m",
            self.model,
            "--output-format",
            "json",
        ]

        completed = subprocess.run(cmd, text=True, capture_output=True)

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if completed.returncode != 0:
            error_msg = f"Gemini CLI error: {stderr}" if stderr else f"returncode {completed.returncode}"
            raise RuntimeError(f"Gemini CLI failed: {error_msg}")

        if not stdout:
            raise RuntimeError("Gemini CLI returned empty response")

        try:
            data = json.loads(stdout)
            # Gemini CLI returns JSON with "response" field
            response_text = (
                data.get("response")
                or data.get("output")
                or data.get("output_text")
                or data.get("text")
                or None
            )
            # Check candidates array if direct fields weren't found
            if not response_text and isinstance(data, dict) and "candidates" in data:
                candidates = data.get("candidates") or []
                if candidates:
                    response_text = (
                        candidates[0].get("response")  # type: ignore[attr-defined]
                        or candidates[0].get("output")  # type: ignore[attr-defined]
                        or candidates[0].get("output_text")  # type: ignore[attr-defined]
                        or None
                    )
            # Final fallback to raw stdout if nothing found
            if not response_text:
                response_text = stdout
        except json.JSONDecodeError:
            response_text = stdout

        return SamplerResponse(
            response_text=response_text,
            response_metadata={
                "returncode": completed.returncode,
                "stderr": stderr,
                "raw_stdout": stdout,
            },
            actual_queried_message_list=message_list,
        )
