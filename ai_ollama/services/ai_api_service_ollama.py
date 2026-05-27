from typing import Any, Unpack, override

from odoo.api import Environment

from odoo.addons.ai.services.ai_api_service import AIApiService
from odoo.addons.ai.services.types.types import CompletionOptions, EmbeddingOptions, Tool


DEFAULT_URL = "http://localhost:11434/v1"


class AIApiServiceOllama(AIApiService):
    @override
    def get_completions(
        self,
        model: str,
        messages: list[dict[str, Any]],
        instructions: list[str],
        tools: list[Tool] | None = None,
        **options: Unpack[CompletionOptions],
    ) -> list[dict[str, Any]]:
        super().get_completions(model, messages, instructions, tools, **options)
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": "\n".join([*instructions, "Reply in plain text. Do not use markdown formatting."])}, *messages],
            "temperature": options.get("temperature", 0.5),
            "max_tokens": 512,
            "num_ctx": 2048,
            "stream": False,
            "think": False,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        response = self._request("post", "/chat/completions", body, self._get_base_headers(), timeout=options.get("timeout") or 120)
        return [response["choices"][0]["message"]]

    @override
    def get_embeddings(self, model: str, input: str | list[str], **options: Unpack[EmbeddingOptions]) -> list[list[float]]:
        raise NotImplementedError

    @override
    def get_transcription(self, model: str, audio: bytes, **options: Any) -> str | None:
        raise NotImplementedError

    @override
    def get_realtime_session(self, model: str, expiry: int, language: str, prompt: str) -> Any:
        raise NotImplementedError

    @override
    def _get_base_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    @classmethod
    @override
    def _get_api_token(cls, env: Environment, raise_if_not_found: bool = True) -> str:
        return ""

    @classmethod
    def get_base_url(cls, env: Environment) -> str:
        return env["ir.config_parameter"].sudo().get_str("ai.ollama_url") or DEFAULT_URL
