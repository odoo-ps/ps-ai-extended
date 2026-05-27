import json
import urllib.parse
from collections.abc import Sequence
from typing import Any, final, override

import requests

from odoo.api import Environment

from odoo.addons.ai.services.ai_provider import AIProvider
from odoo.addons.ai.services.types.types import AIMessageParts, Tool, ToolSchema

from .ai_api_service_ollama import AIApiServiceOllama, DEFAULT_URL


def _ollama_base_url() -> str:
    parsed = urllib.parse.urlparse(DEFAULT_URL)
    return f"{parsed.scheme}://{parsed.netloc}"


@final
class AIProviderOllama(AIProvider):
    name = "ollama"
    display_name = "Ollama"

    @override
    @classmethod
    def get_llm_models(cls) -> list[tuple[str, str]]:
        try:
            resp = requests.get(f"{_ollama_base_url()}/api/tags", timeout=3)
            resp.raise_for_status()
            return [(model["name"], model["name"]) for model in resp.json().get("models", [])]
        except Exception:
            return []

    @override
    @classmethod
    def get_embedding_model(cls) -> str:
        return ""

    @override
    @classmethod
    def get_transcription_models(cls) -> list[str]:
        return []

    @override
    @classmethod
    def get_configuration(cls) -> dict[str, Any]:
        return {"max_batch_size": 1, "max_tokens_per_request": 32000}

    @override
    @classmethod
    def get_service(cls, env: Environment, model: str) -> AIApiServiceOllama:
        return AIApiServiceOllama(env, AIApiServiceOllama.get_base_url(env))

    @override
    @classmethod
    def _format_from_llm(cls, response: Sequence[dict[str, Any]]) -> dict[str, Any]:
        msg = response[0]
        if tool_calls := msg.get("tool_calls"):
            return {
                "tool_calls": [{
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"]),
                    "call_id": tc["id"],
                } for tc in tool_calls]
            }
        return {"message_parts": [{"type": "text", "content": {"data": msg.get("content") or ""}}]}

    @override
    @classmethod
    def _format_tool_outputs_to_llm(cls, tool_outputs: Sequence[dict[str, Any]], model: str) -> list[dict]:
        return [{
            "role": "tool",
            "tool_call_id": tool_output["tool_call"]["call_id"],
            "content": " ".join(
                part["content"]["data"]
                for part in tool_output["result"]
                if part["type"] == "text"
            ),
        } for tool_output in tool_outputs]

    @override
    @classmethod
    def _format_message_to_llm(cls, input_message: AIMessageParts, role: str, model: str) -> list[dict]:
        content = " ".join(
            part["content"]["data"]
            for part in input_message
            if part["type"] == "text"
        )
        return [{"role": "assistant" if role in ("assistant", "model") else "user", "content": content}]

    @override
    @classmethod
    def _format_tool(cls, env: Environment, name: str, description: str, schema: ToolSchema, tool_xml_id: str) -> Tool:
        return {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": schema},
        }

    @override
    @classmethod
    def _is_active(cls, env: Environment) -> bool:
        return bool(cls.get_llm_models())

    @override
    @classmethod
    def _allows_multimodal_tool_output(cls, llm_model: str) -> bool:
        return False
