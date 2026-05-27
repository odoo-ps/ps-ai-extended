import json
import requests as _requests

from odoo.addons.ai.utils.llm_api_service import LLMApiService
from odoo.addons.ai.utils.llm_providers import PROVIDERS, Provider

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


def _fetch_ollama_models():
    try:
        base = DEFAULT_OLLAMA_URL.removesuffix("/v1").rstrip("/")
        resp = _requests.get(f"{base}/api/tags", timeout=3)
        resp.raise_for_status()
        return [(m["name"], m["name"]) for m in resp.json().get("models", [])]
    except Exception:
        return []


PROVIDERS.append(Provider(
    name="ollama",
    display_name="Ollama",
    embedding_model="",
    embedding_config={},
    llms=_fetch_ollama_models(),
))


# ── LLMApiService patches ──────────────────────────────────────────────────

_orig_init = LLMApiService.__init__


def _patched_init(self, env, provider="openai"):
    if provider == "ollama":
        self.provider = "ollama"
        self.base_url = (
            env["ir.config_parameter"].sudo().get_param("ai.ollama_url")
            or DEFAULT_OLLAMA_URL
        )
        self.env = env
    else:
        _orig_init(self, env, provider)


LLMApiService.__init__ = _patched_init


_orig_get_api_token = LLMApiService._get_api_token


def _patched_get_api_token(self):
    if self.provider == "ollama":
        return ""
    return _orig_get_api_token(self)


LLMApiService._get_api_token = _patched_get_api_token


def _request_llm_ollama(
    self, llm_model, system_prompts, user_prompts, tools=None,
    files=None, schema=None, temperature=0.2, inputs=(), web_grounding=False,
):
    messages = []
    if system_prompts:
        messages.append({"role": "system", "content": "\n".join(system_prompts)})
    if user_prompts:
        messages.append({"role": "user", "content": "\n".join(user_prompts)})
    messages.extend(inputs)

    body = {
        "model": llm_model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        body["tools"] = [{
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": tool_parameter_schema,
            },
        } for tool_name, (tool_description, __, __, tool_parameter_schema) in tools.items()]
        body["tool_choice"] = "auto"

    response = self._request(
        method="post",
        endpoint="/chat/completions",
        headers={"Content-Type": "application/json"},
        body=body,
    )

    msg = response["choices"][0]["message"]
    to_call = []
    next_inputs = list(inputs)
    responses = []

    if tool_calls := msg.get("tool_calls"):
        next_inputs.append(msg)
        for tc in tool_calls:
            try:
                arguments = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                continue
            to_call.append((tc["function"]["name"], tc["id"], arguments))
    elif content := msg.get("content"):
        responses.append(content)

    return responses, to_call, next_inputs


LLMApiService._request_llm_ollama = _request_llm_ollama


_orig_request_llm = LLMApiService._request_llm


def _patched_request_llm(self, *args, **kwargs):
    if self.provider == "ollama":
        return self._request_llm_ollama(*args, **kwargs)
    return _orig_request_llm(self, *args, **kwargs)


LLMApiService._request_llm = _patched_request_llm


_orig_build_tool_call_response = LLMApiService._build_tool_call_response


def _patched_build_tool_call_response(self, tool_call_id, return_value):
    if self.provider == "ollama":
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(return_value),
        }
    return _orig_build_tool_call_response(self, tool_call_id, return_value)


LLMApiService._build_tool_call_response = _patched_build_tool_call_response
