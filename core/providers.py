"""Model providers: one streaming chat interface, native tool calling, retries.

Every provider implements ``chat_stream(messages, system_prompt, tools)`` and
yields :class:`StreamEvent` objects (text deltas, reasoning deltas, completed
tool calls, token usage). ``complete()`` / ``stream_complete()`` remain as thin
convenience wrappers so older callers keep working.

Internal message format (provider-neutral, converted per wire protocol):

    {"role": "user", "content": "text" | [parts]}
    {"role": "assistant", "content": "text", "tool_calls": [{"id", "name", "arguments"}]}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "text"}

where a content part is ``{"type": "text", "text": ...}`` or
``{"type": "image", "mime": "image/png", "data": "<base64>"}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Idle-read timeout (seconds) between bytes on a streamed response. Reasoning
# models can sit silent for a while before the first token, so this is much
# more generous than the connect timeout. Override with options.timeout.
DEFAULT_READ_TIMEOUT = 120.0
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


@dataclass
class ModelConfig:
    name: str
    endpoint: str
    api_key: Optional[str] = None
    provider_type: str = "cloud"  # "cloud", "local", "proxy"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    # Set when the provider streamed arguments that were not valid JSON, so the
    # agent can report the problem to the model instead of running a bad call.
    parse_error: Optional[str] = None


@dataclass
class StreamEvent:
    kind: str  # "text" | "reasoning" | "tool_call" | "usage"
    text: str = ""
    tool_call: Optional[ToolCall] = None
    usage: Optional[Dict[str, int]] = None


@dataclass
class ChatResult:
    text: str = ""
    reasoning: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None


class ProviderError(httpx.HTTPError):
    """An HTTP-level failure from a provider (kept an ``httpx.HTTPError`` so
    existing ``except httpx.HTTPError`` handlers still catch it)."""

    def __init__(self, message: str, status_code: Optional[int] = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class StreamCutError(ProviderError):
    """The connection closed before the provider said the response was complete (no [DONE] / finish_reason /
    done / message_stop). Whatever arrived is partial: text may be cut mid-sentence and a tool call's arguments
    incomplete, so it must never be treated as a finished answer. ``lenient_streams: true`` in a model's
    options accepts such streams for gateways that never send a terminator."""


def _cut_error(events: int, what: str) -> StreamCutError:
    return StreamCutError(
        f"the connection closed before the model finished ({what}; {events} stream events received). "
        "The reply is incomplete: try again, or switch provider."
    )


class NativeToolsUnsupported(ProviderError):
    """The endpoint/model rejected the native ``tools`` parameter."""


class _Retry(Exception):
    def __init__(self, delay: float) -> None:
        self.delay = delay


# ── usage helpers ───────────────────────────────────────────────────────────

def _openai_usage(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Extract real token usage from an OpenAI-compatible chat response."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None and completion is None:
        return None
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    out = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(usage.get("total_tokens") or (prompt + completion)),
    }
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    if cached:
        out["cached_tokens"] = cached  # already part of prompt_tokens; billed at a discount
    return out


def _anthropic_usage(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Extract real token usage from an Anthropic Messages API response."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    if prompt is None and completion is None:
        return None
    cached = int(usage.get("cache_read_input_tokens") or 0)
    prompt = int(prompt or 0) + int(usage.get("cache_creation_input_tokens") or 0) + cached
    completion = int(completion or 0)
    out = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    if cached:
        out["cached_tokens"] = cached
    return out


def _ollama_usage(data: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Extract real token usage from an Ollama /api/chat response."""
    if not isinstance(data, dict):
        return None
    prompt = data.get("prompt_eval_count")
    completion = data.get("eval_count")
    if prompt is None and completion is None:
        return None
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


# ── message conversion ──────────────────────────────────────────────────────

def _text_of(content: Any) -> str:
    """Flatten message content (str or parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            out.append(part.get("text", ""))
    return "\n".join(out)


def _images_of(content: Any) -> List[Dict[str, str]]:
    if isinstance(content, list):
        return [p for p in content if isinstance(p, dict) and p.get("type") == "image"]
    return []


def _to_openai_messages(messages: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if system_prompt:
        out.append({"role": "system", "content": join_system_prompt(system_prompt)})
    for m in messages:
        role = m.get("role")
        if role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", ""),
                "content": _text_of(m.get("content")),
            })
        elif role == "assistant":
            msg: Dict[str, Any] = {"role": "assistant", "content": _text_of(m.get("content")) or None}
            calls = m.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})},
                    }
                    for c in calls
                ]
            elif msg["content"] is None:
                msg["content"] = ""
            out.append(msg)
        else:
            content = m.get("content")
            images = _images_of(content)
            if images:
                parts: List[Dict[str, Any]] = []
                text = _text_of(content)
                if text:
                    parts.append({"type": "text", "text": text})
                for img in images:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{img.get('mime', 'image/png')};base64,{img.get('data', '')}"},
                    })
                out.append({"role": "user", "content": parts})
            else:
                out.append({"role": "user", "content": _text_of(content)})
    return out


def _to_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def push(role: str, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            return
        # The API alternates roles; merge consecutive same-role turns.
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for m in messages:
        role = m.get("role")
        if role == "tool":
            push("user", [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": _text_of(m.get("content")) or "(empty)",
            }])
        elif role == "assistant":
            blocks: List[Dict[str, Any]] = []
            text = _text_of(m.get("content"))
            if text.strip():
                blocks.append({"type": "text", "text": text})
            for c in m.get("tool_calls") or []:
                blocks.append({
                    "type": "tool_use", "id": c["id"], "name": c["name"], "input": c.get("arguments") or {},
                })
            push("assistant", blocks)
        else:
            content = m.get("content")
            blocks = []
            for img in _images_of(content):
                blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": img.get("mime", "image/png"), "data": img.get("data", "")},
                })
            text = _text_of(content)
            if text.strip() or not blocks:
                blocks.append({"type": "text", "text": text or "(empty)"})
            push("user", blocks)
    # The API requires the first message to be from the user.
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(conversation continues)"}]})
    return out


def _to_ollama_messages(messages: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if system_prompt:
        out.append({"role": "system", "content": join_system_prompt(system_prompt)})
    for m in messages:
        role = m.get("role")
        if role == "tool":
            out.append({"role": "tool", "content": _text_of(m.get("content")), "tool_name": m.get("name", "")})
        elif role == "assistant":
            msg: Dict[str, Any] = {"role": "assistant", "content": _text_of(m.get("content"))}
            calls = m.get("tool_calls") or []
            if calls:
                msg["tool_calls"] = [
                    {"function": {"name": c["name"], "arguments": c.get("arguments") or {}}} for c in calls
                ]
            out.append(msg)
        else:
            content = m.get("content")
            msg = {"role": "user", "content": _text_of(content)}
            images = _images_of(content)
            if images:
                msg["images"] = [i.get("data", "") for i in images]
            out.append(msg)
    return out


def _openai_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def _finish_tool_call(call_id: str, name: str, raw_args: str) -> ToolCall:
    raw = (raw_args or "").strip()
    if not raw:
        return ToolCall(id=call_id, name=name, arguments={})
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ToolCall(id=call_id, name=name, arguments={}, parse_error=f"invalid JSON arguments: {exc}")
    if not isinstance(parsed, dict):
        return ToolCall(id=call_id, name=name, arguments={}, parse_error="arguments must be a JSON object")
    return ToolCall(id=call_id, name=name, arguments=parsed)


class BaseProvider(ABC):
    """Abstract base class for all model providers."""

    def __init__(self, config: ModelConfig):
        self.config = config
        # Idle-read timeout via options.timeout; connect stays short so an
        # unreachable endpoint fails fast instead of stalling the whole turn.
        read_timeout = float(config.options.get("timeout", DEFAULT_READ_TIMEOUT))
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(read_timeout, connect=DEFAULT_CONNECT_TIMEOUT)
        )
        self.max_retries = int(config.options.get("max_retries", DEFAULT_MAX_RETRIES))
        # Real token usage from the most recently completed request, when the
        # provider's response includes it. None means "not reported" and
        # callers should fall back to a char-based estimate.
        self.last_usage: Optional[Dict[str, int]] = None
        # Flipped when the endpoint rejects native tool calling, so the agent
        # falls back to the text (XML) tool protocol for the rest of the session.
        self._native_tools_disabled = False

    # ── capabilities ────────────────────────────────────────────────────
    @property
    def native_tools(self) -> bool:
        opt = self.config.options.get("native_tools", True)
        return bool(opt) and not self._native_tools_disabled

    def disable_native_tools(self) -> None:
        self._native_tools_disabled = True

    @property
    def supports_vision(self) -> bool:
        opt = self.config.options.get("vision")
        if opt is not None:
            return bool(opt)
        model = str(self.config.options.get("model", "")).lower()
        markers = ("claude", "gpt-4o", "gpt-4.1", "gpt-5", "gemma3", "gemma4", "llava", "-vl", "vision",
                   "kimi-k2.5", "kimi-k2.6", "kimi-k3", "qwen3.5", "mistral-large-3", "gemini", "fable")
        return any(m in model for m in markers)

    @property
    def context_window(self) -> int:
        try:
            return int(self.config.options.get("context_window") or 32768)
        except (TypeError, ValueError):
            return 32768

    # ── the one method providers implement ──────────────────────────────
    @abstractmethod
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a chat turn as :class:`StreamEvent` objects."""

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Collect a full streamed turn into one :class:`ChatResult`."""
        result = ChatResult()
        text: List[str] = []
        reasoning: List[str] = []
        async for ev in self.chat_stream(messages, system_prompt=system_prompt, tools=tools, **kwargs):
            if ev.kind == "text":
                text.append(ev.text)
            elif ev.kind == "reasoning":
                reasoning.append(ev.text)
            elif ev.kind == "tool_call" and ev.tool_call:
                result.tool_calls.append(ev.tool_call)
            elif ev.kind == "usage" and ev.usage:
                result.usage = ev.usage
        result.text = "".join(text)
        result.reasoning = "".join(reasoning)
        return result

    # ── convenience wrappers (older API) ────────────────────────────────
    @staticmethod
    def _prompt_messages(prompt: str, history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        messages = list(history or [])
        messages.append({"role": "user", "content": prompt})
        return messages

    async def complete(self, prompt: str, system_prompt: str = "", history: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
        """Generate a completion (history is prior ``{role, content}`` turns)."""
        result = await self.chat(self._prompt_messages(prompt, history), system_prompt=system_prompt)
        return result.text

    async def stream_complete(self, prompt: str, system_prompt: str = "", history: Optional[List[Dict[str, Any]]] = None, **kwargs) -> AsyncIterator[str]:
        async for ev in self.chat_stream(self._prompt_messages(prompt, history), system_prompt=system_prompt):
            if ev.kind == "text" and ev.text:
                yield ev.text

    async def close(self):
        await self._client.aclose()

    # ── shared HTTP plumbing ────────────────────────────────────────────
    def _retry_delay(self, resp: Optional[httpx.Response], attempt: int) -> float:
        if resp is not None:
            header = resp.headers.get("retry-after")
            if header:
                try:
                    return min(float(header), 20.0)
                except ValueError:
                    pass
        return min(0.5 * (2 ** attempt), 8.0) + random.uniform(0, 0.25)

    async def _stream_lines(
        self, url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> AsyncIterator[str]:
        """POST ``payload`` and yield response lines, retrying transient
        failures (connect errors, 429/5xx) that occur *before* any output has
        been received. Errors after output started are never retried, since
        that would duplicate streamed text."""
        attempt = 0
        while True:
            started = False
            try:
                async with self._client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")[:2000]
                        if resp.status_code in RETRYABLE_STATUS and attempt < self.max_retries:
                            raise _Retry(self._retry_delay(resp, attempt))
                        host = urlparse(url).netloc
                        message = f"HTTP {resp.status_code} from {host}: {body[:300]}"
                        if resp.status_code in (401, 403):
                            message += "\n" + _auth_hint(url, sent_key=bool((headers or {}).get("Authorization") or (headers or {}).get("x-api-key")))
                        raise ProviderError(message, status_code=resp.status_code, body=body)
                    async for line in resp.aiter_lines():
                        started = True
                        yield line
                    return
            except _Retry as retry:
                attempt += 1
                logger.info("provider retry %d/%d in %.1fs", attempt, self.max_retries, retry.delay)
                await asyncio.sleep(retry.delay)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.WriteError) as exc:
                if started or attempt >= self.max_retries:
                    raise
                delay = self._retry_delay(None, attempt)
                attempt += 1
                logger.info("provider connection retry %d/%d in %.1fs: %s", attempt, self.max_retries, delay, exc)
                await asyncio.sleep(delay)


def _tool_rejection(exc: ProviderError) -> bool:
    body = (exc.body or str(exc)).lower()
    return exc.status_code in (400, 404, 422) and (
        "tool" in body and any(w in body for w in ("support", "not allowed", "unknown", "invalid", "unrecognized", "function"))
    )


class _OpenAICompatMixin:
    """Chat/streaming for OpenAI-compatible ``/chat/completions`` endpoints."""

    def _openai_headers(self, api_key: str) -> Dict[str, str]:
        headers = {"content-type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _openai_stream(
        self,
        url: str,
        api_key: str,
        model: str,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]],
        *,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> AsyncIterator[StreamEvent]:
        headers = self._openai_headers(api_key)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": _to_openai_messages(messages, system_prompt),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        oa_tools = _openai_tools(tools)
        if oa_tools:
            payload["tools"] = oa_tools
        effort = self.config.options.get("reasoning_effort")  # type: ignore[attr-defined]
        if effort:
            payload["reasoning_effort"] = effort

        adjustments = 0
        while True:
            try:
                async for ev in self._openai_parse(url, payload, headers):
                    yield ev
                return
            except ProviderError as exc:
                body = (exc.body or "").lower()
                if exc.status_code != 400 or adjustments >= 4:
                    if oa_tools and _tool_rejection(exc):
                        raise NativeToolsUnsupported(str(exc), exc.status_code, exc.body) from exc
                    raise
                # Endpoints disagree on optional params; peel off what they reject.
                if "max_completion_tokens" in body and "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                elif "stream_options" in body and "stream_options" in payload:
                    payload.pop("stream_options")
                elif "temperature" in body and "temperature" in payload:
                    payload.pop("temperature")
                elif "reasoning_effort" in body and "reasoning_effort" in payload:
                    payload.pop("reasoning_effort")
                elif oa_tools and _tool_rejection(exc):
                    raise NativeToolsUnsupported(str(exc), exc.status_code, exc.body) from exc
                else:
                    raise
                adjustments += 1

    async def _openai_parse(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> AsyncIterator[StreamEvent]:
        acc: Dict[int, Dict[str, str]] = {}
        usage: Optional[Dict[str, int]] = None
        finished, events = False, 0
        async for line in self._stream_lines(url, payload, headers):  # type: ignore[attr-defined]
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                finished = True
                break
            try:
                data = json.loads(data_str)
            except Exception:
                continue
            if isinstance(data, dict) and data.get("error"):
                err = data["error"]
                msg = err.get("message") if isinstance(err, dict) else str(err)
                raise ProviderError(f"provider stream error: {msg}", body=json.dumps(data)[:1000])
            usage = _openai_usage(data) or usage
            choices = data.get("choices") or []
            if not choices:
                continue
            events += 1
            if choices[0].get("finish_reason"):
                finished = True
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                yield StreamEvent("reasoning", text=reasoning)
            content = delta.get("content")
            if content:
                yield StreamEvent("text", text=content)
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = acc.setdefault(idx, {"id": "", "name": "", "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name") and not slot["name"]:
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
        if not finished and not self.config.options.get("lenient_streams"):  # type: ignore[attr-defined]
            raise _cut_error(events, "no [DONE] or finish_reason")  # before any partial tool call is emitted
        for idx in sorted(acc):
            slot = acc[idx]
            if slot["name"]:
                yield StreamEvent(
                    "tool_call",
                    tool_call=_finish_tool_call(slot["id"] or f"call_{idx}", slot["name"], slot["args"]),
                )
        if usage:
            self.last_usage = usage  # type: ignore[attr-defined]
            yield StreamEvent("usage", usage=usage)


class LocalProvider(BaseProvider):
    """Provider for local LLMs via an Ollama-compatible ``/api/chat`` endpoint."""

    def _model(self) -> str:
        return self.config.options.get("model", self.config.name.lower())

    async def chat_stream(self, messages, system_prompt="", tools=None, **kwargs) -> AsyncIterator[StreamEvent]:
        url = f"{self.config.endpoint.rstrip('/')}/api/chat"
        payload: Dict[str, Any] = {
            "model": self._model(),
            "messages": _to_ollama_messages(messages, system_prompt),
            "stream": True,
            "options": {},
        }
        if "temperature" in self.config.options:
            payload["options"]["temperature"] = self.config.options["temperature"]
        if "num_ctx" in self.config.options:
            payload["options"]["num_ctx"] = self.config.options["num_ctx"]
        if "think" in self.config.options:
            payload["think"] = self.config.options["think"]
        oa_tools = _openai_tools(tools)
        if oa_tools:
            payload["tools"] = oa_tools

        n = 0
        done_seen, events_seen = False, 0
        try:
            async for line in self._stream_lines(url, payload):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                if data.get("error"):
                    raise ProviderError(f"ollama error: {data['error']}", body=str(data["error"]))
                events_seen += 1
                msg = data.get("message") or {}
                if msg.get("thinking"):
                    yield StreamEvent("reasoning", text=msg["thinking"])
                if msg.get("content"):
                    yield StreamEvent("text", text=msg["content"])
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    call = (
                        ToolCall(id=f"call_{n}", name=fn.get("name", ""), arguments=args)
                        if isinstance(args, dict)
                        else _finish_tool_call(f"call_{n}", fn.get("name", ""), args if isinstance(args, str) else "")
                    )
                    n += 1
                    if call.name:
                        yield StreamEvent("tool_call", tool_call=call)
                if data.get("done"):
                    done_seen = True
                    usage = _ollama_usage(data)
                    if usage:
                        self.last_usage = usage
                        yield StreamEvent("usage", usage=usage)
            if not done_seen and not self.config.options.get("lenient_streams"):
                raise _cut_error(events_seen, "no done marker")
        except ProviderError as exc:
            if oa_tools and _tool_rejection(exc):
                raise NativeToolsUnsupported(str(exc), exc.status_code, exc.body) from exc
            raise

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding via Ollama /api/embeddings."""
        url = f"{self.config.endpoint.rstrip('/')}/api/embeddings"
        model = self.config.options.get("embed_model", self.config.options.get("model", self.config.name.lower()))
        resp = await self._client.post(url, json={"model": model, "prompt": text})
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    async def list_models(self) -> List[str]:
        """Discover installed models via Ollama /api/tags.

        Returns a list of model names (e.g. ``llama3:latest``). Falls back to
        the configured model if the endpoint is unreachable or not Ollama.
        """
        url = f"{self.config.endpoint.rstrip('/')}/api/tags"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            models = [m for m in models if m]
            if models:
                return models
        except Exception:
            pass
        configured = self.config.options.get("model")
        return [configured] if configured else []


# Env var to consult per API host. Deliberately host-specific: the old code
# tried every provider's variable in turn, which could send one vendor's key
# to another vendor's endpoint.
_HOST_KEY_ENV = (
    ("anthropic.com", "ANTHROPIC_API_KEY"),
    ("openai.com", "OPENAI_API_KEY"),
    ("ollama.com", "OLLAMA_API_KEY"),
)


def _env_var_for(endpoint: str) -> str:
    host = urlparse(endpoint).netloc.lower()
    for suffix, env in _HOST_KEY_ENV:
        if host.endswith(suffix):
            return env
    return "MOTION_API_KEY"


def _key_fix(endpoint: str) -> str:
    return f"Add a valid key: `motion auth login` (or Ctrl+A in the app), or set {_env_var_for(endpoint)}."


def _auth_hint(endpoint: str, sent_key: bool) -> str:
    """What to do about a 401/403, phrased for the person at the keyboard."""
    host = urlparse(endpoint).netloc
    if not sent_key:
        return f"No API key was sent to {host}. {_key_fix(endpoint)}"
    return f"{host} rejected the API key (it may be expired, revoked or for another account). {_key_fix(endpoint)}"


def _missing_key_error(endpoint: str) -> Optional["ProviderError"]:
    """A clear error for hosts that always need a key, so we don't send a doomed request."""
    host = urlparse(endpoint).netloc.lower()
    if any(host.endswith(suffix) for suffix, _ in _HOST_KEY_ENV):
        return ProviderError(f"No API key configured for {host}. {_key_fix(endpoint)}", status_code=401)
    return None


def _env_key_for(endpoint: str) -> str:
    host = urlparse(endpoint).netloc.lower()
    for suffix, env in _HOST_KEY_ENV:
        if host.endswith(suffix):
            return os.environ.get(env, "")
    return os.environ.get("MOTION_API_KEY", "")


_UNCACHEABLE_BLOCKS = ("thinking", "redacted_thinking")


def _with_conversation_cache_breakpoint(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mark the end of the conversation so Anthropic caches everything before it.

    Every step of a tool loop re-sends the whole conversation; with a breakpoint on the last block, the
    next step reads the previous prefix at ~10% of the input price instead of paying for it again.
    (System prompt and tools carry their own breakpoints, so this makes 3 of the 4 allowed.) Only done
    once there is something to reuse (more than one message), and never on thinking blocks or empty text.
    """
    if len(messages) < 2:
        return messages
    last = dict(messages[-1])
    content = last.get("content")
    if isinstance(content, str):
        if not content.strip():
            return messages
        last["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    elif isinstance(content, list) and content:
        blocks = [dict(b) if isinstance(b, dict) else b for b in content]
        tail = blocks[-1]
        if not isinstance(tail, dict) or tail.get("type") in _UNCACHEABLE_BLOCKS:
            return messages
        if tail.get("type") == "text" and not str(tail.get("text", "")).strip():
            return messages
        tail["cache_control"] = {"type": "ephemeral"}
        last["content"] = blocks
    else:
        return messages
    return [*messages[:-1], last]


# Separates the STATIC part of a system prompt (tool instructions - the same for every step of a
# turn, and for every turn of a session) from the DYNAMIC part (retrieved memory, git status, the
# date - different almost every turn). Anthropic's system block can only be cached as a whole, so if
# the two are concatenated as one string the dynamic tail invalidates the cache on every single
# request; splitting lets the large static part actually get reused across a session. Composed by
# core.agent_loop._compose_system_prompt; a plain string with no marker (every other provider, and
# any caller that never splits it) behaves exactly as before.
SYSTEM_CACHE_SPLIT = "\n<<MOTION-DYNAMIC-C1F2A9>>\n"


def split_system_prompt(system_prompt: str) -> "tuple[str, str]":
    """(static, dynamic); dynamic is '' when the caller never split the prompt."""
    if SYSTEM_CACHE_SPLIT not in system_prompt:
        return system_prompt, ""
    static, _, dynamic = system_prompt.partition(SYSTEM_CACHE_SPLIT)
    return static, dynamic


def join_system_prompt(system_prompt: str) -> str:
    """The prompt as one string, for providers that cannot use the split (OpenAI-compatible, Ollama,
    legacy completion)."""
    static, dynamic = split_system_prompt(system_prompt)
    return f"{static}\n\n{dynamic}" if dynamic else static


async def _raise_stream(exc: Exception) -> AsyncIterator[StreamEvent]:
    """An async iterator that fails on first use (matches how real streams surface errors)."""
    raise exc
    yield  # pragma: no cover  (makes this an async generator)


class CloudProvider(_OpenAICompatMixin, BaseProvider):
    """Cloud LLMs: Anthropic's native Messages API, or any OpenAI-compatible API."""

    def _api_key(self) -> str:
        return self.config.api_key or _env_key_for(self.config.endpoint)

    @property
    def is_anthropic(self) -> bool:
        return "anthropic" in self.config.endpoint

    def chat_stream(self, messages, system_prompt="", tools=None, **kwargs) -> AsyncIterator[StreamEvent]:
        if not self._api_key():
            missing = _missing_key_error(self.config.endpoint)
            if missing is not None:
                return _raise_stream(missing)
        if self.is_anthropic:
            return self._anthropic_stream(messages, system_prompt, tools)
        base = self.config.endpoint.rstrip("/")
        return self._openai_stream(
            f"{base}/chat/completions",
            self._api_key(),
            self.config.options.get("model", "gpt-4o"),
            messages,
            system_prompt,
            tools,
            max_tokens=self.config.options.get("max_tokens", 4096),
            temperature=self.config.options.get("temperature", 0.8),
        )

    def _anthropic_url(self) -> str:
        base = (self.config.endpoint or "https://api.anthropic.com").rstrip("/")
        return f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"

    async def _anthropic_stream(self, messages, system_prompt, tools) -> AsyncIterator[StreamEvent]:
        opts = self.config.options
        headers = {
            "x-api-key": self._api_key(),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": opts.get("model", "claude-sonnet-5"),
            "max_tokens": opts.get("max_tokens", 4096),
            "messages": _with_conversation_cache_breakpoint(_to_anthropic_messages(messages)),
            "stream": True,
        }
        thinking_budget = opts.get("thinking_budget")
        if thinking_budget:
            payload["thinking"] = {"type": "enabled", "budget_tokens": int(thinking_budget)}
        elif opts.get("temperature") is not None:
            payload["temperature"] = opts["temperature"]
        if system_prompt:
            static, dynamic = split_system_prompt(system_prompt)
            # The static part (tool instructions) is identical for every step of this turn and every
            # turn of the session, so it gets its own cache breakpoint; the dynamic part (memory, git
            # status, the date) changes almost every turn and would invalidate a shared block, so it
            # is sent plainly, appended after. See SYSTEM_CACHE_SPLIT above. (Anthropic rejects an
            # empty text block, so a caller that split with nothing before the marker just sends the
            # dynamic part, uncached.)
            blocks = []
            if static:
                blocks.append({"type": "text", "text": static, "cache_control": {"type": "ephemeral"}})
            if dynamic:
                blocks.append({"type": "text", "text": dynamic})
            payload["system"] = blocks or [{"type": "text", "text": system_prompt}]
        if tools:
            specs = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters") or {"type": "object", "properties": {}},
                }
                for t in tools
            ]
            specs[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = specs

        blocks: Dict[int, Dict[str, Any]] = {}
        in_tokens = 0
        out_tokens = 0
        cached_tokens = 0
        stopped, seen = False, 0
        try:
            async for line in self._stream_lines(self._anthropic_url(), payload, headers):
                if not line.startswith("data:"):
                    continue
                try:
                    data = json.loads(line[5:].strip())
                except Exception:
                    continue
                etype = data.get("type")
                seen += 1
                if etype == "message_stop":
                    stopped = True
                if etype == "message_start":
                    u = (data.get("message") or {}).get("usage") or {}
                    cached_tokens = int(u.get("cache_read_input_tokens") or 0)
                    in_tokens = (
                        int(u.get("input_tokens") or 0)
                        + int(u.get("cache_creation_input_tokens") or 0)
                        + cached_tokens
                    )
                    out_tokens = int(u.get("output_tokens") or 0)
                elif etype == "content_block_start":
                    block = data.get("content_block") or {}
                    blocks[data.get("index", 0)] = {
                        "type": block.get("type"), "id": block.get("id", ""), "name": block.get("name", ""), "args": "",
                    }
                elif etype == "content_block_delta":
                    delta = data.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "text_delta" and delta.get("text"):
                        yield StreamEvent("text", text=delta["text"])
                    elif dtype == "thinking_delta" and delta.get("thinking"):
                        yield StreamEvent("reasoning", text=delta["thinking"])
                    elif dtype == "input_json_delta":
                        slot = blocks.get(data.get("index", 0))
                        if slot is not None:
                            slot["args"] += delta.get("partial_json", "")
                elif etype == "content_block_stop":
                    slot = blocks.pop(data.get("index", 0), None)
                    if slot and slot["type"] == "tool_use":
                        yield StreamEvent(
                            "tool_call", tool_call=_finish_tool_call(slot["id"], slot["name"], slot["args"])
                        )
                elif etype == "message_delta":
                    u = data.get("usage") or {}
                    out_tokens = int(u.get("output_tokens") or out_tokens)
                elif etype == "error":
                    err = data.get("error") or {}
                    raise ProviderError(
                        f"anthropic stream error: {err.get('type', 'error')}: {err.get('message', '')}",
                        body=json.dumps(data)[:1000],
                    )
            if not stopped and not opts.get("lenient_streams"):
                # includes a tool_use block cut before content_block_stop, which would otherwise vanish silently
                raise _cut_error(seen, "no message_stop")
        except ProviderError as exc:
            if tools and _tool_rejection(exc):
                raise NativeToolsUnsupported(str(exc), exc.status_code, exc.body) from exc
            raise
        if in_tokens or out_tokens:
            usage = {"prompt_tokens": in_tokens, "completion_tokens": out_tokens, "total_tokens": in_tokens + out_tokens}
            if cached_tokens:
                usage["cached_tokens"] = cached_tokens
            self.last_usage = usage
            yield StreamEvent("usage", usage=usage)

    async def embed(self, text: str) -> List[float]:
        """Embeddings via an OpenAI-compatible ``/embeddings`` endpoint.
        Only used when ``options.embed_model`` is configured."""
        model = self.config.options.get("embed_model")
        if not model or self.is_anthropic:
            raise ProviderError("no embedding model configured for this provider")
        resp = await self._client.post(
            f"{self.config.endpoint.rstrip('/')}/embeddings",
            json={"model": model, "input": text},
            headers=self._openai_headers(self._api_key()),
        )
        resp.raise_for_status()
        data = resp.json().get("data") or [{}]
        return data[0].get("embedding", [])

    @property
    def can_embed(self) -> bool:
        return bool(self.config.options.get("embed_model")) and not self.is_anthropic


class ProxyProvider(_OpenAICompatMixin, BaseProvider):
    """Custom proxy/gateway endpoints (OpenAI-compatible)."""

    def chat_stream(self, messages, system_prompt="", tools=None, **kwargs) -> AsyncIterator[StreamEvent]:
        api_key = self.config.api_key or os.environ.get("PROXY_API_KEY", "")
        return self._openai_stream(
            f"{self.config.endpoint.rstrip('/')}/chat/completions",
            api_key,
            self.config.options.get("model", "default"),
            messages,
            system_prompt,
            tools,
            max_tokens=self.config.options.get("max_tokens"),
            temperature=self.config.options.get("temperature", 0.7),
        )


class ProviderFactory:
    """Factory to resolve the correct provider based on configuration."""

    @staticmethod
    def get_provider(config: ModelConfig) -> BaseProvider:
        if config.provider_type == "local":
            return LocalProvider(config)
        elif config.provider_type == "cloud":
            return CloudProvider(config)
        elif config.provider_type == "proxy":
            return ProxyProvider(config)
        elif config.provider_type == "cli":
            # Deferred import: core.cli_delegate imports FROM this module, so a top-level import here
            # would be circular.
            from core.cli_delegate import DELEGATE_CLASSES

            delegate_key = config.options.get("delegate", "")
            cls = DELEGATE_CLASSES.get(delegate_key)
            if cls is None:
                raise ValueError(f"Unknown CLI delegate: '{delegate_key}' (known: {', '.join(DELEGATE_CLASSES)})")
            return cls(config)
        else:
            raise ValueError(f"Unsupported provider type: {config.provider_type}")
