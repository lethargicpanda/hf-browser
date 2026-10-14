"""Dependency-free Hugging Face HTTP and streamed chat clients."""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

HUB = "https://huggingface.co"
ROUTER = "https://router.huggingface.co/v1/chat/completions"


class CredentialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise APIError("Refusing a redirect to an insecure endpoint.")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
            redirected.remove_header("X-hf-browser-key")
        return redirected


class APIError(Exception):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def token():
    value = os.environ.get("HF_TOKEN", "").strip()
    if not value:
        raise APIError("Set HF_TOKEN before starting hf-browser. Providers need inference permission; deployment needs write access.")
    return value


def redact(text):
    value = os.environ.get("HF_TOKEN", "")
    if value:
        text = str(text).replace(value, "[redacted]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", str(text))


def request(url, payload=None, method=None, headers=None, authenticated=True, timeout=30):
    headers = dict(headers or {})
    headers["User-Agent"] = "hf-browser"
    if authenticated:
        headers["Authorization"] = "Bearer " + token()
    if isinstance(payload, bytes):
        body = payload
    elif payload is not None:
        body = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    else:
        body = None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        return urllib.request.build_opener(CredentialRedirect()).open(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        reasons = {401: "Invalid HF_TOKEN", 402: "Billing or inference credits required",
                   403: "Missing permission or gated model access", 404: "Resource not found",
                   409: "Resource already exists", 429: "Rate limited; try again later",
                   503: "Model or Space is not ready"}
        detail = redact(exc.read(2048).decode("utf-8", "replace"))
        raise APIError(f"{reasons.get(exc.code, 'HTTP error')} ({exc.code}): {detail}", exc.code) from None
    except (OSError, urllib.error.URLError) as exc:
        raise APIError("Network error: " + redact(str(exc))) from None


def json_request(url, **kwargs):
    with request(url, **kwargs) as response:
        return json.load(response)


def available_providers(model_id):
    from urllib.parse import quote
    # The router catalog contains chat-capable models, unlike task tags alone.
    try:
        data = json_request("https://router.huggingface.co/v1/models/" + quote(model_id, safe="/"))
    except APIError as exc:
        if exc.status == 404:
            return []
        raise
    return sorted({info["provider"] for info in data.get("providers", [])
                   if info.get("status") == "live" and info.get("provider")})


def chat_payload(model, prompt, system="", temperature=0.7, max_tokens=256):
    if not prompt.strip():
        raise ValueError("Enter a prompt first.")
    if not 0 <= temperature <= 2 or not 1 <= max_tokens <= 4096:
        raise ValueError("Temperature must be 0–2 and maximum tokens 1–4096.")
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": True}


def stream_chat(url, payload, cancel, emit, headers=None):
    """Consume SSE frames, including comments, usage-only events and UTF-8 text."""
    if cancel.is_set():
        return
    with request(url, payload=payload, headers=headers, timeout=60) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise APIError("Expected a streaming chat response from the backend.")
        data = []
        finished = False
        for raw in response:
            if cancel.is_set():
                return
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
            elif not line and data:
                frame = "\n".join(data)
                data = []
                if frame == "[DONE]":
                    return
                event = json.loads(frame)
                if event.get("error"):
                    raise APIError(redact(str(event["error"])))
                for choice in event.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if isinstance(content, str):
                        emit(content)
                    if choice.get("finish_reason"):
                        finished = True
        if not finished and not cancel.is_set():
            raise APIError("The stream ended before completion. Retry the request.")
