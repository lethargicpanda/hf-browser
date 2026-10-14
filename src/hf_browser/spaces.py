"""Create model-serving Docker Spaces; never overwrite unrelated repositories."""

import base64
import hashlib
import hmac
import json
from pathlib import Path
import re
from urllib.parse import quote, urlparse

from .inference import APIError, HUB, json_request, token

HARDWARE = ("cpu-basic", "t4-small", "a10g-small", "a100-large")
TEMPLATE_VERSION = 1


def validate_repo(repo_id):
    if not re.fullmatch(r"[\w-]+/[\w.-]+", repo_id, re.ASCII) or ".." in repo_id or "--" in repo_id:
        raise ValueError("Use a Space ID like username/model-test (no '..' or '--').")
    return quote(repo_id, safe="/")


def deployment_key(repo_id):
    return hmac.new(token().encode(), ("hf-browser-space:" + repo_id).encode(), hashlib.sha256).hexdigest()


def deployment_spec(detail):
    if detail.get("pipeline_tag") != "text-generation" or detail.get("library_name") != "transformers":
        raise ValueError("Space deployment supports Transformers text-generation models only.")
    revision = detail.get("sha")
    if not revision or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Cannot deploy without an immutable model revision.")
    model_id = detail["id"]
    validate_repo(model_id)
    config = json_request(f"{HUB}/{model_id}/resolve/{revision}/config.json")
    architectures = config.get("architectures") or []
    if config.get("auto_map") or config.get("quantization_config") or not any(
            a.endswith(("ForCausalLM", "LMHeadModel")) for a in architectures):
        raise ValueError("This template requires a standard causal language model without remote code or quantized weights.")
    return {"model": model_id, "revision": revision, "version": TEMPLATE_VERSION}


def space_files(spec):
    root = Path(__file__).parent / "templates"
    readme = ("---\ntitle: hf-browser model API\nemoji: 🤗\ncolorFrom: yellow\ncolorTo: gray\n"
              "sdk: docker\napp_port: 7860\nhf_browser: " + json.dumps(spec) +
              "\n---\n\nText inference API for hf-browser. The interactive UI runs in the terminal.\n")
    return {"README.md": readme, "model.json": json.dumps(spec),
            "Dockerfile": (root / "Dockerfile").read_text(),
            "server.py": (root / "server.py").read_text()}


def deploy(repo_id, spec, hardware="cpu-basic", private=True, gated=False, emit=lambda _: None):
    path = validate_repo(repo_id)
    if hardware not in HARDWARE:
        raise ValueError("Unsupported hardware.")
    # Metadata equality allows reuse but not replacement of other apps/models.
    try:
        existing = space_info(repo_id)
    except APIError as exc:
        if exc.status != 404:
            raise
        existing = None
    if existing is not None:
        if (existing.get("cardData") or {}).get("hf_browser") != spec:
            raise APIError("Space already exists with different content. Choose another name; it will not be overwritten.")
        emit("Reusing existing Space; hardware and visibility are unchanged.")
        return repo_id
    owner, name = repo_id.split("/")
    secrets = [{"key": "API_KEY", "value": deployment_key(repo_id)}]
    if gated:
        secrets.append({"key": "HF_TOKEN", "value": token()})
    emit("Creating Space repository…")
    json_request(f"{HUB}/api/repos/create", payload={"type": "space", "name": name,
                 "organization": owner, "visibility": "private" if private else "public", "sdk": "docker",
                 "hardware": hardware, "secrets": secrets})
    emit("Uploading serving template…")
    entries = [{"key": "header", "value": {"summary": "Deploy hf-browser model API", "description": ""}}]
    entries.extend({"key": "file", "value": {"path": name, "encoding": "base64",
                   "content": base64.b64encode(content.encode()).decode()}}
                   for name, content in space_files(spec).items())
    payload = ("\n".join(json.dumps(entry) for entry in entries) + "\n").encode()
    try:
        json_request(f"{HUB}/api/spaces/{path}/commit/main", payload=payload,
                     headers={"Content-Type": "application/x-ndjson"})
    except Exception as exc:
        raise APIError(f"Space {repo_id} was created but upload failed. Inspect or remove it in Space settings before retrying: {exc}") from None
    emit("Template uploaded. Waiting for build and model startup…")
    return repo_id


def space_info(repo_id):
    return json_request(f"{HUB}/api/spaces/{validate_repo(repo_id)}")


def space_endpoint(info):
    host = info.get("host") or (info.get("runtime") or {}).get("host")
    if not host:
        raise APIError("Space has not published its API hostname yet.")
    parsed = urlparse(host if "://" in host else "https://" + host)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".hf.space") or parsed.username or parsed.port:
        raise APIError("Unexpected Space hostname; refusing to send credentials.")
    return f"https://{parsed.hostname}/v1/chat/completions"
