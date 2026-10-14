"""Space-only model server. UI and provider credentials remain in the TUI."""

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

with open("model.json") as handle:
    SPEC = json.load(handle)
KEY = os.environ["API_KEY"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = AutoTokenizer.from_pretrained(SPEC["model"], revision=SPEC["revision"], trust_remote_code=False)
MODEL = AutoModelForCausalLM.from_pretrained(
    SPEC["model"], revision=SPEC["revision"], trust_remote_code=False,
    use_safetensors=True, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE)
MODEL.eval()
LOCK = threading.Lock()


class Stop(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # Avoid storing prompts, authorization or request bodies in logs.

    def reply(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply(200 if self.path == "/health" else 404, {"model": SPEC["model"], "ready": True})

    def do_POST(self):
        if not hmac.compare_digest(self.headers.get("X-HF-Browser-Key", ""), KEY):
            self.reply(401, {"error": "Invalid deployment key"})
            self.close_connection = True
            return
        if self.path != "/v1/chat/completions":
            self.reply(404, {"error": "Unknown endpoint"})
            self.close_connection = True
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("Request size must be between 1 and 65536 bytes")
            data = json.loads(self.rfile.read(size))
            messages = data["messages"]
            if not isinstance(messages, list) or not messages or any(
                    not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                    or not isinstance(m.get("content"), str) for m in messages):
                raise ValueError("Invalid messages")
            temperature = float(data.get("temperature", 0.7))
            limit = int(data.get("max_tokens", 256))
            if not 0 <= temperature <= 2 or not 1 <= limit <= 4096:
                raise ValueError("Invalid generation settings")
            if TOKENIZER.chat_template:
                prompt = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt = "\n".join(m["content"] for m in messages)
            inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
            context = getattr(MODEL.config, "max_position_embeddings", 4096)
            if inputs.input_ids.shape[-1] + limit > min(context, 32768):
                raise ValueError("Prompt plus output exceeds model context limit")
        except (ValueError, KeyError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})
            self.close_connection = True
            return
        if not LOCK.acquire(blocking=False):
            self.reply(429, {"error": "Model is busy; retry shortly"})
            return
        stop = threading.Event()
        streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True, timeout=60)
        errors = []

        def generate():
            try:
                with torch.inference_mode():
                    options = dict(inputs, streamer=streamer, max_new_tokens=limit,
                                   do_sample=temperature > 0, stopping_criteria=StoppingCriteriaList([Stop(stop)]),
                                   pad_token_id=TOKENIZER.eos_token_id)
                    if temperature > 0:
                        options["temperature"] = temperature
                    MODEL.generate(**options)
            except Exception:
                errors.append("Model generation failed. Check Space logs and hardware capacity.")
                streamer.on_finalized_text("", stream_end=True)
            finally:
                LOCK.release()

        worker = threading.Thread(target=generate, daemon=True)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            worker.start()
            for text in streamer:
                self.event({"choices": [{"delta": {"content": text}}]})
            if errors:
                self.event({"error": errors[0]})
            else:
                self.event({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            stop.set()
        except Exception:
            stop.set()
            try:
                self.event({"error": "Generation timed out or connection failed."})
            except OSError:
                pass
        finally:
            stop.set()
            if worker.ident is None:
                LOCK.release()

    def event(self, data):
        self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
        self.wfile.flush()


ThreadingHTTPServer(("0.0.0.0", 7860), Handler).serve_forever()
