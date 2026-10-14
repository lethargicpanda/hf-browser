"""Exercise the actual Space HTTP handler with a small simulated model."""

import contextlib
import io
import json
import os
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, mock_open, patch


class Streamer:
    def __init__(self, *args, **kwargs):
        self.queue = queue.Queue()

    def on_finalized_text(self, text, stream_end=False):
        if text:
            self.queue.put(text)
        if stream_end:
            self.queue.put(None)

    def __iter__(self):
        while True:
            text = self.queue.get(timeout=2)
            if text is None:
                return
            yield text


class Inputs(dict):
    def __init__(self):
        self.input_ids = SimpleNamespace(shape=(1, 5))
        super().__init__(input_ids=self.input_ids)

    def to(self, device):
        return self


class SpaceServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (Path(__file__).parents[1] / "src/hf_browser/templates/server.py").read_text()
        tokenizer = Mock(chat_template="template", eos_token_id=0)
        tokenizer.return_value = Inputs()
        tokenizer.apply_chat_template.return_value = "Hello"
        model = Mock(config=SimpleNamespace(max_position_embeddings=4096))
        model.to.return_value = model

        def generate(**kwargs):
            kwargs["streamer"].on_finalized_text("你好 from Space", stream_end=True)

        model.generate.side_effect = generate
        cls.model = model
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), float32="float32",
                                float16="float16", inference_mode=contextlib.nullcontext)
        transformers = SimpleNamespace(AutoTokenizer=Mock(from_pretrained=Mock(return_value=tokenizer)),
            AutoModelForCausalLM=Mock(from_pretrained=Mock(return_value=model)),
            TextIteratorStreamer=Streamer, StoppingCriteria=object, StoppingCriteriaList=list)
        namespace = {}
        spec = json.dumps({"model": "owner/model", "revision": "a" * 40})
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.dict(
                os.environ, {"API_KEY": "test-deployment-key"}), patch("builtins.open", mock_open(read_data=spec)), patch(
                "http.server.ThreadingHTTPServer"):
            exec(compile(source, "space-server.py", "exec"), namespace)
        cls.Handler = namespace["Handler"]
        cls.lock = namespace["LOCK"]
        cls.factory = transformers.AutoModelForCausalLM.from_pretrained

    def handler(self, payload, key="test-deployment-key"):
        instance = self.Handler.__new__(self.Handler)
        body = json.dumps(payload).encode()
        instance.headers = {"Content-Length": str(len(body)), "X-HF-Browser-Key": key}
        instance.path = "/v1/chat/completions"
        instance.rfile = io.BytesIO(body)
        instance.wfile = io.BytesIO()
        instance.send_response = Mock()
        instance.send_header = Mock()
        instance.end_headers = Mock()
        return instance

    def test_real_handler_streams_generated_text(self):
        handler = self.handler({"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64})
        handler.do_POST()
        handler.send_response.assert_called_once_with(200)
        body = handler.wfile.getvalue().decode()
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {")]
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "你好 from Space")
        self.assertIn("data: [DONE]", body)
        self.assertFalse(self.lock.locked())
        self.assertTrue(self.factory.call_args.kwargs["use_safetensors"])
        self.assertFalse(self.factory.call_args.kwargs["trust_remote_code"])

    def test_auth_validation_and_busy_responses(self):
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        invalid = self.handler(payload, key="wrong")
        invalid.do_POST()
        invalid.send_response.assert_called_with(401)
        malformed = self.handler({"messages": "invalid"})
        malformed.do_POST()
        malformed.send_response.assert_called_with(400)
        over_limit = self.handler(dict(payload, max_tokens=10000))
        over_limit.do_POST()
        over_limit.send_response.assert_called_with(400)
        self.lock.acquire()
        try:
            busy = self.handler(payload)
            busy.do_POST()
            busy.send_response.assert_called_with(429)
        finally:
            self.lock.release()


if __name__ == "__main__":
    unittest.main()
