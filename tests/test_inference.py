import io
import json
import os
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from hf_browser import inference as api


class Response(io.BytesIO):
    headers = {"Content-Type": "text/event-stream; charset=utf-8"}


class InferenceTests(unittest.TestCase):
    def test_stream_handles_unicode_usage_comments_and_done(self):
        events = [b": heartbeat\n\n", b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
                  ('data: ' + json.dumps({"choices": [{"delta": {"content": "你好"}}]}, ensure_ascii=False) + '\r\n\r\n').encode(),
                  b'data: {"choices":[],"usage":{"total_tokens":3}}\n\n', b'data: [DONE]\n\n']
        chunks = []
        with patch.object(api, "request", return_value=Response(b"".join(events))):
            api.stream_chat("https://example.test", {}, threading.Event(), chunks.append)
        self.assertEqual(chunks, ["你好"])

    def test_stream_rejects_truncated_response_and_in_band_error(self):
        for body in (b'data: {"choices":[]}\n\n', b'data: {"error":"provider failed"}\n\n'):
            with self.subTest(body=body), patch.object(api, "request", return_value=Response(body)):
                with self.assertRaises(api.APIError):
                    api.stream_chat("https://example.test", {}, threading.Event(), lambda _: None)

    def test_cancellation_before_and_during_stream(self):
        cancel = threading.Event()
        cancel.set()
        with patch.object(api, "request") as request:
            api.stream_chat("https://example.test", {}, cancel, lambda _: None)
            request.assert_not_called()
        cancel.clear()
        chunks = []

        def emit(text):
            chunks.append(text)
            cancel.set()

        body = b'data: {"choices":[{"delta":{"content":"one"}}]}\n\ndata: {"choices":[{"delta":{"content":"two"}}]}\n\n'
        with patch.object(api, "request", return_value=Response(body)):
            api.stream_chat("https://example.test", {}, cancel, emit)
        self.assertEqual(chunks, ["one"])

    def test_catalog_only_offers_live_chat_providers(self):
        data = {"providers": [{"provider": "a", "status": "live"}, {"provider": "b", "status": "error"}]}
        with patch.object(api, "json_request", return_value=data) as request:
            self.assertEqual(api.available_providers("owner/model"), ["a"])
            self.assertIn("/v1/models/owner/model", request.call_args.args[0])
        with patch.object(api, "json_request", side_effect=api.APIError("missing", 404)):
            self.assertEqual(api.available_providers("owner/model"), [])
        with patch.object(api, "json_request", side_effect=api.APIError("forbidden", 403)):
            with self.assertRaises(api.APIError):
                api.available_providers("owner/model")

    def test_http_auth_errors_redact_token(self):
        error = urllib.error.HTTPError("https://example.test", 403, "Forbidden", {}, io.BytesIO(b"hf_secret123 denied"))
        with patch.dict(os.environ, {"HF_TOKEN": "hf_secret123"}), patch.object(urllib.request, "build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(api.APIError) as caught:
                api.request("https://example.test", {"x": 1})
            self.assertEqual(caught.exception.status, 403)
            self.assertNotIn("hf_secret123", str(caught.exception))
            req = opener.return_value.open.call_args.args[0]
            self.assertEqual(req.get_header("Authorization"), "Bearer hf_secret123")

    def test_redirect_drops_credentials_across_hosts(self):
        req = urllib.request.Request("https://huggingface.co/file", headers={
            "Authorization": "Bearer hf_secret", "X-HF-Browser-Key": "deployment-key"})
        result = api.CredentialRedirect().redirect_request(req, None, 302, "Found", {}, "https://cdn.example/file")
        self.assertIsNone(result.get_header("Authorization"))
        self.assertIsNone(result.get_header("X-hf-browser-key"))

    def test_payload_validation_and_system_prompt(self):
        payload = api.chat_payload("model:provider", "Hello", "Be brief", 0, 64)
        self.assertEqual([m["role"] for m in payload["messages"]], ["system", "user"])
        for prompt, temp, limit in (("", 1, 1), ("ok", 3, 1), ("ok", 1, 0), ("ok", float("nan"), 1)):
            with self.assertRaises(ValueError):
                api.chat_payload("model", prompt, temperature=temp, max_tokens=limit)


if __name__ == "__main__":
    unittest.main()
