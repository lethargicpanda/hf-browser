import json
import os
import unittest
from unittest.mock import patch

from hf_browser import spaces
from hf_browser.inference import APIError

SPEC = {"model": "owner/model", "revision": "a" * 40, "version": 1}


class SpaceTests(unittest.TestCase):
    def test_creation_commits_files_without_secrets(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_secret123"}), patch.object(
                spaces, "space_info", side_effect=APIError("missing", 404)), patch.object(spaces, "json_request") as api:
            self.assertEqual(spaces.deploy("user/test", SPEC, "t4-small", gated=True), "user/test")
            create = api.call_args_list[0].kwargs["payload"]
            self.assertEqual(create["hardware"], "t4-small")
            self.assertEqual(create["visibility"], "private")
            self.assertEqual({s["key"] for s in create["secrets"]}, {"API_KEY", "HF_TOKEN"})
            commit = api.call_args_list[1].kwargs["payload"].decode()
            self.assertNotIn("hf_secret123", commit)
            entries = [json.loads(line) for line in commit.splitlines()]
            self.assertEqual({e["value"]["path"] for e in entries[1:]}, {"README.md", "Dockerfile", "server.py", "model.json"})

    def test_reuse_never_changes_hardware_or_visibility(self):
        with patch.object(spaces, "space_info", return_value={"cardData": {"hf_browser": SPEC}}), patch.object(spaces, "json_request") as api:
            spaces.deploy("user/test", SPEC, "a100-large", private=False)
            api.assert_not_called()

    def test_existing_unrelated_space_and_auth_error_are_not_overwritten(self):
        for result in ({"cardData": {}}, APIError("forbidden", 403)):
            kwargs = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
            with patch.object(spaces, "space_info", **kwargs), patch.object(spaces, "json_request") as api:
                with self.assertRaises(APIError):
                    spaces.deploy("user/test", SPEC)
                api.assert_not_called()

    def test_partial_failure_reports_created_repo(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_secret123"}), patch.object(
                spaces, "space_info", side_effect=APIError("missing", 404)), patch.object(
                spaces, "json_request", side_effect=[{}, APIError("upload failed", 500)]):
            with self.assertRaisesRegex(APIError, "user/test was created but upload failed"):
                spaces.deploy("user/test", SPEC)

    def test_endpoint_only_accepts_https_hf_space_hosts(self):
        self.assertEqual(spaces.space_endpoint({"host": "https://user-test.hf.space"}),
                         "https://user-test.hf.space/v1/chat/completions")
        for host in ("https://hf.space.evil.test", "http://user-test.hf.space", "https://evil.test", "https://user@user-test.hf.space", "https://user-test.hf.space:444"):
            with self.assertRaises(APIError):
                spaces.space_endpoint({"host": host})

    def test_rejects_unsupported_models_before_creating_resources(self):
        detail = {"id": "owner/model", "sha": "a" * 40, "pipeline_tag": "text-generation", "library_name": "transformers"}
        with patch.object(spaces, "json_request", return_value={"architectures": ["LlamaForCausalLM"]}):
            self.assertEqual(spaces.deployment_spec(detail), SPEC)
        for config in ({"auto_map": {"x": "y"}}, {"architectures": ["BertModel"]},
                       {"architectures": ["LlamaForCausalLM"], "quantization_config": {"bits": 4}}):
            with patch.object(spaces, "json_request", return_value=config), self.assertRaises(ValueError):
                spaces.deployment_spec(detail)
        with self.assertRaises(ValueError):
            spaces.deployment_spec(dict(detail, pipeline_tag="text2text-generation"))


if __name__ == "__main__":
    unittest.main()
