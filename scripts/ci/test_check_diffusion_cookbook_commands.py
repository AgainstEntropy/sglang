"""Unit tests for the command extraction in check_diffusion_cookbook_commands.py.

Run with:
    python3 -m unittest discover -s scripts/ci -p 'test_check_diffusion_cookbook_commands.py'
"""

import argparse
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_diffusion_cookbook_commands as checker  # noqa: E402


def _commands(script: str):
    return checker.parse_shell(textwrap.dedent(script).strip().splitlines())


class TestParseShell(unittest.TestCase):
    def test_continuations_env_prefix_and_redirections(self):
        (cmd,) = _commands(
            """
            # comment
            SGLANG_CACHE_DIT_ENABLED=true CUDA_VISIBLE_DEVICES=0,1 sglang serve \\
              --model-path Qwen/Qwen-Image \\
              --port 30000 > server.log 2>&1 &
            """
        )
        line, subcommand, argv, env = cmd
        self.assertEqual(line, 2)
        self.assertEqual(subcommand, "serve")
        self.assertEqual(argv, ["--model-path", "Qwen/Qwen-Image", "--port", "30000"])
        self.assertEqual(
            env, {"SGLANG_CACHE_DIT_ENABLED": "true", "CUDA_VISIBLE_DEVICES": "0,1"}
        )

    def test_variables_and_arrays(self):
        (cmd,) = _commands(
            """
            REPO=org/lora
            EXTRA=()
            # EXTRA=(--lora-alpha 8)
            SERVER_ARGS=(
              --model-path Wan-AI/Wan2.1-T2V-14B-Diffusers
              --num-gpus 4  # inline comment
            )
            SAMPLING=(--prompt "A curious raccoon" --save-output)
            sglang generate "${SERVER_ARGS[@]}" "${SAMPLING[@]}" --lora-path "$REPO" "${EXTRA[@]}"
            """
        )
        self.assertEqual(
            cmd[2],
            [
                "--model-path",
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "--num-gpus",
                "4",
                "--prompt",
                "A curious raccoon",
                "--save-output",
                "--lora-path",
                "org/lora",
            ],
        )

    def test_exported_env_and_chained_commands(self):
        commands = _commands(
            """
            export SGLANG_DIFFUSION_PLATFORM_OVERRIDE=cpu
            pip install foo && sglang serve --model-path a/b; echo done
            """
        )
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][2], ["--model-path", "a/b"])
        self.assertEqual(commands[0][3], {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "cpu"})

    def test_placeholders_survive_tokenizing(self):
        (cmd,) = _commands(
            """
            PUP_PATH="<local-path>"
            sglang serve ${PUP_PATH} --image-path <YOUR-IMAGE> \\
              --node-rank {{NODE_RANK}} --model-path "$UNSET_VAR"
            """
        )
        self.assertEqual(
            cmd[2],
            [
                "<local-path>",
                "--image-path",
                "<YOUR-IMAGE>",
                "--node-rank",
                "{{NODE_RANK}}",
                "--model-path",
                "$UNSET_VAR",
            ],
        )

    def test_heredocs(self):
        files = {}
        commands = checker.parse_shell(
            textwrap.dedent(
                """
                MODEL=$(python - <<'PY'
                print("ignored")
                PY
                )
                cat > run.json <<'EOF'
                {"seed": 1}
                EOF
                sglang generate --model-path "$MODEL" --config run.json
                """
            )
            .strip()
            .splitlines(),
            files=files,
        )
        self.assertEqual(files, {"run.json": '{"seed": 1}\n'})
        self.assertEqual(
            commands[0][2], ["--model-path", "$MODEL", "--config", "run.json"]
        )


class TestExtractPage(unittest.TestCase):
    def test_page_context_files_and_default_model(self):
        page = textwrap.dedent(
            """
            ## Serve

            ```yaml run.yaml
            prompt: hi
            ```

            ```bash Command
            sglang generate --model-path org/model --config run.yaml
            ```

            <Tabs>
              <Tab title="Ascend A3 Series">
                ```bash
                sglang serve --model-path org/model --num-gpus 2
                ```
              </Tab>
            </Tabs>

            The Xeon command:

            ```bash
            sglang serve --model-path "<MODEL>"
            ```

            ```python
            subprocess.run("sglang serve --model-path x/y")
            ```
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "page.mdx")
            path.write_text(page)
            commands = checker.extract_page(path, root=Path(tmp))
        self.assertEqual(
            [c.source for c in commands], ["page.mdx:9", "page.mdx:15", "page.mdx:23"]
        )
        self.assertEqual([c.hw for c in commands], ["h100", "a3", "xeon"])
        self.assertEqual(commands[0].files, {"run.yaml": "prompt: hi\n"})
        self.assertEqual(commands[2].default_model, "org/model")

    def test_generated_entry(self):
        entry = {
            "source": "docs/x.mdx#hw=h200",
            "hw": "h200",
            "vendor": "hopper",
            "model": "org/model",
            "command": "# Multi-node (2 nodes).\nSGLANG_X=1 \\\nsglang serve \\\n  --model-path org/model \\\n  --node-rank 0",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "gen.json")
            path.write_text(json.dumps([entry]))
            (command,) = checker.load_generated(path)
        self.assertEqual(
            command.argv, ["--model-path", "org/model", "--node-rank", "0"]
        )
        self.assertEqual(command.env, {"SGLANG_X": "1"})
        self.assertEqual(command.hw, "h200")


class TestSubstitution(unittest.TestCase):
    def test_typed_dummies(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path")
        parser.add_argument("--node-rank", type=int)
        parser.add_argument("--dist-init-addr")
        parser.add_argument("--image-path")
        parser.add_argument("--num-frames", type=int)
        parser.add_argument("--transformer-weights-path")
        with tempfile.TemporaryDirectory() as workdir:
            argv, _ = checker._substitute_placeholders(
                argv=[
                    "--model-path",
                    "$MODEL",
                    "--node-rank",
                    "{{NODE_RANK}}",
                    "--dist-init-addr",
                    "{{NODE0_IP}}:20000",
                    "--image-path=<YOUR-IMAGE-PATH>",
                    "--num-frames",
                    "<N>",
                    "--transformer-weights-path",
                    "$DIR/transformer",
                ],
                parser=parser,
                workdir=workdir,
                default_model="org/model",
            )
            self.assertTrue(os.path.exists(argv[6].split("=", 1)[1]))
        self.assertEqual(
            argv[:6],
            [
                "--model-path",
                "org/model",
                "--node-rank",
                "0",
                "--dist-init-addr",
                "127.0.0.1:20000",
            ],
        )
        self.assertTrue(argv[6].endswith("YOUR_IMAGE_PATH.png"))
        self.assertEqual(argv[7:9], ["--num-frames", "1"])
        self.assertEqual(argv[10], f"{workdir}/transformer")


class TestFetchModelIndex(unittest.TestCase):
    def test_fetches_only_named_json_files(self):
        requested = []
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp, "model_index.json")
            index.write_text(
                json.dumps(
                    {
                        "_class_name": "WanPipeline",
                        "transformer": ["diffusers", "WanTransformer3DModel"],
                        "scheduler": ["diffusers", "UniPCMultistepScheduler"],
                        "tokenizer": ["transformers", "T5TokenizerFast"],
                        "transformer_2": [None, None],
                    }
                )
            )

            def fake_download(*, repo_id, filename):
                requested.append((repo_id, filename))
                if repo_id == "org/diffusers" and filename == "model_index.json":
                    return str(index)
                raise FileNotFoundError(filename)

            commands = [
                checker.Command(
                    source="a",
                    kind="static",
                    subcommand="serve",
                    argv=["--model-path", "org/diffusers"],
                ),
                checker.Command(
                    source="b",
                    kind="static",
                    subcommand="serve",
                    argv=["--model-path", "org/package", "--model-variant", "fp8r"],
                ),
                checker.Command(
                    source="c",
                    kind="static",
                    subcommand="serve",
                    argv=["--model-path", "<local>"],
                ),
            ]
            checker.fetch_model_indices(commands, hf_hub_download=fake_download)
        self.assertEqual(
            requested,
            [
                ("org/diffusers", "model_index.json"),
                ("org/diffusers", "transformer/config.json"),
                ("org/diffusers", "scheduler/scheduler_config.json"),
                ("org/package", "model_index.json"),
                ("org/package", "config.json"),
                ("org/package", "manifest.json"),
                ("org/package", "hf_quant_config.json"),
                ("org/package", "variants/fp8r/config.json"),
                ("org/package", "variants/fp8r/manifest.json"),
                ("org/package", "variants/fp8r/hf_quant_config.json"),
            ],
        )
        self.assertTrue(all(name.endswith(".json") for _, name in requested))


class TestWorkerEnv(unittest.TestCase):
    def test_workers_see_no_gpu_and_no_online_hub(self):
        env = checker._worker_env(checker.HARDWARE["h100"])
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["SGLANG_USE_MODELSCOPE"], "false")

    def test_page_env_cannot_reenable_downloads_or_gpus(self):
        env = checker._command_env(
            {
                "SGLANG_USE_MODELSCOPE": "true",
                "HF_ENDPOINT": "https://example.com",
                "HF_HUB_OFFLINE": "0",
                "CUDA_VISIBLE_DEVICES": "0,1",
                "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "npu",
                "SGLANG_CACHE_DIT_ENABLED": "true",
            }
        )
        self.assertEqual(env, {"SGLANG_CACHE_DIT_ENABLED": "true"})


if __name__ == "__main__":
    unittest.main()
