# hf-browser

A tiny, zero-dependency terminal UI for browsing and testing [Hugging Face](https://huggingface.co) models. Search, inspect models, stream text through Inference Providers, or deploy a model-serving Space — without leaving your terminal.

> Unofficial tool, not affiliated with Hugging Face. For the official CLI, see [`huggingface_hub`](https://github.com/huggingface/huggingface_hub).

<!-- TODO: demo GIF (record with https://github.com/charmbracelet/vhs) -->

## Try it instantly

```sh
uvx hf-browser
```

## Install

```sh
pipx install hf-browser   # or: pip install hf-browser
```

Pure stdlib (`curses` + `urllib`) — no dependencies on macOS/Linux. On Windows, `windows-curses` is pulled in automatically.

## Usage

```sh
hf-browser
```

Opens on the top models by downloads.

| Key | Action |
| --- | --- |
| `/` | search (Enter to run, Esc to cancel) |
| `s` | cycle sort: downloads → likes → recently updated → trending |
| `f` | toggle text-generation model filter |
| `j`/`k`, arrows, `g`/`G`, PgUp/PgDn | navigate |
| `Enter` | model details: task, library, license, architecture, parameters, tags, files |
| `o` | (in details) open the model page in your browser |
| `t` | (in details) test through an available Inference Provider |
| `d` | (in details) deploy a model-serving Docker Space |
| `x` | (in details) connect to an existing hf-browser Space |
| `r` | refresh |
| `q` / `Esc` | back / quit |

## Test text models

Set `HF_TOKEN` in your environment before launching the app. Create a token at
[Hugging Face settings](https://huggingface.co/settings/tokens) with **Inference
Providers** permission for provider testing. Space deployment also requires write
access to the target namespace. Gated models require accepted access terms.

Select a `text-generation` model, press `Enter`, then `t`. The app discovers live
chat-capable providers from the [router catalog](https://huggingface.co/docs/inference-providers/hub-api).
The first release supports text chat/generation; translation, summarization and
other `text2text-generation` tasks are not supported yet.

Inside the testing screen:

| Key | Action |
| --- | --- |
| `p` | edit a multiline prompt; Enter inserts a newline, Ctrl-D saves, Esc cancels |
| `s` | edit an optional system prompt |
| `v` | cycle available providers |
| `c` | edit temperature and maximum output tokens, e.g. `0.7,256` |
| `Enter` | send a request, or retry using the same prompt and settings |
| `j`/`k`, arrows, PgUp/PgDn | scroll the streamed response |
| `G` | jump to the bottom and follow new output |
| `Esc` / `q` | cancel an active request; press again to return |

Each run is independent, with no conversation history sent automatically.
Provider testing uses the [Hugging Face chat completion API](https://huggingface.co/docs/inference-providers/tasks/chat-completion)
and may consume credits or incur charges. Cancellation returns control immediately;
a backend may finish its current request and charge for it. No prompts or tokens
are saved locally.

## Deploy a Space

From model details press `d`, supply `username/space-name`, choose hardware and
visibility, then review and confirm. The default visibility is private. The Space
hosts a Docker inference API; **the interactive UI remains in this TUI**. No model
weights or inference libraries are installed on your local machine.

The template supports standard Transformers causal language models with
Safetensors weights. It pins the model's immutable revision and disables remote
code. Quantized models, adapters, multimodal models and custom model code are not
supported. Models newer than the template's pinned Transformers version may fail
to load. CPU inference is intended for small models; select sufficient RAM/VRAM
for the weights and generation cache. The server uses float32 on CPU and float16
on GPU, without quantization or multi-GPU sharding.

Builds and downloads can take minutes. The TUI shows build/startup status and
opens the same testing screen when the Space is running. Press `q` to leave
startup monitoring; the Space continues running. Use `x` to reconnect later.
An existing Space with the same hf-browser template/model/revision is reused;
other repositories are never overwritten. Reuse leaves hardware and visibility
unchanged. Upload failures report the created repository so you can inspect or
remove it before trying a fresh deployment.

**Hardware can incur ongoing charges. Leaving the TUI does not pause or delete a
Space.** Use its Hugging Face settings to pause, restart, change hardware, or
delete it. Check [current Space pricing](https://huggingface.co/pricing#spaces)
and account eligibility before deployment.

The server requires a deployment API key stored as a Space secret. This key is
derived from your `HF_TOKEN` and Space ID; reconnect using the same token. If you
rotate that token, update the Space's `API_KEY` secret using
`hf_browser.spaces.deployment_key(space_id)`, or create a new deployment.
For gated models only, deployment also stores `HF_TOKEN` as a Space secret to
download weights; this transfer is shown on the review screen. Repository files
contain no credentials. Use a token with the minimum necessary permissions.

## Development

```sh
PYTHONPATH=src python -m hf_browser
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests use simulated HTTP responses and model serving; they do not consume
inference credits or create live Spaces. Live testing needs a Hugging Face token
and an eligible account.

## License

MIT
