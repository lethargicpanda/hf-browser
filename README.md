# hf-browser

A tiny, zero-dependency terminal UI for browsing [Hugging Face](https://huggingface.co) models. Search, sort, inspect model details, and jump to the model page — without leaving your terminal.

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
| `j`/`k`, arrows, `g`/`G`, PgUp/PgDn | navigate |
| `Enter` | model details: task, library, license, architecture, parameters, tags, files |
| `o` | (in details) open the model page in your browser |
| `r` | refresh |
| `q` / `Esc` | back / quit |

## License

MIT
