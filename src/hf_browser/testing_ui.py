"""Interactive text testing and deployment screens (all curses calls on UI thread)."""

import curses
import queue
import threading
import time
import textwrap

from .inference import APIError, ROUTER, available_providers, chat_payload, redact, stream_chat, token
from .spaces import HARDWARE, deploy, deployment_key, deployment_spec, space_endpoint, space_info, validate_repo


class Job:
    """A screen owns its worker and queue; cancelled screens cannot receive stale results."""
    def __init__(self, work):
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.done = False
        self.result = None
        self.error = ""

        def run():
            try:
                result = work(self.cancel, lambda text: self.events.put(("text", text)))
                self.events.put(("done", result))
            except Exception as exc:
                self.events.put(("error", redact(str(exc))))

        threading.Thread(target=run, daemon=True).start()

    def drain(self):
        chunks = []
        while True:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "text":
                chunks.append(value)
            else:
                self.done = True
                if kind == "error":
                    self.error = value
                else:
                    self.result = value
        return chunks


def draw(app, title, lines, footer, offset=0):
    from . import safe_addstr
    scr = app.scr
    scr.erase()
    height, width = scr.getmaxyx()
    safe_addstr(scr, 0, 1, title, curses.A_BOLD | curses.color_pair(1))
    wrapped = []
    for line in (part for value in lines for part in str(value).split("\n")):
        # Strip terminal controls received from models/API errors before rendering.
        line = "".join(c for c in str(line) if c.isprintable() or c == "\t")
        wrapped.extend(textwrap.wrap(line.expandtabs(4), max(1, width - 4)) or [""])
    rows = max(1, height - 4)
    offset = max(0, min(offset, max(0, len(wrapped) - rows)))
    for i, line in enumerate(wrapped[offset:offset + rows]):
        safe_addstr(scr, i + 2, 2, line)
    safe_addstr(scr, height - 1, 1, footer, curses.A_DIM)
    scr.refresh()
    return offset, len(wrapped), rows


def message(app, title, text):
    app.scr.timeout(-1)
    offset = 0
    while True:
        offset, _, rows = draw(app, title, text.splitlines(), "j/k:scroll  q/esc:back", offset)
        ch = app.scr.getch()
        if ch in (27, ord("q"), 10, 13):
            return
        offset = scroll(ch, offset, rows)


def scroll(ch, offset, rows):
    if ch in (ord("j"), curses.KEY_DOWN):
        return offset + 1
    if ch in (ord("k"), curses.KEY_UP):
        return max(0, offset - 1)
    if ch == curses.KEY_NPAGE:
        return offset + rows
    if ch == curses.KEY_PPAGE:
        return max(0, offset - rows)
    return offset


def wait_job(app, title, job, cancellable=True):
    lines = []
    app.scr.timeout(100)
    try:
        while True:
            lines.extend(job.drain())
            draw(app, title, lines + (["Working…"] if not job.done else []),
                 "q/esc:cancel" if cancellable else "Creating remote resources; please wait")
            if job.done:
                if job.error:
                    raise APIError(job.error)
                return job.result
            ch = app.scr.getch()
            if cancellable and ch in (27, ord("q")):
                job.cancel.set()
                return None
    finally:
        app.scr.timeout(-1)


def edit(app, title, value, multiline=False):
    """Append/backspace editor with Unicode, newline input and horizontal/vertical wrapping."""
    original = value
    app.scr.timeout(-1)
    curses.curs_set(1)
    try:
        while True:
            scr = app.scr
            height, width = scr.getmaxyx()
            content = []
            for line in value.split("\n"):
                content.extend(textwrap.wrap(line, max(1, width - 4), replace_whitespace=False,
                                             drop_whitespace=False) or [""])
            rows = max(1, height - 4)
            draw(app, title, content[-rows:], "Ctrl-D:save  Enter:newline  Esc:cancel" if multiline
                 else "Enter:save  Esc:cancel")
            try:
                scr.move(min(height - 2, 2 + min(rows, len(content)) - 1),
                         min(max(0, width - 2), 2 + len(content[-1])))
            except curses.error:
                pass
            ch = scr.get_wch()
            if ch == "\x1b":
                return original
            if ch == "\x04" or (not multiline and ch in ("\n", "\r")):
                return value
            if ch in ("\x7f", "\b", curses.KEY_BACKSPACE):
                value = value[:-1]
            elif ch in ("\n", "\r") and multiline:
                value += "\n"
            elif isinstance(ch, str) and ch.isprintable() and len(value) < 16000:
                value += ch
    finally:
        curses.curs_set(0)


def choose(app, title, options):
    app.scr.timeout(-1)
    selected = 0
    while True:
        draw(app, title, [("→ " if i == selected else "  ") + label
                         for i, label in enumerate(options)], "j/k:select  Enter:choose  Esc:cancel")
        ch = app.scr.getch()
        if ch in (27, ord("q")):
            return None
        if ch in (ord("j"), curses.KEY_DOWN):
            selected = min(len(options) - 1, selected + 1)
        elif ch in (ord("k"), curses.KEY_UP):
            selected = max(0, selected - 1)
        elif ch in (10, 13, curses.KEY_ENTER):
            return selected


def deploy_screen(app, detail):
    token()
    spec = wait_job(app, "Checking Space compatibility", Job(lambda cancel, emit: deployment_spec(detail)))
    if spec is None:
        return None
    repo_id = edit(app, "Space ID (username/name)", "")
    if not repo_id:
        return None
    validate_repo(repo_id)
    hardware_idx = choose(app, "Space hardware — verify current pricing in Hugging Face settings", [
        "cpu-basic: suitable only for small models; subscription may be required",
        "t4-small: paid GPU, 16 GB VRAM", "a10g-small: paid GPU, 24 GB VRAM",
        "a100-large: paid GPU, 80 GB VRAM"])
    if hardware_idx is None:
        return None
    visibility = choose(app, "Space visibility", ["Private (default)", "Public — code and model identity are visible"])
    if visibility is None:
        return None
    hardware = HARDWARE[hardware_idx]
    lines = [f"Space: {repo_id}", f"Model: {spec['model']}", f"Revision: {spec['revision']}",
             f"Hardware: {hardware}; visibility: {'private' if visibility == 0 else 'public'}",
             "Creates a Docker API server. The UI stays here in the terminal.",
             "GPU hardware incurs charges while running. Check current pricing:",
             "https://huggingface.co/pricing#spaces", "Builds and model downloads can take several minutes.",
             "Large models may exceed RAM/VRAM. Remote code and quantized weights are unsupported.",
             "Existing matching deployments are reused without changing hardware or visibility."]
    if detail.get("gated"):
        lines.append("Gated model: HF_TOKEN will be stored as a Space secret to download weights. The token must have model access.")
    review_offset = 0
    while True:
        review_offset, _, rows = draw(app, "Review deployment", lines,
                                     "j/k:scroll  Enter:create/reuse Space  Esc:cancel", review_offset)
        ch = app.scr.getch()
        if ch in (27, ord("q")):
            return None
        if ch in (10, 13, curses.KEY_ENTER):
            break
        review_offset = scroll(ch, review_offset, rows)
    job = Job(lambda cancel, emit: deploy(repo_id, spec, hardware, visibility == 0,
                                         bool(detail.get("gated")), emit))
    wait_job(app, "Deploying " + repo_id, job, cancellable=False)
    # Remember the repository immediately, even if the user leaves startup monitoring.
    app.deployments[detail["id"]] = repo_id
    return monitor_space(app, repo_id)


def monitor_space(app, repo_id):
    def poll(cancel, emit):
        previous = None
        while not cancel.is_set():
            info = space_info(repo_id)
            runtime = info.get("runtime") or {}
            stage = runtime.get("stage", "UNKNOWN")
            if stage != previous:
                emit(f"{repo_id}: {stage}")
                previous = stage
            if stage == "RUNNING":
                return (repo_id, space_endpoint(info))
            if stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "PAUSED", "STOPPED", "SLEEPING"):
                raise APIError(f"Space is {stage}. Open https://huggingface.co/spaces/{repo_id} for logs or restart, then retry.")
            cancel.wait(5)
        return None
    return wait_job(app, "Space startup — leaving this screen does not stop billing", Job(poll))


def test_screen(app, detail, space=None):
    if detail.get("pipeline_tag") != "text-generation":
        message(app, "Text testing", "This release supports text-generation models. Translation and other text2text tasks are not supported yet.")
        return
    token()
    providers = []
    if space is None:
        providers = wait_job(app, "Finding live text providers", Job(lambda cancel, emit: available_providers(detail["id"])))
        if providers is None:
            return
    if not providers and space is None:
        existing = app.deployments.get(detail["id"])
        if existing:
            space = monitor_space(app, existing)
        if space is None:
            message(app, "No live text provider", "No provider currently serves this model. Use d from model details to deploy a compatible Transformers model to a Space.")
            return
    provider_idx = 0
    prompt, system = "", ""
    temperature, max_tokens = 0.7, 256
    output, status = "", "Enter a prompt to begin. Each run is an independent request."
    offset = 0
    job = None
    follow = True
    started = 0
    app.scr.timeout(100)
    try:
        while True:
            app.scr.timeout(100)
            if job is not None:
                chunks = job.drain()
                output += "".join(chunks)
                if chunks and follow:
                    offset = 10**9
                if job.done:
                    status = "error: " + job.error if job.error else f"Completed in {time.monotonic() - started:.1f}s"
                    job = None
            backend = f"Space: {space[0]}" if space else "Provider: " + providers[provider_idx]
            lines = [backend, f"Temperature: {temperature:g}    Max tokens: {max_tokens}",
                     "Prompt: " + (prompt[:200] + ("…" if len(prompt) > 200 else "") or "(empty)"),
                     "System: " + (system[:200] + ("…" if len(system) > 200 else "") or "(none)"),
                     "", status, "", "Response:"] + output.splitlines()
            offset, _, rows = draw(app, "Test " + detail["id"], lines,
                                  "j/k:scroll  G:follow  Esc:cancel/back" if job else "p:prompt  s:system  v:provider  c:settings  Enter:run  j/k:scroll  q:back", offset)
            ch = app.scr.getch()
            if ch in (27, ord("q")):
                if job:
                    job.cancel.set()
                    job = None
                    status = "Cancelled locally. The backend may finish the current request and charge for it."
                else:
                    return
            elif job is None and ch == ord("p"):
                prompt = edit(app, "Prompt", prompt, multiline=True)
                offset = 0
            elif job is None and ch == ord("s"):
                system = edit(app, "System prompt (optional)", system, multiline=True)
                offset = 0
            elif job is None and ch == ord("v") and not space:
                provider_idx = (provider_idx + 1) % len(providers)
                status = "Provider changed. Requests may incur inference charges."
            elif job is None and ch == ord("c"):
                value = edit(app, "Temperature (0–2), maximum tokens (1–4096)", f"{temperature:g},{max_tokens}")
                try:
                    temp, limit = value.split(",")
                    temp, limit = float(temp), int(limit)
                    chat_payload(detail["id"], "validate", temperature=temp, max_tokens=limit)
                    temperature, max_tokens = temp, limit
                except ValueError as exc:
                    status = "error: " + str(exc)
            elif job is None and ch in (10, 13, curses.KEY_ENTER):
                try:
                    model = detail["id"] if space else detail["id"] + ":" + providers[provider_idx]
                    payload = chat_payload(model, prompt, system, temperature, max_tokens)
                    url = space[1] if space else ROUTER
                    headers = {"X-HF-Browser-Key": deployment_key(space[0])} if space else None
                    output = ""
                    offset = 0
                    follow = True
                    started = time.monotonic()
                    status = "Generating… Esc cancels. Inference may incur charges."
                    job = Job(lambda cancel, emit: stream_chat(url, payload, cancel, emit, headers))
                except (ValueError, APIError) as exc:
                    status = "error: " + str(exc)
            else:
                if ch == ord("G"):
                    follow = True
                    offset = 10**9
                else:
                    next_offset = scroll(ch, offset, rows)
                    if next_offset != offset:
                        follow = False
                    offset = next_offset
    finally:
        if job:
            job.cancel.set()
        app.scr.timeout(-1)


def handle_action(app, detail, action):
    try:
        if action == "deploy":
            space = deploy_screen(app, detail)
            if space:
                test_screen(app, detail, space)
        elif action == "space":
            repo_id = app.deployments.get(detail["id"])
            if not repo_id:
                repo_id = edit(app, "Existing hf-browser Space ID", "")
            if repo_id:
                validate_repo(repo_id)
                info = wait_job(app, "Checking existing Space", Job(lambda cancel, emit: space_info(repo_id)))
                if info is None:
                    return
                spec = (info.get("cardData") or {}).get("hf_browser") or {}
                if spec.get("model") != detail["id"] or spec.get("version") != 1:
                    raise APIError("This Space is not a compatible hf-browser deployment of the selected model.")
                space = monitor_space(app, repo_id)
                if space:
                    app.deployments[detail["id"]] = repo_id
                    test_screen(app, detail, space)
        else:
            test_screen(app, detail)
    except (APIError, ValueError) as exc:
        message(app, "Cannot test/deploy model", redact(str(exc)))
