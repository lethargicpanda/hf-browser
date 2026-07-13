#!/usr/bin/env python3
"""hf-browser: a tiny TUI to browse Hugging Face models.

Zero dependencies (stdlib curses + urllib). Usage: hf-browser

Keys:
  /          edit search query
  s          cycle sort (downloads / likes / lastModified / trending)
  j/k, arrows  move selection
  g/G        jump to top / bottom
  enter      show model details
  r          refresh
  q / esc    back or quit
"""

import curses
import json
import textwrap
import urllib.parse
import urllib.request
import webbrowser

__version__ = "0.1.0"

API = "https://huggingface.co/api"
SORTS = ["downloads", "likes", "lastModified", "trendingScore"]
SORT_LABELS = {"downloads": "downloads", "likes": "likes",
               "lastModified": "recently updated", "trendingScore": "trending"}
LIMIT = 50


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "hf-browser/0.1"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp)


def fetch_models(query, sort):
    params = {"limit": str(LIMIT), "sort": sort, "direction": "-1"}
    if query:
        params["search"] = query
    url = f"{API}/models?{urllib.parse.urlencode(params)}"
    return fetch_json(url)


def fetch_model_detail(model_id):
    return fetch_json(f"{API}/models/{urllib.parse.quote(model_id, safe='/')}")


def human_count(n):
    if n is None:
        return "-"
    for div, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if n >= div:
            return f"{n / div:.1f}{suffix}"
    return str(n)


def safe_addstr(win, y, x, text, attr=0):
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    try:
        win.addstr(y, x, text[: w - x - 1], attr)
    except curses.error:
        pass


class App:
    def __init__(self, stdscr):
        self.scr = stdscr
        self.query = ""
        self.sort_idx = 0
        self.models = []
        self.selected = 0
        self.top = 0
        self.status = ""
        curses.curs_set(0)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_YELLOW, -1)   # header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # selection
        curses.init_pair(3, curses.COLOR_CYAN, -1)     # accents
        curses.init_pair(4, curses.COLOR_RED, -1)      # errors

    @property
    def sort(self):
        return SORTS[self.sort_idx]

    # ---------- data ----------

    def load(self):
        self.draw_list(loading=True)
        try:
            self.models = fetch_models(self.query, self.sort)
            self.status = f"{len(self.models)} models"
        except Exception as e:
            self.models = []
            self.status = f"error: {e}"
        self.selected = 0
        self.top = 0

    # ---------- drawing ----------

    def draw_header(self):
        h, w = self.scr.getmaxyx()
        title = " hf-browser "
        sort_label = SORT_LABELS[self.sort]
        query = self.query or "(top models)"
        safe_addstr(self.scr, 0, 0, title, curses.color_pair(1) | curses.A_BOLD)
        safe_addstr(self.scr, 0, len(title) + 1,
                    f"search: {query}   sort: {sort_label}", curses.color_pair(3))
        safe_addstr(self.scr, 1, 0, "─" * (w - 1))

    def draw_footer(self, keys):
        h, w = self.scr.getmaxyx()
        safe_addstr(self.scr, h - 2, 0, "─" * (w - 1))
        attr = curses.color_pair(4) if self.status.startswith("error") else curses.A_DIM
        safe_addstr(self.scr, h - 1, 0, f" {self.status}", attr)
        safe_addstr(self.scr, h - 1, max(0, w - len(keys) - 2), keys, curses.A_DIM)

    def draw_list(self, loading=False):
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        self.draw_header()
        rows = h - 4  # header 2 lines, footer 2 lines
        if loading:
            safe_addstr(self.scr, 3, 2, "Loading…", curses.A_DIM)
            self.draw_footer("")
            self.scr.refresh()
            return
        if not self.models:
            safe_addstr(self.scr, 3, 2, "No results.", curses.A_DIM)
        if self.selected < self.top:
            self.top = self.selected
        elif self.selected >= self.top + rows:
            self.top = self.selected - rows + 1
        for i in range(self.top, min(len(self.models), self.top + rows)):
            m = self.models[i]
            y = 2 + i - self.top
            dl = human_count(m.get("downloads"))
            likes = human_count(m.get("likes"))
            task = m.get("pipeline_tag") or ""
            stats = f"{dl:>7} dl  {likes:>6} ♥  {task}"
            line_attr = curses.color_pair(2) if i == self.selected else 0
            if i == self.selected:
                safe_addstr(self.scr, y, 0, " " * (w - 1), line_attr)
            safe_addstr(self.scr, y, 1, m.get("id", "?"), line_attr | curses.A_BOLD)
            safe_addstr(self.scr, y, max(2, w - len(stats) - 2), stats, line_attr)
        self.draw_footer("/:search  s:sort  enter:details  r:refresh  q:quit ")
        self.scr.refresh()

    # ---------- search input ----------

    def edit_query(self):
        curses.curs_set(1)
        buf = self.query
        while True:
            h, w = self.scr.getmaxyx()
            safe_addstr(self.scr, h - 1, 0, " " * (w - 1))
            safe_addstr(self.scr, h - 1, 0, f" search: {buf}", curses.color_pair(3))
            self.scr.move(h - 1, min(w - 2, 9 + len(buf)))
            self.scr.refresh()
            ch = self.scr.get_wch()
            if ch in ("\n", "\r"):
                curses.curs_set(0)
                return buf.strip()
            if ch == "\x1b":  # esc: cancel
                curses.curs_set(0)
                return self.query
            if ch in ("\x7f", "\b", curses.KEY_BACKSPACE):
                buf = buf[:-1]
            elif isinstance(ch, str) and ch.isprintable():
                buf += ch

    # ---------- details view ----------

    def show_detail(self, model_id):
        self.scr.erase()
        self.draw_header()
        safe_addstr(self.scr, 3, 2, f"Loading {model_id}…", curses.A_DIM)
        self.scr.refresh()
        try:
            d = fetch_model_detail(model_id)
        except Exception as e:
            self.status = f"error: {e}"
            return
        lines = self.detail_lines(d)
        offset = 0
        while True:
            self.scr.erase()
            h, w = self.scr.getmaxyx()
            self.draw_header()
            rows = h - 4
            offset = max(0, min(offset, max(0, len(lines) - rows)))
            for i in range(offset, min(len(lines), offset + rows)):
                text, attr = lines[i]
                safe_addstr(self.scr, 2 + i - offset, 2, text, attr)
            self.status = model_id
            self.draw_footer("j/k:scroll  o:open URL  q/esc:back ")
            self.scr.refresh()
            ch = self.scr.getch()
            if ch in (ord("q"), 27):
                return
            elif ch in (ord("j"), curses.KEY_DOWN):
                offset += 1
            elif ch in (ord("k"), curses.KEY_UP):
                offset -= 1
            elif ch == curses.KEY_NPAGE:
                offset += rows
            elif ch == curses.KEY_PPAGE:
                offset -= rows
            elif ch == ord("o"):
                webbrowser.open(f"https://huggingface.co/{model_id}")

    def detail_lines(self, d):
        bold = curses.A_BOLD
        cyan = curses.color_pair(3)
        w = max(40, self.scr.getmaxyx()[1] - 6)
        lines = [(d.get("id", "?"), bold | curses.color_pair(1)), ("", 0)]

        def field(label, value):
            if value not in (None, "", []):
                lines.append((f"{label:<14}{value}", 0))

        field("task", d.get("pipeline_tag"))
        field("library", d.get("library_name"))
        field("downloads", f'{d.get("downloads", 0):,}')
        field("likes", f'{d.get("likes", 0):,}')
        field("updated", (d.get("lastModified") or "")[:10])
        field("created", (d.get("createdAt") or "")[:10])
        field("license", next((t.split(":", 1)[1] for t in d.get("tags", [])
                               if t.startswith("license:")), None))
        field("gated", d.get("gated") if d.get("gated") else None)
        cfg = d.get("config") or {}
        arch = cfg.get("architectures")
        if arch:
            field("architecture", ", ".join(arch))
        st = (d.get("safetensors") or {}).get("total")
        if st:
            field("parameters", human_count(st))
        tags = [t for t in d.get("tags", []) if ":" not in t]
        if tags:
            lines.append(("", 0))
            lines.append(("tags", bold))
            for row in textwrap.wrap("  ".join(tags), w):
                lines.append((row, cyan))
        siblings = d.get("siblings") or []
        if siblings:
            lines.append(("", 0))
            lines.append((f"files ({len(siblings)})", bold))
            for s in siblings[:40]:
                lines.append((f"  {s.get('rfilename')}", 0))
            if len(siblings) > 40:
                lines.append((f"  … and {len(siblings) - 40} more", curses.A_DIM))
        lines.append(("", 0))
        lines.append((f"https://huggingface.co/{d.get('id')}", cyan))
        return lines

    # ---------- main loop ----------

    def run(self):
        self.load()
        while True:
            self.draw_list()
            ch = self.scr.getch()
            if ch == ord("q"):
                return
            elif ch in (ord("j"), curses.KEY_DOWN):
                self.selected = min(len(self.models) - 1, self.selected + 1)
            elif ch in (ord("k"), curses.KEY_UP):
                self.selected = max(0, self.selected - 1)
            elif ch == ord("g"):
                self.selected = 0
            elif ch == ord("G"):
                self.selected = max(0, len(self.models) - 1)
            elif ch == curses.KEY_NPAGE:
                self.selected = min(len(self.models) - 1, self.selected + 10)
            elif ch == curses.KEY_PPAGE:
                self.selected = max(0, self.selected - 10)
            elif ch == ord("/"):
                new_q = self.edit_query()
                if new_q != self.query:
                    self.query = new_q
                    self.load()
            elif ch == ord("s"):
                self.sort_idx = (self.sort_idx + 1) % len(SORTS)
                self.load()
            elif ch == ord("r"):
                self.load()
            elif ch in (curses.KEY_ENTER, 10, 13) and self.models:
                self.show_detail(self.models[self.selected]["id"])
            elif ch == curses.KEY_RESIZE:
                pass  # redraw happens at top of loop


def main():
    curses.wrapper(lambda scr: App(scr).run())


if __name__ == "__main__":
    main()
