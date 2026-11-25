"""Pygame view of the School Director environment.

Layout (drawn on a fixed 1600 x 900 canvas, then scaled to fit the screen):
  - header: term, year, agent
  - the director's decision this term, its reward and the running total
  - the school scene, the school's numbers, and what happened this term
  - one card per child: learning, attendance, confidence, home pressure with
    this term's change, the finish-strong targets, a status and who was helped
  - an 18-term timeline: every decision and its reward
  - at the end: a 6-year report card for every child
"""

from __future__ import annotations

import math

import numpy as np

W, H = 1600, 900  # design canvas; the window is scaled to fit the screen
M = 24            # outer margin

# Colours
BG = (245, 241, 232)
INK = (36, 40, 48)
MUTED = (112, 116, 126)
PANEL = (255, 252, 246)
LINE = (220, 214, 200)
TRACK = (232, 228, 218)
GREEN = (40, 150, 84)
YELLOW = (214, 156, 24)
RED = (200, 64, 54)
BLUE = (52, 112, 196)
PURPLE = (150, 96, 176)
GREY = (150, 150, 150)
GOLD = (250, 196, 40)
WALL = (196, 120, 84)
ROOF = (150, 70, 60)
GRASS = (170, 205, 140)
SKY = (205, 228, 245)

SCHOOL_BARS = [
    ("Budget", "budget", (70, 150, 110)),
    ("Gov funding", "gov_funding", BLUE),
    ("Community trust", "trust", PURPLE),
    ("Tech level", "tech", (40, 160, 170)),
    ("Teacher quality", "quality", (220, 140, 50)),
    ("Teacher morale", "morale", (200, 90, 110)),
]
CHILD_BARS = [  # label, key, colour, higher is better
    ("Learning", "learning", BLUE, True),
    ("Attendance", "attendance", GREEN, True),
    ("Confidence", "confidence", PURPLE, True),
    ("Home pressure", "home_pressure", RED, False),
]


def fmt_value(v: float, target: float | None = None) -> str:
    """Two decimals, but three when rounding would hide that a value misses its
    target (0.549 must not show as 0.55 next to 'below 0.55')."""
    if target is not None and v < target and round(v, 2) >= target:
        return f"{v:.3f}"
    return f"{v:.2f}"


def short_action(name: str) -> tuple[str, str]:
    """Two short lines for a timeline cell."""
    if name.startswith("Mentor "):
        return "Mentor", name[len("Mentor "):]
    if name.startswith("Support "):
        return "Family", name[len("Support "):].replace("'s family", "")
    table = {
        "Hire a teacher": ("Hire", "teacher"), "Train teachers": ("Train", "teachers"),
        "Buy technology": ("Buy", "tech"), "Reach remote areas": ("Reach", "remote"),
        "Engage the community": ("Engage", "community"), "Lobby the government": ("Lobby", "gov't"),
        "Save money": ("Save", "money"),
    }
    return table.get(name, (name, ""))


def child_status(c, cfg, ended: bool) -> tuple[str, tuple]:
    """Status label and colour, based on the finish-strong rule."""
    if not c["enrolled"]:
        return "Dropped out", GREY
    if ended and c["finished_strong"]:
        return "Finished strong", GREEN
    learn_ok = c["learning"] >= cfg["finish_learning"]
    conf_ok = c["confidence"] >= cfg["finish_confidence"]
    if learn_ok and conf_ok:
        return ("Below target" if ended else "On track"), (RED if ended else GREEN)
    close = c["learning"] >= cfg["finish_learning"] - 0.10 and c["confidence"] >= cfg["finish_confidence"] - 0.10
    if ended:
        return "Below target", RED
    if c["attendance"] < 0.35 or c["confidence"] < 0.25:
        return "At risk", RED
    return ("Getting there", YELLOW) if close else ("Needs help", RED)


def event_colour(e: str):
    low = e.lower()
    if any(k in low for k in ("funding cut", "family shock", "dropped out", "teacher left", "could not")):
        return RED
    if "paid off" in low:
        return GREEN
    if "missed class" in low:
        return YELLOW
    return MUTED


def _enable_dpi_awareness():
    """On Windows with display scaling, ask for real pixels so the window is sharp
    and its size is measured correctly."""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class SchoolRenderer:
    def __init__(self, render_mode: str = "human"):
        if render_mode == "human":
            _enable_dpi_awareness()
        import pygame

        self.pg = pygame
        self.mode = render_mode
        pygame.init()
        pygame.font.init()
        self.canvas = pygame.Surface((W, H))
        if render_mode == "human":
            dw, dh = pygame.display.get_desktop_sizes()[0]
            scale = min(1.0, 0.96 * dw / W, 0.86 * dh / H)  # leave room for title bar + taskbar
            self.window = pygame.display.set_mode((int(W * scale), int(H * scale)), pygame.RESIZABLE)
            pygame.display.set_caption("School Director - Mission RL")
        self.clock = pygame.time.Clock()
        name = "segoeui,arial,helvetica"
        F = pygame.font.SysFont
        self.f_title = F(name, 34, bold=True)
        self.f_huge = F(name, 40, bold=True)
        self.f_big = F(name, 26, bold=True)
        self.f_head = F(name, 22, bold=True)
        self.f_med = F(name, 19)
        self.f_bold = F(name, 19, bold=True)
        self.f_small = F(name, 16)
        self.f_sbold = F(name, 16, bold=True)
        self.f_tiny = F(name, 14)
        self.f_tbold = F(name, 14, bold=True)
        self.key_pressed = False
        self._reset_history()

    # ------------------------------------------------------------------
    # History: what changed each term (kept by the renderer, not the env)
    # ------------------------------------------------------------------
    def _reset_history(self):
        self.seen_term = None
        self.history = []        # one dict per finished term
        self.snap = None         # values at the current term
        self.prev = None         # values one term earlier
        self.dropout_term = {}

    def _snapshot(self, env):
        return {
            "children": [{k: c[k] for _, k, _, _ in CHILD_BARS} for c in env.children],
            "school": {k: getattr(env, k) for _, k, _ in SCHOOL_BARS},
            "on_track": self._on_track(env),
        }

    @staticmethod
    def _on_track(env):
        cfg = env.cfg
        return sum(c["enrolled"] and c["learning"] >= cfg["finish_learning"]
                   and c["confidence"] >= cfg["finish_confidence"] for c in env.children)

    def _track(self, env):
        from environment.custom_env import ACTION_NAMES
        if env.term == 0 or self.seen_term is None or env.term < self.seen_term:
            self._reset_history()
            self.seen_term = env.term
            self.snap = self._snapshot(env)
            return
        if env.term == self.seen_term:
            return
        self.prev, self.snap = self.snap, self._snapshot(env)
        self.seen_term = env.term
        # The timeline comes from the env's own log, so no term is missed even if
        # the window was not redrawn every term.
        self.history = [{
            "term": t["term"],
            "action": ACTION_NAMES[t["action"]],
            "reward": t["reward"],
            "failed": any("could not" in e.lower() for e in t["events"]),
            "bad_event": any(event_colour(e) == RED and "could not" not in e.lower() for e in t["events"]),
        } for t in getattr(env, "term_log", [])]
        self.dropout_term = {}
        for t in getattr(env, "term_log", []):
            for e in t["events"]:
                for i, c in enumerate(env.children):
                    if e == f"{c['name']} dropped out of school":
                        self.dropout_term.setdefault(i, t["term"])

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def text(self, s, font, colour, pos, anchor="topleft"):
        surf = font.render(str(s), True, colour)
        rect = surf.get_rect(**{anchor: pos})
        self.canvas.blit(surf, rect)
        return rect

    def fit(self, s, font, width):
        """Shorten text with '...' so it fits the width."""
        if font.size(s)[0] <= width:
            return s
        while s and font.size(s + "...")[0] > width:
            s = s[:-1]
        return s + "..."

    def panel(self, rect, fill=PANEL, border=LINE, radius=12, width=1):
        self.pg.draw.rect(self.canvas, fill, rect, border_radius=radius)
        self.pg.draw.rect(self.canvas, border, rect, width, border_radius=radius)

    def pill(self, s, font, fg, bg, pos, anchor="topleft", pad=(10, 3)):
        w, h = font.size(s)
        r = self.pg.Rect(0, 0, w + 2 * pad[0], h + 2 * pad[1])
        setattr(r, anchor, pos)
        self.pg.draw.rect(self.canvas, bg, r, border_radius=r.h // 2)
        self.text(s, font, fg, r.center, "center")
        return r

    def bar(self, x, y, w, h, value, colour, target=None):
        pg = self.pg
        pg.draw.rect(self.canvas, TRACK, (x, y, w, h), border_radius=h // 2)
        fill = max(0.0, min(1.0, value)) * w
        if fill > 1:
            pg.draw.rect(self.canvas, colour, (x, y, int(fill), h), border_radius=h // 2)
        if target is not None:
            tx = x + int(target * w)
            pg.draw.line(self.canvas, INK, (tx, y - 4), (tx, y + h + 3), 2)

    def delta(self, x, y, d, good_when_up=True):
        """Small arrow + number for this term's change. Returns the width used."""
        if abs(d) < 0.005:
            return self.text("=", self.f_tiny, MUTED, (x, y)).w
        up = d > 0
        colour = GREEN if up == good_when_up else RED
        cy = y + 9
        pts = [(x, cy + 4), (x + 10, cy + 4), (x + 5, cy - 5)] if up else \
              [(x, cy - 4), (x + 10, cy - 4), (x + 5, cy + 5)]
        self.pg.draw.polygon(self.canvas, colour, pts)
        r = self.text(f"{d:+.2f}", self.f_tbold, colour, (x + 13, y))
        return r.right - x

    def person(self, x, y, colour, scale=1.0, star=False, hat=False, faded=False):
        pg = self.pg
        skin = (225, 225, 225) if faded else (245, 205, 170)
        edge = GREY if faded else INK
        r = int(13 * scale)
        pg.draw.circle(self.canvas, skin, (x, y), r)
        pg.draw.circle(self.canvas, edge, (x, y), r, 2)
        body = pg.Rect(0, 0, int(34 * scale), int(40 * scale))
        body.midtop = (x, y + r - 2)
        pg.draw.rect(self.canvas, colour, body, border_radius=int(10 * scale))
        pg.draw.rect(self.canvas, edge, body, 2, border_radius=int(10 * scale))
        if hat:
            pg.draw.rect(self.canvas, INK, (x - r, y - r - 4, 2 * r, 6))
        if star:
            self.star(x + r + 8, y - r, 11, GOLD)

    def star(self, cx, cy, r, colour):
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        self.pg.draw.polygon(self.canvas, colour, pts)
        self.pg.draw.polygon(self.canvas, INK, pts, 1)

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------
    def draw_header(self, env):
        from environment.custom_env import MAX_TERMS
        self.text("School Director", self.f_title, INK, (M, 12))
        self.text("Keep six children in school, learning and confident for 6 years (18 terms)",
                  self.f_med, MUTED, (M + 2, 54))
        if env.term == 0:
            when = "Before term 1"
        else:
            when = f"Term {env.term} of {MAX_TERMS}   (Year {(env.term - 1) // 3 + 1}, term {(env.term - 1) % 3 + 1})"
        self.text(when, self.f_big, INK, (W - M, 14), "topright")
        who = getattr(env, "agent_label", None)
        sub = f"{env.scenario} world" + (f"   |   director: {who}" if who else "")
        self.text(sub, self.f_small, MUTED, (W - M, 52), "topright")

    def draw_decision(self, env):
        from environment.custom_env import ACTION_NAMES, MENTOR_0, SUPPORT_0
        rect = self.pg.Rect(M, 84, W - 2 * M, 62)
        self.panel(rect)
        if env.last_action is None:
            self.text("Start: the director has not made a decision yet", self.f_big, INK,
                      (M + 18, rect.centery), "midleft")
        else:
            a = env.last_action
            self.text(f"Term {env.term} decision:", self.f_med, MUTED, (M + 18, rect.centery), "midleft")
            r = self.text(ACTION_NAMES[a], self.f_big, INK, (M + 190, rect.centery), "midleft")
            x = r.right + 16
            if any("could not" in e.lower() for e in env.events):
                idx = a - SUPPORT_0 if a >= SUPPORT_0 else (a - MENTOR_0 if a >= MENTOR_0 else None)
                gone = idx is not None and not env.children[idx]["enrolled"]
                why = "child has already left school" if gone else "not enough money"
                self.pill(f"could not be done: {why}", self.f_sbold, (255, 255, 255), RED,
                          (x, rect.centery), "midleft")
            elif a >= SUPPORT_0:
                self.pill("helps one family", self.f_sbold, (255, 255, 255), BLUE, (x, rect.centery), "midleft")
            elif a >= MENTOR_0:
                self.pill("helps one child", self.f_sbold, (255, 255, 255), BLUE, (x, rect.centery), "midleft")
        rc = GREEN if env.last_reward >= 0 else RED
        tc = GREEN if env.total_reward >= 0 else RED
        r = self.text(f"{env.total_reward:+.2f}", self.f_big, tc, (W - M - 18, rect.centery), "midright")
        r = self.text("total", self.f_small, MUTED, (r.left - 8, rect.centery), "midright")
        r = self.text(f"{env.last_reward:+.2f}", self.f_big, rc, (r.left - 28, rect.centery), "midright")
        self.text("this term", self.f_small, MUTED, (r.left - 8, rect.centery), "midright")

    def draw_scene(self, env):
        from environment.custom_env import MAX_TEACHERS
        pg, s = self.pg, self.canvas
        scene = pg.Rect(M, 160, 520, 280)
        pg.draw.rect(s, SKY, scene, border_radius=12)
        pg.draw.rect(s, GRASS, (M, 350, 520, 90), border_bottom_left_radius=12, border_bottom_right_radius=12)
        pg.draw.rect(s, WALL, (110, 240, 340, 120))
        pg.draw.polygon(s, ROOF, [(90, 244), (280, 174), (470, 244)])
        pg.draw.rect(s, (110, 70, 50), (258, 300, 44, 60))
        for wx in (140, 200, 340, 400):
            pg.draw.rect(s, (230, 240, 250), (wx, 264, 34, 26))
            pg.draw.rect(s, INK, (wx, 264, 34, 26), 2)
        self.text("SCHOOL", self.f_bold, (255, 245, 230), (280, 218), "center")
        for i in range(MAX_TEACHERS):
            x = 70 + i * 82
            if i < env.teachers:
                self.person(x, 378, BLUE, 0.8, hat=True)
            else:
                pg.draw.circle(s, (190, 200, 185), (x, 390), 12, 2)
        self.text(f"Teachers: {env.teachers} of {MAX_TEACHERS}", self.f_bold, INK, (M + 16, 170))
        if env.tech > 0.25:
            pg.draw.rect(s, (60, 60, 70), (462, 176, 64, 42), border_radius=4)
            pg.draw.rect(s, (120, 220, 230), (467, 181, 54, 32))
            self.text("tech", self.f_tiny, MUTED, (494, 222), "midtop")

    def draw_school(self, env):
        px, w = 560, 500
        self.panel(self.pg.Rect(px, 160, w, 280))
        self.text("School", self.f_head, INK, (px + 16, 168))
        self.text("(change this term)", self.f_tiny, MUTED, (px + 100, 175))
        for i, (label, key, colour) in enumerate(SCHOOL_BARS):
            col, row = i % 2, i // 2
            x, y = px + 18 + col * 240, 210 + row * 76
            v = getattr(env, key)
            self.text(label, self.f_small, MUTED, (x, y))
            r = self.text(f"{v:.2f}", self.f_sbold, INK, (x + 140, y))
            if self.prev is not None:
                self.delta(r.right + 8, y, v - self.prev["school"][key])
            self.bar(x, y + 28, 210, 12, v, colour)

    def draw_term_report(self, env):
        px, w = 1076, W - M - 1076
        self.panel(self.pg.Rect(px, 160, w, 280))
        self.text("What happened this term", self.f_head, INK, (px + 16, 168))
        events = env.events if env.term > 0 else []
        y = 208
        if not events:
            self.text("No surprises" if env.term > 0 else "The 6 years are about to start",
                      self.f_med, MUTED, (px + 20, y))
        for e in events[:5]:
            colour = event_colour(e)
            self.pg.draw.circle(self.canvas, colour, (px + 26, y + 12), 6)
            self.text(self.fit(e, self.f_med, w - 60), self.f_med, INK, (px + 40, y))
            y += 30
        if len(events) > 5:
            self.text(f"+ {len(events) - 5} more", self.f_small, MUTED, (px + 40, y))
        now = self._on_track(env)
        y0 = 372
        self.pg.draw.line(self.canvas, LINE, (px + 16, y0 - 8), (px + w - 16, y0 - 8))
        label = "Finished strong" if env.end_reason else "On track to finish strong"
        val = env.finished_strong_count() if env.end_reason else now
        if self.prev is not None and not env.end_reason:
            txt = f"{label}: {self.prev['on_track']} -> {val} of 6"
        else:
            txt = f"{label}: {val} of 6"
        self.text(txt, self.f_bold, INK, (px + 20, y0))
        self.text(f"Dropouts {env.dropouts}     Teacher quits {env.quits}     "
                  f"Actions not possible {env.invalid_actions}", self.f_small, MUTED, (px + 20, y0 + 32))

    def draw_children(self, env):
        from environment.custom_env import MENTOR_0, SUPPORT_0
        cfg = env.cfg
        y0, hgt, gap = 452, 266, 12
        cw = (W - 2 * M - 5 * gap) / 6
        a = env.last_action
        ended = bool(env.end_reason)
        for i, c in enumerate(env.children):
            x = int(M + i * (cw + gap))
            w = int(cw)
            mentored = a is not None and MENTOR_0 <= a < SUPPORT_0 and a - MENTOR_0 == i
            supported = a is not None and a >= SUPPORT_0 and a - SUPPORT_0 == i
            helped = mentored or supported
            self.panel(self.pg.Rect(x, y0, w, hgt), border=BLUE if helped else LINE, width=3 if helped else 1)
            label, colour = child_status(c, cfg, ended)
            enrolled = c["enrolled"]
            self.person(x + 34, y0 + 30, colour, 0.95, star=ended and c["finished_strong"], faded=not enrolled)
            self.text(c["name"], self.f_head, INK if enrolled else MUTED, (x + 72, y0 + 12))
            self.pill(label, self.f_sbold, (255, 255, 255), colour, (x + 72, y0 + 48))
            if not enrolled:
                t = self.dropout_term.get(i)
                self.text(f"Left school{f' in term {t}' if t else ''}.", self.f_small, MUTED, (x + 14, y0 + 104))
                self.text("Cannot come back.", self.f_small, MUTED, (x + 14, y0 + 128))
                continue
            for j, (lab, key, col, up_good) in enumerate(CHILD_BARS):
                yy = y0 + 92 + j * 36
                v = c[key]
                target = {"learning": cfg["finish_learning"], "confidence": cfg["finish_confidence"]}.get(key)
                self.text(lab, self.f_small, MUTED, (x + 14, yy))
                r = self.text(fmt_value(v, target), self.f_sbold, INK, (x + 124, yy))
                if self.prev is not None:
                    self.delta(r.right + 6, yy, v - self.prev["children"][i][key], up_good)
                self.bar(x + 14, yy + 22, w - 28, 8, v, col, target)
            # Tags: who was helped and what hit this child this term.
            tags = []
            if mentored:
                tags.append(("Mentored this term", BLUE))
            if supported:
                tags.append(("Family supported", BLUE))
            for e in (env.events if env.term > 0 else []):
                if c["name"] in e:
                    if "Family shock" in e:
                        tags.append(("Family shock", RED))
                    elif "missed class" in e:
                        tags.append(("Missed class (work)", YELLOW))
            tx = x + 12
            for t, tc in tags:
                tw = self.f_tbold.size(t)[0] + 16
                if tx + tw > x + w - 8:  # never spill outside the card
                    break
                r = self.pill(t, self.f_tbold, (255, 255, 255), tc, (tx, y0 + hgt - 30), pad=(8, 2))
                tx = r.right + 6

    def draw_timeline(self, env):
        from environment.custom_env import MAX_TERMS
        top, ch, gap = 744, 92, 6
        cw = (W - 2 * M - (MAX_TERMS - 1) * gap) / MAX_TERMS
        for yr in range(6):
            x0 = M + yr * 3 * (cw + gap)
            x1 = x0 + 3 * cw + 2 * gap
            self.pg.draw.line(self.canvas, LINE, (x0, top - 6), (x1, top - 6), 2)
            self.text(f"Year {yr + 1}", self.f_tbold, MUTED, ((x0 + x1) / 2, top - 26), "midtop")
        by_term = {h["term"]: h for h in self.history}
        for t in range(1, MAX_TERMS + 1):
            x = int(M + (t - 1) * (cw + gap))
            rect = self.pg.Rect(x, top, int(cw), ch)
            h = by_term.get(t)
            if h is None:
                self.panel(rect, fill=(238, 234, 224), border=LINE, radius=8)
                self.text(str(t), self.f_tbold, (180, 176, 168), (rect.centerx, rect.y + 6), "midtop")
                continue
            if h["failed"]:
                fill = (246, 214, 208)
            elif h["reward"] >= 1.0:
                fill = (200, 232, 208)
            elif h["reward"] >= 0:
                fill = (224, 240, 226)
            else:
                fill = (246, 222, 216)
            current = t == env.term
            self.panel(rect, fill=fill, border=INK if current else LINE, radius=8, width=3 if current else 1)
            self.text(str(t), self.f_tbold, MUTED, (rect.centerx, rect.y + 5), "midtop")
            l1, l2 = short_action(h["action"])
            self.text(self.fit(l1, self.f_tbold, rect.w - 6), self.f_tbold, INK, (rect.centerx, rect.y + 24), "midtop")
            self.text(self.fit(l2, self.f_tiny, rect.w - 6), self.f_tiny, INK, (rect.centerx, rect.y + 42), "midtop")
            rc = GREEN if h["reward"] >= 0 else RED
            self.text(f"{h['reward']:+.2f}", self.f_tbold, rc, (rect.centerx, rect.y + 66), "midtop")
            if h["bad_event"]:
                self.pg.draw.circle(self.canvas, RED, (rect.right - 9, rect.y + 10), 4)
        ly = top + ch + 10
        x = M
        for lab, col in [("On track", GREEN), ("Getting there", YELLOW), ("At risk / needs help", RED),
                         ("Dropped out", GREY)]:
            self.pg.draw.circle(self.canvas, col, (x + 7, ly + 9), 7)
            x = self.text(lab, self.f_tiny, MUTED, (x + 20, ly)).right + 22
        x += 10
        self.pg.draw.line(self.canvas, INK, (x, ly + 1), (x, ly + 17), 2)
        x = self.text("= finish-strong target (learning 0.65, confidence 0.55)", self.f_tiny, MUTED,
                      (x + 8, ly)).right + 22
        self.pg.draw.circle(self.canvas, RED, (x + 4, ly + 9), 4)
        self.text("= bad luck that term (shock, cut, quit)", self.f_tiny, MUTED, (x + 14, ly))

    def draw_report_card(self, env):
        from environment.custom_env import ACTION_NAMES
        pg, cfg = self.pg, env.cfg
        shade = pg.Surface((W, H), pg.SRCALPHA)
        shade.fill((36, 40, 48, 70))
        self.canvas.blit(shade, (0, 0))
        rect = pg.Rect(M, 156, W - 2 * M, 562)
        self.panel(rect, fill=(255, 255, 255), border=INK, radius=16, width=2)
        x0 = rect.x + 28
        self.text("6-year report card", self.f_title, INK, (x0, rect.y + 16))
        finished = env.end_reason == "Term 18 reached"
        sub = "All 18 terms completed." if finished else f"The school closed after term {env.term}: {env.end_reason}."
        self.text(sub, self.f_med, MUTED if finished else RED, (x0 + 2, rect.y + 60))
        hint = getattr(env, "end_hint", "")
        if hint:
            self.text(hint, self.f_small, MUTED, (rect.right - 24, rect.y + 24), "topright")

        # Stat tiles
        tiles = [
            (f"{env.finished_strong_count()} / 6", "children finished strong", GREEN),
            (str(env.dropouts), "dropouts", RED if env.dropouts else INK),
            (str(env.quits), "teachers quit", RED if env.quits else INK),
            (f"{env.total_reward:+.2f}", "total reward", GREEN if env.total_reward >= 0 else RED),
        ]
        tw = (rect.w - 56 - 3 * 16) / 4
        for k, (big, lab, col) in enumerate(tiles):
            r = pg.Rect(int(x0 + k * (tw + 16)), rect.y + 96, int(tw), 78)
            self.panel(r, fill=(248, 245, 238), border=LINE, radius=12)
            self.text(big, self.f_huge, col, (r.x + 18, r.y + 8))
            self.text(lab, self.f_small, MUTED, (r.x + 20, r.y + 54))

        # Children table
        ty = rect.y + 196
        cols = [("Child", x0), ("Learning", x0 + 200), ("Attendance", x0 + 330), ("Confidence", x0 + 470),
                ("Result", x0 + 610)]
        for lab, cx in cols:
            self.text(lab, self.f_sbold, MUTED, (cx, ty))
        pg.draw.line(self.canvas, LINE, (x0, ty + 26), (x0 + 860, ty + 26))
        for i, c in enumerate(env.children):
            y = ty + 38 + i * 46
            label, colour = child_status(c, cfg, True)
            pg.draw.circle(self.canvas, colour, (x0 + 8, y + 13), 8)
            self.text(c["name"], self.f_bold, INK, (x0 + 26, y))
            if c["finished_strong"]:
                self.star(x0 + 40 + self.f_bold.size(c["name"])[0], y + 12, 10, GOLD)
            learn_ok = c["learning"] >= cfg["finish_learning"]
            conf_ok = c["confidence"] >= cfg["finish_confidence"]
            self.text(fmt_value(c["learning"], cfg["finish_learning"]), self.f_bold,
                      GREEN if learn_ok else RED, (cols[1][1], y))
            self.text(f"{c['attendance']:.2f}", self.f_bold, INK, (cols[2][1], y))
            self.text(fmt_value(c["confidence"], cfg["finish_confidence"]), self.f_bold,
                      GREEN if conf_ok else RED, (cols[3][1], y))
            if not c["enrolled"]:
                t = self.dropout_term.get(i)
                why = f"Dropped out{f' (term {t})' if t else ''}"
            elif c["finished_strong"]:
                why = "Finished strong"
            elif not finished:
                why = "School closed early"
            elif not learn_ok and not conf_ok:
                why = "Learning and confidence below target"
            elif not learn_ok:
                why = "Learning below 0.65"
            else:
                why = "Confidence below 0.55"
            self.text(why, self.f_bold, colour, (cols[4][1], y))

        # How the terms were spent
        sx = x0 + 900
        self.text(f"How the director spent {env.term} terms", self.f_head, INK, (sx, ty - 4))
        counts = sorted([(n, ACTION_NAMES[i]) for i, n in enumerate(env.action_counts) if n > 0], reverse=True)
        top_n = counts[:8]
        mx = max((n for n, _ in top_n), default=1)
        for k, (n, name) in enumerate(top_n):
            y = ty + 38 + k * 34
            self.text(self.fit(name, self.f_small, 190), self.f_small, INK, (sx, y))
            self.bar(sx + 200, y + 5, 300 * n / mx, 12, 1.0, BLUE)
            self.text(f"x{n}", self.f_sbold, INK, (sx + 200 + 300 * n / mx + 8, y))
        if len(counts) > 8:
            self.text(f"+ {len(counts) - 8} other actions", self.f_tiny, MUTED,
                      (sx, ty + 38 + 8 * 34))
        self.text(f"Actions that could not be done: {env.invalid_actions}", self.f_small,
                  RED if env.invalid_actions else MUTED, (sx, rect.bottom - 36))

    # ------------------------------------------------------------------
    def draw(self, env):
        pg = self.pg
        if self.mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                    pg.quit()
                    raise SystemExit
                if event.type == pg.KEYDOWN:
                    self.key_pressed = True
        self._track(env)
        self.canvas.fill(BG)
        self.draw_header(env)
        self.draw_decision(env)
        self.draw_scene(env)
        self.draw_school(env)
        self.draw_term_report(env)
        self.draw_children(env)
        self.draw_timeline(env)
        if env.end_reason:
            self.draw_report_card(env)

        if self.mode == "human":
            win = pg.display.get_surface()
            if win.get_size() == (W, H):
                win.blit(self.canvas, (0, 0))
            else:
                win.blit(pg.transform.smoothscale(self.canvas, win.get_size()), (0, 0))
            pg.display.flip()
            self.clock.tick(30)
            return None
        return np.transpose(pg.surfarray.array3d(self.canvas), (1, 0, 2)).copy()

    def close(self):
        if self.mode == "human":
            self.pg.display.quit()
        self.pg.quit()


# ---------------------------------------------------------------------------
# 3D replay (three.js / WebGL)
# ---------------------------------------------------------------------------

def export_3d_replay(record: dict, out_path: str, open_browser: bool = True) -> str:
    """Write a self-contained three.js (WebGL) replay of one episode.

    `record` is the episode log written by main.py: per-term state (from
    SchoolEnv.get_state()), action, reward and events. The page reads only that
    JSON - the same serialization path a web or mobile front-end would use.
    """
    import json
    import os
    import webbrowser

    template = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer3d.html")
    with open(template, encoding="utf-8") as f:
        html = f.read()
    data = json.dumps(record).replace("</", "<\\/")
    html = html.replace("/*EPISODE_JSON*/null", data)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    if open_browser:
        webbrowser.open("file://" + os.path.abspath(out_path))
    return out_path
