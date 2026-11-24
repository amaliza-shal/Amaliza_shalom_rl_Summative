"""Pygame view of the School Director environment.

Shows the school building with one icon per teacher, the six children
coloured by how they are doing, school bars, the term, the chosen action and
this term's events. Dropouts walk off the screen.
"""

from __future__ import annotations

import math

import numpy as np

W, H = 1180, 720

# Colours
BG = (245, 241, 232)
INK = (40, 44, 52)
MUTED = (120, 124, 132)
PANEL = (255, 252, 246)
LINE = (220, 214, 200)
GREEN = (46, 160, 90)
YELLOW = (226, 172, 40)
RED = (210, 70, 60)
BLUE = (60, 120, 200)
WALL = (196, 120, 84)
ROOF = (150, 70, 60)
GRASS = (170, 205, 140)
SKY = (205, 228, 245)

BAR_COLOURS = {
    "Budget": (70, 150, 110),
    "Gov funding": (60, 120, 200),
    "Community trust": (170, 110, 190),
    "Tech level": (40, 160, 170),
    "Teacher quality": (220, 140, 50),
    "Teacher morale": (200, 90, 110),
}


def status_colour(c) -> tuple:
    if not c["enrolled"]:
        return (150, 150, 150)
    score = (c["learning"] + c["attendance"] + c["confidence"]) / 3
    if score >= 0.6:
        return GREEN
    if score >= 0.45:
        return YELLOW
    return RED


class SchoolRenderer:
    def __init__(self, render_mode: str = "human"):
        import pygame

        self.pg = pygame
        self.mode = render_mode
        pygame.init()
        pygame.font.init()
        if render_mode == "human":
            self.screen = pygame.display.set_mode((W, H))
            pygame.display.set_caption("School Director - Mission RL")
        else:
            self.screen = pygame.Surface((W, H))
        self.clock = pygame.time.Clock()
        name = "segoeui,arial,helvetica"
        self.f_title = pygame.font.SysFont(name, 30, bold=True)
        self.f_big = pygame.font.SysFont(name, 22, bold=True)
        self.f_med = pygame.font.SysFont(name, 17)
        self.f_bold = pygame.font.SysFont(name, 17, bold=True)
        self.f_small = pygame.font.SysFont(name, 14)
        self.walk_offset = {}  # child index -> pixels walked off screen
        self.frame = 0

    # ------------------------------------------------------------------
    def text(self, s, font, colour, pos, anchor="topleft"):
        surf = font.render(str(s), True, colour)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def bar(self, x, y, w, h, value, colour, label=None, show_value=True):
        pg = self.pg
        pg.draw.rect(self.screen, (232, 228, 218), (x, y, w, h), border_radius=h // 2)
        fill = max(0, min(1, value)) * w
        if fill > 1:
            pg.draw.rect(self.screen, colour, (x, y, int(fill), h), border_radius=h // 2)
        if label:
            self.text(label, self.f_small, MUTED, (x, y - 17))
        if show_value:
            self.text(f"{value:.2f}", self.f_small, INK, (x + w + 6, y - 2))

    def person(self, x, y, colour, scale=1.0, star=False, hat=False):
        pg = self.pg
        r = int(13 * scale)
        pg.draw.circle(self.screen, (245, 205, 170), (x, y), r)
        pg.draw.circle(self.screen, INK, (x, y), r, 2)
        body = pg.Rect(0, 0, int(34 * scale), int(40 * scale))
        body.midtop = (x, y + r - 2)
        pg.draw.rect(self.screen, colour, body, border_radius=int(10 * scale))
        pg.draw.rect(self.screen, INK, body, 2, border_radius=int(10 * scale))
        if hat:  # teacher's cap
            pg.draw.rect(self.screen, INK, (x - r, y - r - 4, 2 * r, 6))
        if star:
            self.star(x + r + 10, y - r, 10, (250, 200, 40))

    def star(self, cx, cy, r, colour):
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        self.pg.draw.polygon(self.screen, colour, pts)
        self.pg.draw.polygon(self.screen, INK, pts, 1)

    # ------------------------------------------------------------------
    def draw(self, env):
        from environment.custom_env import ACTION_NAMES, MAX_TEACHERS, MAX_TERMS

        pg = self.pg
        if self.mode == "human":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    raise SystemExit
        self.frame += 1
        s = self.screen
        s.fill(BG)

        # Header
        self.text("School Director", self.f_title, INK, (24, 14))
        self.text("Keep six children in school and learning for 6 years",
                  self.f_med, MUTED, (26, 52))
        year = min(6, max(env.term - 1, 0) // 3 + 1)
        self.text(f"Term {env.term} / {MAX_TERMS}   (Year {year})", self.f_big, INK, (W - 24, 18), "topright")
        pg.draw.rect(s, (232, 228, 218), (W - 324, 54, 300, 10), border_radius=5)
        pg.draw.rect(s, BLUE, (W - 324, 54, int(300 * env.term / MAX_TERMS), 10), border_radius=5)

        # Action banner
        act = "Start of the 6 years" if env.last_action is None else ACTION_NAMES[env.last_action]
        pg.draw.rect(s, PANEL, (24, 82, W - 48, 46), border_radius=10)
        pg.draw.rect(s, LINE, (24, 82, W - 48, 46), 1, border_radius=10)
        self.text("Director's choice:", self.f_med, MUTED, (40, 94))
        self.text(act, self.f_big, INK, (190, 91))
        rc = GREEN if env.last_reward >= 0 else RED
        self.text(f"term reward {env.last_reward:+.2f}    total {env.total_reward:+.2f}",
                  self.f_bold, rc, (W - 40, 95), "topright")

        # School scene
        scene = pg.Rect(24, 142, 520, 250)
        pg.draw.rect(s, SKY, scene, border_radius=12)
        pg.draw.rect(s, GRASS, (24, 322, 520, 70), border_bottom_left_radius=12,
                     border_bottom_right_radius=12)
        pg.draw.rect(s, WALL, (110, 222, 340, 120))
        pg.draw.polygon(s, ROOF, [(90, 226), (280, 156), (470, 226)])
        pg.draw.rect(s, (110, 70, 50), (258, 282, 44, 60))
        for wx in (140, 200, 340, 400):
            pg.draw.rect(s, (230, 240, 250), (wx, 246, 34, 26))
            pg.draw.rect(s, INK, (wx, 246, 34, 26), 2)
        self.text("SCHOOL", self.f_bold, (255, 245, 230), (280, 200), "center")
        # One icon per teacher, standing in front of the school.
        for i in range(MAX_TEACHERS):
            x = 70 + i * 82
            if i < env.teachers:
                self.person(x, 340, BLUE, 0.8, hat=True)
            else:
                pg.draw.circle(s, (190, 200, 185), (x, 352), 12, 2)
        self.text(f"Teachers: {env.teachers}", self.f_bold, INK, (40, 150))
        if env.tech > 0.25:
            pg.draw.rect(s, (60, 60, 70), (460, 160, 64, 42), border_radius=4)
            pg.draw.rect(s, (120, 220, 230), (465, 165, 54, 32))
            self.text("tech", self.f_small, MUTED, (492, 206), "midtop")

        # School bars
        px = 568
        pg.draw.rect(s, PANEL, (px, 142, W - px - 24, 250), border_radius=12)
        pg.draw.rect(s, LINE, (px, 142, W - px - 24, 250), 1, border_radius=12)
        self.text("School", self.f_big, INK, (px + 16, 152))
        bars = [
            ("Budget", env.budget), ("Gov funding", env.gov_funding),
            ("Community trust", env.trust), ("Tech level", env.tech),
            ("Teacher quality", env.quality), ("Teacher morale", env.morale),
        ]
        for i, (label, v) in enumerate(bars):
            col, row = i % 2, i // 2
            self.bar(px + 18 + col * 292, 212 + row * 60, 220, 12, v, BAR_COLOURS[label], label)

        # Children
        cy = 410
        pg.draw.rect(s, PANEL, (24, cy, W - 48, 220), border_radius=12)
        pg.draw.rect(s, LINE, (24, cy, W - 48, 220), 1, border_radius=12)
        slot = (W - 48) / 6
        for i, c in enumerate(env.children):
            x0 = int(24 + i * slot)
            cx = x0 + int(slot / 2)
            if not c["enrolled"]:
                off = self.walk_offset.get(i, 0) + 40
                self.walk_offset[i] = off
                if off < 260:
                    self.person(cx + off, cy + 48, (150, 150, 150), 0.9)
                self.text(c["name"], self.f_bold, MUTED, (cx, cy + 118), "midtop")
                self.text("dropped out", self.f_small, RED, (cx, cy + 140), "midtop")
                continue
            self.person(cx, cy + 48, status_colour(c), 0.9, star=c["finished_strong"])
            self.text(c["name"], self.f_bold, INK, (cx, cy + 102), "midtop")
            bw = int(slot) - 70
            for j, (lab, key, colr) in enumerate([
                ("learn", "learning", BLUE), ("attend", "attendance", GREEN),
                ("confid", "confidence", (170, 110, 190)), ("home", "home_pressure", RED),
            ]):
                yy = cy + 132 + j * 20
                self.text(lab, self.f_small, MUTED, (x0 + 10, yy - 4))
                self.bar(x0 + 58, yy, bw - 20, 8, c[key], colr, show_value=False)

        # Events and legend
        ey = 642
        on_track = sum(
            c["enrolled"] and c["learning"] >= env.cfg["finish_learning"]
            and c["confidence"] >= env.cfg["finish_confidence"] for c in env.children
        )
        self.text("This term:", self.f_bold, INK, (26, ey))
        events = env.events[-3:] if env.events else ["No surprises"]
        self.text("   |   ".join(events), self.f_med, MUTED, (120, ey))
        self.text(
            f"Dropouts {env.dropouts}    Teacher quits {env.quits}    "
            + (f"Finished strong {env.finished_strong_count()} / 6" if env.end_reason
               else f"On track to finish strong {on_track} / 6"),
            self.f_med, INK, (26, ey + 30),
        )
        lx = W - 470
        for k, (lab, col) in enumerate([("doing well", GREEN), ("at risk", YELLOW), ("struggling", RED)]):
            pg.draw.circle(s, col, (lx + k * 150, ey + 40), 8)
            self.text(lab, self.f_small, MUTED, (lx + 14 + k * 150, ey + 32))
        if env.end_reason:
            msg = pg.Rect(0, 0, 640, 70)
            msg.center = (W // 2, 270)
            pg.draw.rect(s, (255, 255, 255), msg, border_radius=14)
            pg.draw.rect(s, INK, msg, 2, border_radius=14)
            self.text(env.end_reason, self.f_big, INK, (msg.centerx, msg.y + 10), "midtop")
            self.text(f"{env.finished_strong_count()} of 6 children finished strong   "
                      f"total reward {env.total_reward:+.2f}", self.f_med, MUTED,
                      (msg.centerx, msg.y + 40), "midtop")

        if self.mode == "human":
            pg.display.flip()
            self.clock.tick(30)
            return None
        return np.transpose(pg.surfarray.array3d(s), (1, 0, 2)).copy()

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
    data = json.dumps(record).replace("</", "<\/")
    html = html.replace("/*EPISODE_JSON*/null", data)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    if open_browser:
        webbrowser.open("file://" + os.path.abspath(out_path))
    return out_path
