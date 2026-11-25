"""School Director environment.

The agent is a school director in an underserved community. It looks after
six children for 18 school terms (6 years) and makes one choice per term.

Everything sits on a 0 to 1 scale and comes from simple update rules. This is
a simulation of the trade-offs in the mission, not a forecast for a real school.
"""

from __future__ import annotations

import copy
import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Names and actions
# ---------------------------------------------------------------------------

CHILD_NAMES = ["Shalom", "Jolly", "Larrissa", "Channelle", "Delucie", "Keilla"]
N_CHILDREN = len(CHILD_NAMES)
MAX_TERMS = 18
MAX_TEACHERS = 6

SCHOOL_ACTIONS = [
    "Hire a teacher",
    "Train teachers",
    "Buy technology",
    "Reach remote areas",
    "Engage the community",
    "Lobby the government",
    "Save money",
]
ACTION_NAMES = (
    SCHOOL_ACTIONS
    + [f"Mentor {n}" for n in CHILD_NAMES]
    + [f"Support {n}'s family" for n in CHILD_NAMES]
)
N_ACTIONS = len(ACTION_NAMES)  # 19

# Action ids, so the code reads like the plan.
HIRE, TRAIN, TECH, REMOTE, ENGAGE, LOBBY, SAVE = range(7)
MENTOR_0 = 7
SUPPORT_0 = 13

# Observation layout: 6 children x [learning, attendance, confidence,
# home_pressure], then 8 school numbers.
CHILD_FEATURES = ["learning", "attendance", "confidence", "home_pressure"]
SCHOOL_FEATURES = [
    "budget",
    "teachers",
    "teacher_quality",
    "teacher_morale",
    "community_trust",
    "tech_level",
    "gov_funding",
    "term_progress",
]
OBS_NAMES = [f"{n}.{f}" for n in CHILD_NAMES for f in CHILD_FEATURES] + SCHOOL_FEATURES
OBS_SIZE = len(OBS_NAMES)  # 32

# ---------------------------------------------------------------------------
# Config: every weight and rule lives here, in one place.
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # Reward weights. LOCKED on 2026-11-23 after the baseline tests (plan weights
    # unchanged; invalid_action added). Do not change during experiments.
    "reward": {
        "learning_gain": 10.0,
        "confidence_gain": 5.0,
        "attendance_level": 0.2,
        "trust_gain": 2.0,
        "dropout": -5.0,
        "teacher_quit": -2.0,
        "fairness_gap": -1.0,
        "fairness_threshold": 0.4,
        "collapse": -10.0,
        "invalid_action": -1.0,
        "final_bonus": 20.0,
    },
    # What "finishing strong" means at term 18.
    "finish_learning": 0.65,
    "finish_confidence": 0.55,
    # Start state.
    "start_budget": 0.40,
    "start_teachers": 2,
    "start_quality": 0.40,
    "start_morale": 0.60,
    "start_trust": 0.30,
    "start_tech": 0.10,
    "start_gov_funding": 0.30,
    "start_noise": 0.05,
    # Action costs (share of the budget cap).
    "cost": {
        HIRE: 0.20, TRAIN: 0.10, TECH: 0.20, REMOTE: 0.10,
        ENGAGE: 0.03, LOBBY: 0.03, SAVE: 0.0, "mentor": 0.04, "support": 0.08,
    },
    # Money.
    "salary_per_teacher": 0.04,
    "income_scale": 0.30,
    # Random events.
    "p_funding_cut": 0.10,
    "funding_cut_factor": 0.70,
    "p_quit_low_morale": 0.30,
    "p_quit_normal": 0.02,
    "low_morale": 0.30,
    "p_family_shock": 0.05,
    "family_shock_poverty": 0.33,
    # Extra poverty added to every child at the start (used by the hard world).
    "extra_poverty": 0.0,
}


def make_config(scenario: str = "standard") -> dict:
    """Return a config. 'hard' is the harder world used for Figure 5."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if scenario == "hard":
        cfg["p_funding_cut"] = 0.25
        cfg["extra_poverty"] = 0.15
        cfg["start_teachers"] = 1
    elif scenario != "standard":
        raise ValueError(f"Unknown scenario: {scenario}")
    return cfg


# Each child's story as numbers. poverty and distance are hidden; the agent
# sees home_pressure = 0.6 * poverty + 0.4 * distance.
CHILD_PROFILES = [
    # Shalom: walks two hours each way. Sharp, but tired and late.
    dict(learning=0.52, attendance=0.50, confidence=0.45, poverty=0.35, distance=0.90,
         tech_affinity=1.0, work_spike=0.0, neglect_sensitive=False),
    # Jolly: works after school. Misses class when work picks up.
    dict(learning=0.48, attendance=0.60, confidence=0.45, poverty=0.75, distance=0.30,
         tech_affinity=1.0, work_spike=0.25, neglect_sensitive=False),
    # Larrissa: quiet, rarely speaks, needs a mentor.
    dict(learning=0.47, attendance=0.65, confidence=0.20, poverty=0.40, distance=0.30,
         tech_affinity=1.0, work_spike=0.0, neglect_sensitive=False),
    # Channelle: bright, bored without technology.
    dict(learning=0.62, attendance=0.80, confidence=0.55, poverty=0.15, distance=0.20,
         tech_affinity=2.0, work_spike=0.0, neglect_sensitive=False),
    # Delucie: lives with a grandparent, very little money.
    dict(learning=0.42, attendance=0.55, confidence=0.45, poverty=0.90, distance=0.35,
         tech_affinity=1.0, work_spike=0.0, neglect_sensitive=False),
    # Keilla: steady, easy to overlook, slips if ignored.
    dict(learning=0.50, attendance=0.85, confidence=0.50, poverty=0.30, distance=0.30,
         tech_affinity=1.0, work_spike=0.0, neglect_sensitive=True),
]


def _clip(x):
    return float(np.clip(x, 0.0, 1.0))


class SchoolEnv(gym.Env):
    """A school director keeps six children in school and learning for 18 terms."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode: str | None = None, scenario: str = "standard",
                 config: dict | None = None):
        super().__init__()
        self.scenario = scenario
        self.cfg = config if config is not None else make_config(scenario)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._renderer = None
        self.last_action = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.events: list[str] = []

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg, rng = self.cfg, self.np_random
        noise = cfg["start_noise"]

        def jitter(v):
            return _clip(v + rng.uniform(-noise, noise))

        self.children = []
        for name, p in zip(CHILD_NAMES, CHILD_PROFILES):
            self.children.append(dict(
                name=name,
                learning=jitter(p["learning"]),
                attendance=jitter(p["attendance"]),
                confidence=jitter(p["confidence"]),
                poverty=jitter(p["poverty"] + cfg["extra_poverty"]),
                distance=jitter(p["distance"]),
                tech_affinity=p["tech_affinity"],
                work_spike=p["work_spike"],
                neglect_sensitive=p["neglect_sensitive"],
                mentor_boost=0.0,   # fades by 40% each term
                relief=0.0,         # family support that fades if not repeated
                neglect=0,          # terms since this child got direct help
                low_terms=0,        # terms in a row at risk of dropping out
                enrolled=True,
                finished_strong=False,
            ))
            self.children[-1]["home_pressure"] = self._home_pressure(self.children[-1])

        self.budget = jitter(cfg["start_budget"])
        self.teachers = int(cfg["start_teachers"])
        self.quality = jitter(cfg["start_quality"])
        self.morale = jitter(cfg["start_morale"])
        self.trust = jitter(cfg["start_trust"])
        self.tech = jitter(cfg["start_tech"])
        self.gov_funding = jitter(cfg["start_gov_funding"])
        self.term = 0
        self.pending: dict[int, list[tuple]] = {}  # delayed effects by term
        self.prev_avg_learning = self._avg("learning")
        self.dropouts = 0
        self.quits = 0
        self.invalid_actions = 0
        self.action_counts = [0] * N_ACTIONS
        self.last_action = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.events = []
        self.end_reason = None
        self.term_log = []  # one entry per term, for the display and replays (not seen by the agent)

        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action), f"Bad action {action}"
        cfg, w = self.cfg, self.cfg["reward"]
        self.events = []
        self.last_action = action
        self.action_counts[action] += 1

        before_learning = self._avg("learning")
        before_conf = self._avg("confidence")
        before_trust = self.trust

        reward = 0.0
        # 1. The director's choice for this term.
        if not self._apply_action(action):
            reward += w["invalid_action"]
            self.invalid_actions += 1
            self.events.append("Action could not be done (no money or no effect)")

        self.term += 1
        # 2. Delayed effects from earlier terms land now.
        self._apply_pending()
        # 3. Random events: plan B moments.
        quits = self._random_events()
        # 4. The school and the children change.
        self._update_school(action)
        new_dropouts = self._update_children(action)
        # 5. Pay the teachers.
        collapse = self._pay_salaries()

        # Reward for this term.
        reward += w["learning_gain"] * (self._avg("learning") - before_learning)
        reward += w["confidence_gain"] * (self._avg("confidence") - before_conf)
        reward += w["attendance_level"] * self._avg("attendance")
        reward += w["trust_gain"] * (self.trust - before_trust)
        reward += w["dropout"] * new_dropouts
        reward += w["teacher_quit"] * quits
        enrolled_learning = [c["learning"] for c in self.children if c["enrolled"]]
        if enrolled_learning and max(enrolled_learning) - min(enrolled_learning) > w["fairness_threshold"]:
            reward += w["fairness_gap"]

        terminated = False
        if collapse:
            reward += w["collapse"]
            terminated, self.end_reason = True, "Budget collapse: teachers could not be paid"
        elif self.teachers == 0:
            terminated, self.end_reason = True, "All teachers quit"
        elif self.dropouts >= 4:
            terminated, self.end_reason = True, "4 or more children dropped out"
        elif self.term >= MAX_TERMS:
            terminated, self.end_reason = True, "Term 18 reached"
            for c in self.children:
                c["finished_strong"] = bool(
                    c["enrolled"]
                    and c["learning"] >= cfg["finish_learning"]
                    and c["confidence"] >= cfg["finish_confidence"]
                )
            reward += w["final_bonus"] * self.finished_strong_count() / N_CHILDREN

        self.last_reward = float(reward)
        self.total_reward += float(reward)
        self.term_log.append({"term": self.term, "action": action, "reward": float(reward),
                              "events": list(self.events)})
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, False, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from environment.rendering import SchoolRenderer
            self._renderer = SchoolRenderer(self.render_mode, getattr(self, "window_scale", None))
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _cost(self, action):
        cost = self.cfg["cost"]
        if action >= SUPPORT_0:
            return cost["support"]
        if action >= MENTOR_0:
            return cost["mentor"]
        return cost[action]

    def _schedule(self, delay, effect):
        self.pending.setdefault(self.term + delay, []).append(effect)

    def _apply_action(self, a) -> bool:
        """Do the action. Return False if it could not be done."""
        cost = self._cost(a)
        if cost > self.budget + 1e-9:
            return False  # cannot pay: nothing happens

        if a == HIRE:
            if self.teachers >= MAX_TEACHERS:
                return False
            # A new teacher starts with low quality, so the average drops.
            self.quality = (self.quality * self.teachers + 0.30) / (self.teachers + 1)
            self.teachers += 1
        elif a == TRAIN:
            # Pays off in 2 terms, with diminishing returns as quality rises.
            self._schedule(2, ("quality", 0.15 * (1.0 - self.quality)))
            self._schedule(1, ("morale", 0.08))
        elif a == TECH:
            self._schedule(1, ("tech", 0.15))
            self._schedule(2, ("tech", 0.10))
        elif a == REMOTE:
            for d in (1, 2, 3):                    # spreads over 3 terms
                self._schedule(d, ("distance", 0.05))
        elif a == ENGAGE:
            for d in (1, 2, 3):                    # trust builds slowly
                self._schedule(d, ("trust", 0.025))
        elif a == LOBBY:
            # Works better when results are improving.
            improving = self._avg("learning") > self.prev_avg_learning
            p = 0.55 + (0.25 if improving else 0.0) + 0.2 * (self.trust - 0.3)
            if self.np_random.random() < p:
                self._schedule(2, ("gov", 0.08))
        elif a == SAVE:
            self.morale = _clip(self.morale - 0.01)
            self.trust = _clip(self.trust - 0.005)
        elif a >= SUPPORT_0:
            c = self.children[a - SUPPORT_0]
            if not c["enrolled"]:
                return False
            c["poverty"] = _clip(c["poverty"] - 0.06)  # part of the help lasts
            c["relief"] = 1.0                          # the rest fades
            c["neglect"] = 0
        elif a >= MENTOR_0:
            c = self.children[a - MENTOR_0]
            if not c["enrolled"]:
                return False
            c["mentor_boost"] = min(1.0, c["mentor_boost"] + 1.0)
            c["neglect"] = 0

        self.budget = _clip(self.budget - cost)
        return True

    def _apply_pending(self):
        for kind, value in self.pending.pop(self.term, []):
            if kind == "quality":
                self.quality = _clip(self.quality + value)
            elif kind == "morale":
                self.morale = _clip(self.morale + value)
            elif kind == "tech":
                self.tech = _clip(self.tech + value)
            elif kind == "distance":
                for c in self.children:
                    c["distance"] = _clip(c["distance"] - value)
            elif kind == "trust":
                self.trust = _clip(self.trust + value)
            elif kind == "gov":
                self.gov_funding = _clip(self.gov_funding + value)
                self.events.append("Lobbying paid off: government funding up")

    # ------------------------------------------------------------------
    # Random events
    # ------------------------------------------------------------------

    def _random_events(self) -> int:
        cfg, rng = self.cfg, self.np_random
        if rng.random() < cfg["p_funding_cut"]:
            self.gov_funding *= cfg["funding_cut_factor"]
            self.events.append("Funding cut: government funding down 30%")

        quits = 0
        p_quit = cfg["p_quit_low_morale"] if self.morale < cfg["low_morale"] else cfg["p_quit_normal"]
        if self.teachers > 0 and rng.random() < p_quit:
            self.teachers -= 1
            quits = 1
            self.quits += 1
            self.trust = _clip(self.trust - 0.05)
            self.events.append("A teacher left the school")

        for c in self.children:
            if c["enrolled"] and rng.random() < cfg["p_family_shock"]:
                c["poverty"] = _clip(c["poverty"] + cfg["family_shock_poverty"])
                self.events.append(f"Family shock at {c['name']}'s home")
        return quits

    # ------------------------------------------------------------------
    # School and children updates
    # ------------------------------------------------------------------

    def _update_school(self, action):
        # Teachers lose a little skill without training.
        self.quality = _clip(self.quality - 0.005)
        # Morale: trust and a healthy budget help; a heavy class load hurts.
        overload = max(0, 4 - self.teachers) / 2
        money = 0.01 if self.budget > 0.20 else -0.02
        self.morale = _clip(self.morale + 0.03 * (self.trust - 0.30) + money - 0.015 * overload)
        # Trust drifts down unless the school engages.
        self.trust = _clip(self.trust - 0.005)
        # Technology wears down.
        self.tech = _clip(self.tech - 0.02)
        # Government funding follows results: it moves 20% toward a target.
        share_enrolled = sum(c["enrolled"] for c in self.children) / N_CHILDREN
        target = -0.10 + 0.45 * self._avg("learning") + 0.15 * share_enrolled + 0.2 * self.trust
        self.gov_funding = _clip(self.gov_funding + 0.2 * (target - self.gov_funding))

    def _home_pressure(self, c):
        return _clip(0.6 * c["poverty"] + 0.4 * c["distance"] - 0.2 * c["relief"])

    def _update_children(self, action) -> int:
        cfg = self.cfg
        teacher_factor = float(np.clip(self.teachers / 4, 0.4, 1.0))
        new_dropouts = 0
        for i, c in enumerate(self.children):
            if not c["enrolled"]:
                continue
            if not (action == MENTOR_0 + i or action == SUPPORT_0 + i):
                c["neglect"] += 1
            c["home_pressure"] = self._home_pressure(c)

            # Attendance moves halfway toward a target each term.
            target = (1.02 - 0.85 * c["home_pressure"] + 0.3 * (self.trust - 0.3)
                      + 0.12 * c["mentor_boost"] + 0.2 * (c["confidence"] - 0.4))
            if c["work_spike"] and c["relief"] < 0.3 and self.np_random.random() < c["work_spike"]:
                target -= 0.2
                self.events.append(f"{c['name']} missed class for work")
            c["attendance"] = _clip(c["attendance"] + 0.5 * (target - c["attendance"]))

            # Learning grows with attendance, teacher quality, class size and tech.
            tech_boost = 1 + 0.5 * self.tech * c["tech_affinity"]
            boredom = 0.75 if (c["tech_affinity"] > 1 and self.tech < 0.3) else 1.0
            gain = (0.07 * c["attendance"] * (0.4 + 0.6 * self.quality)
                    * teacher_factor * tech_boost * boredom)
            if c["attendance"] < 0.4:
                gain -= 0.01  # falls behind
            c["learning"] = _clip(c["learning"] + gain)

            # Confidence: mentoring and progress lift it; falling behind lowers it.
            dconf = (0.07 * c["mentor_boost"] + 0.8 * max(gain, 0)
                     + 0.03 * (self.quality - 0.45) - 0.008)
            if c["attendance"] < 0.4:
                dconf -= 0.03
            if c["neglect_sensitive"] and c["neglect"] >= 4:
                dconf -= 0.015
            c["confidence"] = _clip(c["confidence"] + dconf)

            # Help fades over time.
            c["mentor_boost"] *= 0.6
            c["relief"] *= 0.5

            # Dropout: attendance and confidence both below 0.25 for 2 terms.
            if c["attendance"] < 0.25 and c["confidence"] < 0.25:
                c["low_terms"] += 1
            else:
                c["low_terms"] = 0
            if c["low_terms"] >= 2:
                c["enrolled"] = False
                c["attendance"] = 0.0
                new_dropouts += 1
                self.dropouts += 1
                self.trust = _clip(self.trust - 0.05)
                self.events.append(f"{c['name']} dropped out of school")
        self.prev_avg_learning = self._avg("learning")
        return new_dropouts

    def _pay_salaries(self) -> bool:
        """Add income, pay teachers. Return True if the school collapses."""
        income = self.cfg["income_scale"] * self.gov_funding
        salary = self.cfg["salary_per_teacher"] * self.teachers
        self.budget = self.budget + income - salary
        if self.budget < 0:
            self.budget = 0.0
            return True
        self.budget = _clip(self.budget)
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _avg(self, key):
        # Dropped-out children count with their last learning and zero
        # attendance/confidence, so losing a child always hurts the average.
        vals = [c[key] if c["enrolled"] or key == "learning" else 0.0 for c in self.children]
        return float(np.mean(vals))

    def finished_strong_count(self) -> int:
        return int(sum(c["finished_strong"] for c in self.children))

    def _obs(self):
        values = []
        for c in self.children:
            if c["enrolled"]:
                values += [c["learning"], c["attendance"], c["confidence"], c["home_pressure"]]
            else:
                values += [c["learning"], 0.0, 0.0, c["home_pressure"]]
        values += [
            self.budget, self.teachers / MAX_TEACHERS, self.quality, self.morale,
            self.trust, self.tech, self.gov_funding, self.term / MAX_TERMS,
        ]
        return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)

    def _info(self):
        return {
            "term": self.term,
            "dropouts": self.dropouts,
            "teacher_quits": self.quits,
            "teachers": self.teachers,
            "finished_strong": self.finished_strong_count(),
            "invalid_actions": self.invalid_actions,
            "end_reason": self.end_reason,
            "events": list(self.events),
        }

    def get_state(self) -> dict:
        """Full state as a plain dict (what a dashboard or app would read)."""
        return {
            "term": self.term,
            "max_terms": MAX_TERMS,
            "scenario": self.scenario,
            "school": {
                "budget": round(self.budget, 4),
                "teachers": self.teachers,
                "teacher_quality": round(self.quality, 4),
                "teacher_morale": round(self.morale, 4),
                "community_trust": round(self.trust, 4),
                "tech_level": round(self.tech, 4),
                "gov_funding": round(self.gov_funding, 4),
            },
            "children": [
                {
                    "name": c["name"],
                    "enrolled": c["enrolled"],
                    "learning": round(c["learning"], 4),
                    "attendance": round(c["attendance"], 4),
                    "confidence": round(c["confidence"], 4),
                    "home_pressure": round(c["home_pressure"], 4),
                    "finished_strong": c["finished_strong"],
                }
                for c in self.children
            ],
            "last_action": None if self.last_action is None else ACTION_NAMES[self.last_action],
            "last_reward": round(self.last_reward, 4),
            "total_reward": round(self.total_reward, 4),
            "dropouts": self.dropouts,
            "teacher_quits": self.quits,
            "events": list(self.events),
            "end_reason": self.end_reason,
            "observation": [round(float(v), 4) for v in self._obs()],
        }

    def get_state_json(self) -> str:
        """Full state as JSON. This is the serialization path for integration."""
        return json.dumps(self.get_state())


def obs_from_state(state: dict) -> np.ndarray:
    """Rebuild the 32-number observation from a state dict or JSON string.

    This lets an outside app (web or mobile) send JSON and get an action back.
    """
    if isinstance(state, str):
        state = json.loads(state)
    if "observation" in state:
        return np.asarray(state["observation"], dtype=np.float32)
    values = []
    for c in state["children"]:
        enrolled = c.get("enrolled", True)
        values += [c["learning"], c["attendance"] if enrolled else 0.0,
                   c["confidence"] if enrolled else 0.0, c["home_pressure"]]
    s = state["school"]
    values += [s["budget"], s["teachers"] / MAX_TEACHERS, s["teacher_quality"],
               s["teacher_morale"], s["community_trust"], s["tech_level"],
               s["gov_funding"], state["term"] / MAX_TERMS]
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
