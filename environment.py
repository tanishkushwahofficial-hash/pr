import random
from typing import Tuple, Dict, Any
from pydantic import BaseModel

class Observation(BaseModel):
    title: str
    description: str

class Action(BaseModel):
    priority: int

class State(BaseModel):
    observation: Observation
    done: bool

TASKS = {
    "easy": [
        {"title": "Fix typo in README", "desc": "Correct spelling error", "truth": 0},
        {"title": "Add feature toggle", "desc": "New user setting", "truth": 1},
        {"title": "URGENT: Fix login crash", "desc": "Production hotfix", "truth": 2},
    ],
    "medium": [
        {"title": "Security patch for SQL injection", "desc": "Critical vulnerability", "truth": 2},
        {"title": "Refactor logging module", "desc": "Code cleanup", "truth": 1},
        {"title": "Update button styles", "desc": "UI tweak", "truth": 0},
    ],
    "hard": [
        {"title": "HOTFIX: Payment gateway timeout", "desc": "Customers cannot pay", "truth": 2},
        {"title": "Migrate database schema", "desc": "Add columns without downtime", "truth": 1},
        {"title": "Update dependencies", "desc": "Regular maintenance", "truth": 0},
    ]
}

class PrioritizerEnv:
    def __init__(self):
        self.task_name = "easy"
        self.pool = TASKS["easy"]
        self.current = None
        self.done = False

    def set_task(self, difficulty: str):
        self.task_name = difficulty
        self.pool = TASKS[difficulty]

    def reset(self) -> Observation:
        self.current = random.choice(self.pool).copy()
        self.done = False
        return Observation(title=self.current["title"], description=self.current["desc"])

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode already done")
        pred = action.priority
        truth = self.current["truth"]
        # Base rewards – strictly between 0 and 1
        if pred == truth:
            base = 0.75
        elif abs(pred - truth) == 1:
            base = 0.50
        else:
            base = 0.25
        # Add a tiny, deterministic offset to avoid exact 0.0 or 1.0
        offset = (hash(self.current["title"]) % 100) / 1000.0  # 0.000 to 0.099
        reward = base + offset
        # Final clamp to (0.01, 0.99)
        reward = max(0.01, min(0.99, reward))
        self.done = True
        next_obs = self.reset()
        info = {"true_priority": truth}
        return next_obs, reward, self.done, info

    def state(self) -> State:
        if self.current:
            obs = Observation(title=self.current["title"], description=self.current["desc"])
            return State(observation=obs, done=self.done)
        return State(observation=Observation(title="", description=""), done=True)
