import random
from typing import Tuple, Dict
from pydantic import BaseModel

class Observation(BaseModel):
    title: str
    description: str

class Action(BaseModel):
    priority: int

class State(BaseModel):
    observation: Observation
    done: bool

# Three tasks with ground truth priorities
TASKS = {
    "easy": [
        {"title": "Fix typo", "desc": "Spelling error", "truth": 0},
        {"title": "Add feature", "desc": "New toggle", "truth": 1},
        {"title": "Urgent crash", "desc": "Hotfix", "truth": 2},
    ],
    "medium": [
        {"title": "Security patch", "desc": "SQL injection", "truth": 2},
        {"title": "Refactor code", "desc": "Cleanup", "truth": 1},
        {"title": "UI tweak", "desc": "Button color", "truth": 0},
    ],
    "hard": [
        {"title": "Hotfix payment", "desc": "Timeout", "truth": 2},
        {"title": "DB migration", "desc": "Add columns", "truth": 1},
        {"title": "Dependency update", "desc": "Bump versions", "truth": 0},
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
        # Base rewards (strictly between 0 and 1)
        if pred == truth:
            base = 0.75
        elif abs(pred - truth) == 1:
            base = 0.50
        else:
            base = 0.25
        # Add a tiny deterministic offset to avoid exact 0.0 or 1.0
        offset = (hash(self.current["title"]) % 100) / 1000.0  # 0.000 to 0.099
        reward = base + offset
        # Clamp to safe range (0.01, 0.99)
        reward = max(0.01, min(0.99, reward))
        self.done = True
        # Return a new observation (next PR)
        next_obs = self.reset()
        info = {"true_priority": truth}
        return next_obs, reward, self.done, info

    def state(self) -> State:
        if self.current:
            obs = Observation(title=self.current["title"], description=self.current["desc"])
            return State(observation=obs, done=self.done)
        return State(observation=Observation(title="", description=""), done=True)
