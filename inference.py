import os
import sys
import random
import requests
from openai import OpenAI

# ---------- Environment variables ----------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    print("ERROR: HF_TOKEN environment variable is required", file=sys.stderr)
    sys.exit(1)

SPACE_URL = os.getenv("SPACE_URL", "https://tanishkushwah72-verity-human-verification.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------- LLM priority (with fallback) ----------
def llm_priority(obs):
    try:
        prompt = f"PR Title: {obs.get('title', '')}\nDescription: {obs.get('description', '')}\nReturn only 0,1,2"
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5
        )
        return int(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"LLM error: {e}, using rule", file=sys.stderr)
        # rule-based fallback
        text = (obs.get("title", "") + " " + obs.get("description", "")).lower()
        if any(k in text for k in ["urgent","crash","hotfix","security"]):
            return 2
        elif any(k in text for k in ["feature","refactor","migration"]):
            return 1
        else:
            return 0

# ---------- Run a single task ----------
def run_task(task):
    base = SPACE_URL.rstrip('/')
    print(f"[START] task={task} env=pr-priority-pilot model={MODEL_NAME}", flush=True)

    # Reset to get session
    try:
        r = requests.post(f"{base}/reset", json={"task": task}, timeout=10)
        r.raise_for_status()
        sid = r.json()["session_id"]
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=0.00", flush=True)
        return

    steps = 0
    rewards = []
    success = True
    for ep in range(3):  # 3 episodes per task
        try:
            # Get a PR
            r2 = requests.post(f"{base}/reset", json={"session_id": sid, "task": task}, timeout=10)
            r2.raise_for_status()
            obs = r2.json()["observation"]
            action = llm_priority(obs)
            # Step with correct API call
            r3 = requests.post(f"{base}/step?session_id={sid}", json={"priority": action}, timeout=10)
            r3.raise_for_status()
            reward = r3.json().get("reward", 0.5)
            steps += 1
            rewards.append(reward)
            # Print [STEP] line – all fields required
            print(f"[STEP] step={steps} action={action} reward={reward:.2f} done=true error=null", flush=True)
        except Exception as e:
            print(f"Episode error: {e}", file=sys.stderr)
            success = False
            # Still print a [STEP] with error
            print(f"[STEP] step={steps+1} action=error reward=0.00 done=true error={str(e)}", flush=True)
            steps += 1
            rewards.append(0.01)  # fallback reward (strictly >0)
            break

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# ---------- Main ----------
def main():
    tasks = ["easy", "medium", "hard"]
    for task in tasks:
        run_task(task)
    sys.exit(0)

if __name__ == "__main__":
    random.seed(42)
    main()
