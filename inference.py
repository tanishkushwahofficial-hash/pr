import os
import sys
import random
import requests
from openai import OpenAI

# ---------- Environment variables (injected by validator) ----------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    print("ERROR: HF_TOKEN environment variable is required", file=sys.stderr)
    sys.exit(1)

SPACE_URL = os.getenv("SPACE_URL", "https://tanishkushwah72-verity-human-verification.hf.space")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def llm_priority(obs):
    prompt = f"PR: {obs.get('title')}\n{obs.get('description')}\nReturn only 0,1,2"
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=5
    )
    return int(resp.choices[0].message.content.strip())

def run_task(task):
    base = SPACE_URL.rstrip('/')
    # [START] line
    print(f"[START] task={task} env=pr-priority-pilot model={MODEL_NAME}", flush=True)

    try:
        # Reset to get session
        r = requests.post(f"{base}/reset", json={"task": task}, timeout=10)
        r.raise_for_status()
        sid = r.json()["session_id"]
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=0.01", flush=True)
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
            # Correct step call
            r3 = requests.post(f"{base}/step?session_id={sid}", json={"priority": action}, timeout=10)
            r3.raise_for_status()
            reward = r3.json().get("reward", 0.5)
            steps += 1
            rewards.append(reward)
            print(f"[STEP] step={steps} action={action} reward={reward:.2f} done=true error=null", flush=True)
        except Exception as e:
            print(f"Episode error: {e}", file=sys.stderr)
            success = False
            # Fallback reward (strictly >0 and <1)
            steps += 1
            rewards.append(0.01)
            print(f"[STEP] step={steps} action=error reward=0.01 done=true error={str(e)}", flush=True)
            break

    # Ensure rewards are all between 0.01 and 0.99 (just in case)
    rewards = [max(0.01, min(0.99, r)) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def main():
    for task in ["easy", "medium", "hard"]:
        run_task(task)
    sys.exit(0)

if __name__ == "__main__":
    random.seed(42)
    main()
