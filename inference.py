import os
import json
import requests
import sys
from openai import OpenAI

# Required Environment variables injected by validator
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
SPACE_URL = os.environ.get("SPACE_URL", "https://tanishkushwah72-verity-human-verification.hf.space")

def log_start(task, env, model):
    print(f"[START] {json.dumps({'task': task, 'env': env, 'model': model})}", flush=True)

def log_step(step, action, reward, done, error=None):
    payload = {'step': step, 'action': action, 'reward': reward, 'done': done}
    if error:
        payload['error'] = error
    print(f"[STEP] {json.dumps(payload)}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] {json.dumps({'success': success, 'steps': steps, 'score': score, 'rewards': rewards})}", flush=True)

def llm_priority(client, obs):
    prompt = f"""You are a senior developer prioritizing pull requests.
PR Title: {obs.get('pr_title', '')}
Description: {obs.get('pr_description', '')}
Files changed: {obs.get('files_changed', 0)}
Labels: {obs.get('labels', [])}
Author: {obs.get('author', '')}

Return only an integer 0 (Low), 1 (Medium), or 2 (High)."""

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        content = resp.choices[0].message.content.strip()
        # Handle cases where LLM might include extra text
        import re
        match = re.search(r'\d+', content)
        if match:
             return int(match.group())
        return 1
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", file=sys.stderr)
        return 1  # Default to medium if it fails

def evaluate_task(client, task, episodes=3):
    try:
        # Create session
        reset_res = requests.post(f"{SPACE_URL}/reset", json={"task": task}, timeout=10)
        reset_res.raise_for_status()
        sess = reset_res.json().get("session_id")
    except Exception as e:
        print(f"[DEBUG] Error creating session for task {task}: {e}", file=sys.stderr)
        return 0.0, []

    total_reward = 0.0
    rewards = []
    
    for ep in range(episodes):
        try:
            # Reset environment for this episode
            obs_res = requests.post(f"{SPACE_URL}/reset", json={"session_id": sess, "task": task}, timeout=10)
            obs_res.raise_for_status()
            obs = obs_res.json().get("observation", {})
            
            action = llm_priority(client, obs)
            
            # Step the environment
            step_res = requests.post(f"{SPACE_URL}/step", json={"session_id": sess, "action": {"priority": action}}, timeout=10)
            step_res.raise_for_status()
            
            result = step_res.json()
            reward = result.get("reward", 0.0)
            done = result.get("done", True)
            
            total_reward += reward
            rewards.append(reward)
            
            log_step(step=ep + 1, action=str(action), reward=reward, done=done)
            
            # Re-create session for next episode if done
            if done and ep < episodes - 1:
                sess = requests.post(f"{SPACE_URL}/reset", json={"task": task}, timeout=10).json().get("session_id")

        except Exception as e:
            print(f"[DEBUG] Error during step {ep + 1} for task {task}: {e}", file=sys.stderr)
            log_step(step=ep + 1, action="error", reward=0.0, done=True, error=str(e))
            rewards.append(0.0)
            break
            
    avg_score = total_reward / episodes if episodes > 0 else 0.0
    return avg_score, rewards

def main():
    if not API_BASE_URL or not API_KEY:
        print("ERROR: API_BASE_URL and API_KEY environment variables must be set.", file=sys.stderr)
        # Even if missing, we must let it proceed or fail properly during evaluation?
        # The instructions say to use these injected variables, so let's continue and error at runtime.
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = ["easy", "medium", "hard"]
    
    for task in tasks:
        log_start(task=task, env="pr-priority-pilot", model=MODEL_NAME)
        score, rewards = evaluate_task(client, task, episodes=3)
        success = score > 0.0  # Or appropriately scaled
        log_end(success=success, steps=len(rewards), score=score, rewards=rewards)

if __name__ == "__main__":
    main()
