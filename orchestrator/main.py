import json, subprocess, time
from planner import next_action

def run_step(step):
    cmd = ["python","-m","runner.cli"]
    mapping = {
      "ingest":["ingest-step"],
      "ai":["ai-base-step"],
      "eda":["eda-step"],
      "plan":["ai-plan-step"],
      "features":["features-step"],
      "model_search":["model-search-step"],
      "hpo":["hpo-step"],
      "evaluate":["evaluate-step"],
      "select_and_register":["select-and-register-step","--min-recall","0.6"],
      "report":["report-step"]
    }
    return subprocess.call(cmd + mapping[step]) == 0

def main():
    state = {}
    try:
        state = json.load(open(".state.json"))
    except FileNotFoundError:
        pass
    while True:
        action = next_action(state)
        print("Planner →", action)
        if action=="done": break
        ok = run_step(action)
        state = json.load(open(".state.json"))
        if not ok: raise SystemExit(f"step {action} failed")
        time.sleep(1)

if __name__ == "__main__":
    main()
