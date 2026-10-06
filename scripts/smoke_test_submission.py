"""
Smoke-test the packaged submission.tar.gz.

Extracts submission.tar.gz into a clean TemporaryDirectory, imports the entrypoint
main.py via kaggle_environments, and runs 3 local episodes (2 vs random, 1 vs SEP).
Verifies that there are no import/runtime errors and outputs final banks.
"""

import os
import sys
import tempfile
import tarfile
from kaggle_environments import make

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.benchmark import load_opponent_callable


def main():
    tar_path = os.path.join(PROJECT_ROOT, "submission.tar.gz")
    if not os.path.exists(tar_path):
        print(f"ERROR: {tar_path} not found!")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Extracting {tar_path} ({os.path.getsize(tar_path):,} bytes) to {tmpdir}...")
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall(tmpdir)

        agent_file = os.path.join(tmpdir, "main.py")
        if not os.path.exists(agent_file):
            print("ERROR: main.py not found in extracted archive!")
            sys.exit(1)
        print("Agent entrypoint main.py verified present.")

        # Test 1: vs random, seed 1000
        print("Running Smoke Test 1: vs random (seed 1000)...")
        env1 = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1000, "randomSeed": 1000}, debug=True)
        env1.run([agent_file, "random"])
        r0_1 = env1.steps[-1][0]["reward"]
        r1_1 = env1.steps[-1][1]["reward"]
        s0_1 = env1.state[0].status
        s1_1 = env1.state[1].status
        print(f"Smoke Test 1 Result: Agent=${r0_1:,.1f} ({s0_1}) | Random=${r1_1:,.1f} ({s1_1})")
        assert s0_1 == "DONE", f"Agent ended with unexpected status {s0_1}"

        # Test 2: vs random, seed 1001
        print("Running Smoke Test 2: vs random (seed 1001)...")
        env2 = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1001, "randomSeed": 1001}, debug=True)
        env2.run([agent_file, "random"])
        r0_2 = env2.steps[-1][0]["reward"]
        r1_2 = env2.steps[-1][1]["reward"]
        s0_2 = env2.state[0].status
        s1_2 = env2.state[1].status
        print(f"Smoke Test 2 Result: Agent=${r0_2:,.1f} ({s0_2}) | Random=${r1_2:,.1f} ({s1_2})")
        assert s0_2 == "DONE", f"Agent ended with unexpected status {s0_2}"

        # Test 3: vs SEP, seed 1000
        print("Running Smoke Test 3: vs SEP (seed 1000)...")
        sep_callable = load_opponent_callable("kaggriculture-structured-economic-policy")
        env3 = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 1000, "randomSeed": 1000}, debug=True)
        env3.run([agent_file, sep_callable])
        r0_3 = env3.steps[-1][0]["reward"]
        r1_3 = env3.steps[-1][1]["reward"]
        s0_3 = env3.state[0].status
        s1_3 = env3.state[1].status
        print(f"Smoke Test 3 Result: Agent=${r0_3:,.1f} ({s0_3}) | SEP=${r1_3:,.1f} ({s1_3})")
        assert s0_3 == "DONE", f"Agent ended with unexpected status {s0_3}"

        print("\nAll 3 smoke test episodes passed cleanly! Zero errors.")


if __name__ == "__main__":
    main()
