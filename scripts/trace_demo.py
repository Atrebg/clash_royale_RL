#!/usr/bin/env python3
import argparse
import os
import random
import sys
from pathlib import Path
from typing import List

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clash_royale_env import ClashRoyaleEnv


def collect_events(env: ClashRoyaleEnv, steps: int) -> List[str]:
    lines: List[str] = []
    for _ in range(steps):
        # pass action (no deploys); enemy deploys are disabled by max_elixir=0
        action_pass = env.num_cards * 2
        _, _, term, trunc, info = env.step(action_pass)
        lines.extend(info.get("events", []))
        if term or trunc:
            break
    return lines


def scenario_giant_vs_tower() -> List[str]:
    env = ClashRoyaleEnv()
    random.seed(0)
    env.reset(seed=0)
    # Disable enemy AI so elixir can still rise
    env.disable_enemy_ai = True
    env.spawn_by_name("Giant", lane_id=0, owner=0, pos=0)
    lines = ["=== Scenario: Giant vs Tower ==="]
    lines += collect_events(env, steps=6)
    return lines


def scenario_knight_vs_minions() -> List[str]:
    env = ClashRoyaleEnv()
    random.seed(1)
    env.reset(seed=1)
    env.disable_enemy_ai = True
    env.spawn_by_name("Cavaliere", lane_id=0, owner=0, pos=1)
    env.spawn_by_name("Sgherri", lane_id=0, owner=1, pos=2)
    lines = ["=== Scenario: Cavaliere vs Sgherri ==="]
    lines += collect_events(env, steps=4)
    return lines


def scenario_musketeer_range_air() -> List[str]:
    env = ClashRoyaleEnv()
    random.seed(2)
    env.reset(seed=2)
    env.disable_enemy_ai = True
    env.spawn_by_name("Moschettiere", lane_id=0, owner=0, pos=0)
    env.spawn_by_name("Sgherri", lane_id=0, owner=1, pos=3)
    lines = ["=== Scenario: Moschettiere (range) vs Sgherri (air) ==="]
    lines += collect_events(env, steps=3)
    return lines


def build_trace() -> List[str]:
    all_lines: List[str] = []
    for part in (scenario_giant_vs_tower, scenario_knight_vs_minions, scenario_musketeer_range_air):
        all_lines.extend(part())
    return all_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="update golden expected trace")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected_path = os.path.join(repo_root, "scripts", "expected_trace.txt")

    lines = build_trace()
    text = "\n".join(lines) + "\n"

    if args.update or not os.path.exists(expected_path):
        with open(expected_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote golden trace to {expected_path}")
        return

    with open(expected_path, "r", encoding="utf-8") as f:
        expected = f.read()

    if text == expected:
        print("Trace matches expected ✅")
    else:
        print("Trace differs from expected ❌\n")
        # Print a minimal diff-like output
        cur_lines = text.splitlines()
        exp_lines = expected.splitlines()
        max_len = max(len(cur_lines), len(exp_lines))
        for i in range(max_len):
            cur = cur_lines[i] if i < len(cur_lines) else "<missing>"
            exp = exp_lines[i] if i < len(exp_lines) else "<missing>"
            if cur != exp:
                print(f"Line {i+1}:")
                print(f"  expected: {exp}")
                print(f"  current : {cur}")
                break


if __name__ == "__main__":
    main()
