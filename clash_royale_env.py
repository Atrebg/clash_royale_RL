import random
from typing import Dict, List, Tuple, Any, Union

import numpy as np
try:
    import gym
    from gym import spaces
except ImportError:  # pragma: no cover - used when gym is unavailable
    gym = None
    spaces = None


class ClashRoyaleEnv(gym.Env if gym else object):
    """Simplified text-based Clash Royale environment for RL agents.

    The arena has two lanes. Units march toward the enemy base. Each unit has
    hit points (hp) and damage (dmg). Players spend elixir to deploy units and
    receive rewards for damaging the opponent's base.
    """

    metadata = {"render.modes": ["ansi"]}

    def __init__(self) -> None:
        if gym is None:
            raise ImportError("gym is required to use ClashRoyaleEnv")
        super().__init__()

        # Game configuration
        self.max_turns = 100
        self.turn = 0

        self.base_hp = [100, 100]  # [player, enemy]
        self.elixir = [5, 5]
        self.max_elixir = 10

        # Unit/card definitions
        self.cards: Dict[int, Dict[str, int]] = {
            0: {"name": "Soldier", "cost": 1, "hp": 10, "dmg": 5},
            1: {"name": "Giant", "cost": 3, "hp": 30, "dmg": 10},
        }
        self.num_cards = len(self.cards)

        # Action space: (card, lane) pairs plus a pass action
        self.action_space = spaces.Discrete(self.num_cards * 2 + 1)

        # Observation space: simple numeric vector
        self.observation_space = spaces.Box(low=0, high=100, shape=(10,), dtype=np.int32)

        # Lanes store active units
        self.lanes: List[List[Dict[str, int]]] = [[], []]

    # ------------------------------------------------------------------
    # RL API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.turn = 0
        self.base_hp = [100, 100]
        self.elixir = [5, 5]
        self.lanes = [[], []]
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        reward = 0.0
        done = False

        # Regenerate elixir
        for i in range(2):
            self.elixir[i] = min(self.max_elixir, self.elixir[i] + 1)

        # Player action
        if action < self.num_cards * 2:
            card_id = action // 2
            lane_id = action % 2
            card = self.cards[card_id]
            if self.elixir[0] >= card["cost"]:
                self.elixir[0] -= card["cost"]
                self.lanes[lane_id].append({
                    "hp": card["hp"],
                    "dmg": card["dmg"],
                    "pos": 0,
                    "owner": 0,
                })

        # Enemy random action
        enemy_action = random.randint(0, self.num_cards * 2)
        if enemy_action < self.num_cards * 2:
            card_id = enemy_action // 2
            lane_id = enemy_action % 2
            card = self.cards[card_id]
            if self.elixir[1] >= card["cost"]:
                self.elixir[1] -= card["cost"]
                self.lanes[lane_id].append({
                    "hp": card["hp"],
                    "dmg": card["dmg"],
                    "pos": 5,
                    "owner": 1,
                })

        # Resolve movement and combat for each lane
        for lane_id, lane in enumerate(self.lanes):
            # Move units
            for unit in lane:
                unit["pos"] += 1 if unit["owner"] == 0 else -1

            # Sort by position
            lane.sort(key=lambda u: u["pos"])

            # Combat between adjacent enemy units
            remove: List[int] = []
            for i in range(len(lane) - 1):
                u1, u2 = lane[i], lane[i + 1]
                if u1["owner"] != u2["owner"] and abs(u1["pos"] - u2["pos"]) <= 1:
                    u1["hp"] -= u2["dmg"]
                    u2["hp"] -= u1["dmg"]

            # Remove dead units
            for i, u in enumerate(lane):
                if u["hp"] <= 0:
                    remove.append(i)
            for idx in reversed(remove):
                lane.pop(idx)

            # Units reaching base
            remove.clear()
            for i, u in enumerate(lane):
                if u["owner"] == 0 and u["pos"] >= 5:
                    self.base_hp[1] -= u["dmg"]
                    reward += u["dmg"]
                    remove.append(i)
                elif u["owner"] == 1 and u["pos"] <= 0:
                    self.base_hp[0] -= u["dmg"]
                    reward -= u["dmg"]
                    remove.append(i)
            for idx in reversed(remove):
                lane.pop(idx)

        # End conditions
        self.turn += 1
        if self.base_hp[0] <= 0 or self.base_hp[1] <= 0 or self.turn >= self.max_turns:
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.int32)
        obs[0], obs[1] = self.base_hp
        obs[2], obs[3] = self.elixir
        obs[4] = len([u for u in self.lanes[0] if u["owner"] == 0])
        obs[5] = len([u for u in self.lanes[0] if u["owner"] == 1])
        obs[6] = len([u for u in self.lanes[1] if u["owner"] == 0])
        obs[7] = len([u for u in self.lanes[1] if u["owner"] == 1])
        obs[8] = self.turn
        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, mode: str = "ansi") -> None:
        if mode != "ansi":
            raise NotImplementedError("Only ANSI rendering is supported")
        print(f"Turn {self.turn}")
        print(f"Player base: {self.base_hp[0]} HP, elixir {self.elixir[0]}")
        print(f"Enemy base: {self.base_hp[1]} HP, elixir {self.elixir[1]}")
        print(f"Lane 0: {self.lanes[0]}")
        print(f"Lane 1: {self.lanes[1]}")


if __name__ == "__main__":
    if gym is None:
        raise SystemExit("gym is not installed. Install gym to run the demo.")
    env = ClashRoyaleEnv()
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    print("Episode finished with reward", total_reward)
