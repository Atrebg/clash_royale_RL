import json
import os
import random
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

# Try to use Gymnasium spaces (preferred by most modern RL libs and PufferLib)
try:  # pragma: no cover - optional dependency
    from gymnasium import spaces  # type: ignore
except Exception:  # Fallback: minimal spaces implementation for local runs
    class _Discrete:
        def __init__(self, n: int) -> None:
            self.n = int(n)

        def sample(self) -> int:
            return random.randint(0, self.n - 1)

    class _Box:
        def __init__(self, low, high, shape, dtype) -> None:
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class spaces:  # type: ignore
        Discrete = _Discrete
        Box = _Box


class ClashRoyaleEnv:
    """Simplified text-based Clash Royale environment (PufferLib-friendly).

    Single-agent, two lanes. Units march toward the enemy base. Each unit has
    hit points (hp) and damage (dmg). Players spend elixir to deploy units and
    receive rewards for damaging the opponent's base.

    API: uses Gymnasium-style signatures (reset -> (obs, info),
    step -> (obs, reward, terminated, truncated, info)) to work smoothly
    with PufferLib and modern RL tooling.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, cards: Optional[Any] = None, cards_path: Optional[str] = None, config_path: Optional[str] = None) -> None:
        # Game configuration
        self.max_turns = 100
        self.turn = 0
        self._np_random = np.random.default_rng()
        self.debug = False
        self._next_unit_id = 0
        self.disable_enemy_ai = False
        # Core tunables (may be overridden by unified config)
        self.base_hp_init = [100, 100]  # legacy, not used for win condition
        self.base_hp = [100, 100]
        self.tower_hp_init = 100  # HP per tower (per lane, per side)
        self.tower_hp: List[List[int]] = [[self.tower_hp_init, self.tower_hp_init],
                                          [self.tower_hp_init, self.tower_hp_init]]
        self.starting_elixir = 0
        self.elixir = [0, 0]
        self.max_elixir = 10
        self.tower_dmg = 8
        self.lane_max_pos = 5  # overridden by config
        # Geometry defaults (overridden by config): 4 | 3 (tower) | 7 | 1 (bridge) | 7 | 3 (tower) | 4
        self.side_tiles = 14
        self.bridge_tiles = 1
        self.tower_start = 4
        self.tower_width = 3
        # Derived positions (computed in _load_full_config)
        self.tower_player_start = 1
        self.tower_player_end = 1
        self.tower_enemy_start = 1
        self.tower_enemy_end = 1
        # Spawn (will be computed: start-of-side) and adjacency tiles toward bridge
        self.tower_player_pos = 1
        self.tower_enemy_pos = 1
        self.player_start = 0
        self.enemy_start = self.lane_max_pos
        self.player_adjacent = 2
        self.enemy_adjacent = 2
        # Tower persistent targets: [player_tower_targets, enemy_tower_targets] per lane
        self._tower_target_ids: List[List[Optional[int]]] = [[None, None], [None, None]]

        # Try unified config (config/game.json) for env tunables and AI defaults
        self._load_full_config(config_path)

        # Unit/card definitions loaded from config
        self.cards: List[Dict[str, Any]] = self._load_cards(cards, cards_path)
        self.card_name_to_index = {c["name"]: i for i, c in enumerate(self.cards)}
        self.num_cards = len(self.cards)

        # Action space: (card, lane) pairs plus a pass action
        self.action_space = spaces.Discrete(self.num_cards * 2 + 1)

        # Observation space: numeric vector
        # [p_tower0, p_tower1, e_tower0, e_tower1, elixir_p, elixir_e,
        #  p_lane0_units, e_lane0_units, p_lane1_units, e_lane1_units, turn]
        self.observation_space = spaces.Box(low=0, high=10000, shape=(11,), dtype=np.int32)

        # Lanes store active units
        self.lanes: List[List[Dict[str, Any]]] = [[], []]

    # ------------------------------------------------------------------
    # RL API (Gymnasium style for PufferLib compatibility)
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self.turn = 0
        self._next_unit_id = 0
        self.base_hp = [int(self.base_hp_init[0]), int(self.base_hp_init[1])]
        self.tower_hp = [[int(self.tower_hp_init), int(self.tower_hp_init)],
                         [int(self.tower_hp_init), int(self.tower_hp_init)]]
        self.elixir = [int(self.starting_elixir), int(self.starting_elixir)]
        self.lanes = [[], []]
        self._tower_target_ids = [[None, None], [None, None]]
        return self._get_obs(), {}

    def step(self, action: int | List[Any] | Tuple[Any, ...]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0
        terminated = False
        truncated = False

        events: List[str] = []
        # Regenerate elixir
        for i in range(2):
            self.elixir[i] = min(self.max_elixir, self.elixir[i] + 1)
        events.append(f"Turn {self.turn+1}: Elixir -> P:{self.elixir[0]} E:{self.elixir[1]}")

        # Player action: supports single int or multiple placements
        placements = self._parse_placements(action)
        if placements:
            used = set()
            for item in placements:
                # item may be (card_id, lane_id) or (card_id, lane_id, pos)
                if isinstance(item, tuple) and len(item) in (2, 3):
                    card_id, lane_id = int(item[0]), int(item[1])
                    pos_override = int(item[2]) if len(item) == 3 else None
                else:
                    # backward safety (parser may have given plain tuple)
                    try:
                        card_id, lane_id = item  # type: ignore[misc]
                        pos_override = None
                    except Exception:
                        continue
                if not (0 <= card_id < self.num_cards) or lane_id not in (0, 1):
                    continue
                key = (card_id, lane_id)
                if key in used:
                    continue
                used.add(key)
                card = self.cards[card_id]
                if self.elixir[0] >= card["cost"]:
                    self.elixir[0] -= card["cost"]
                    # Determine deploy tile (respect config-allowed tiles)
                    deploy_pos = self._select_deploy_pos(owner=0, requested=pos_override)
                    self._spawn_unit(card_id, lane_id, owner=0, pos=deploy_pos, pay_cost=False, events=events)
                    events.append(f"Player deploy {card['name']} lane {lane_id}")

        if not self.disable_enemy_ai:
            enemy_action = random.randint(0, self.num_cards * 2)
            if enemy_action < self.num_cards * 2:
                card_id = enemy_action // 2
                lane_id = enemy_action % 2
                card = self.cards[card_id]
                if self.elixir[1] >= card["cost"]:
                    self.elixir[1] -= card["cost"]
                    # Spawn at the beginning of enemy's side
                    self._spawn_unit(card_id, lane_id, owner=1, pos=self.enemy_start, pay_cost=False, events=events)
                    events.append(f"Enemy deploy {card['name']} lane {lane_id}")

        # Resolve movement and combat for each lane
        for lane_id, lane in enumerate(self.lanes):
            # Move units with target lock and push logic (units at tower stay fixed)
            # Build id->unit mapping for this lane
            id_to_unit = {u.get("id"): u for u in lane if "id" in u}

            def ground_unit_at(pos: int) -> Optional[Dict[str, Any]]:
                for u in lane:
                    if int(u.get("pos", -999)) == pos and not bool(u.get("is_air", False)) and not u.get("at_tower", False):
                        return u
                return None

            def attempt_push(next_pos: int, direction: int, pusher_hp: int) -> bool:
                # Try to push a chain of ground units one step in 'direction' if all are lighter
                chain: List[Dict[str, Any]] = []
                cur = next_pos
                while True:
                    blk = ground_unit_at(cur)
                    if blk is None:
                        break
                    if pusher_hp <= int(blk.get("max_hp", blk.get("hp", 0))):
                        return False
                    # cannot push units fixed at tower
                    if blk.get("at_tower", False):
                        return False
                    chain.append(blk)
                    cur += direction
                    if cur < 0 or cur > (self.lane_max_pos if hasattr(self, "lane_max_pos") else 5):
                        return False
                # free spot found; push in reverse order
                for blk in reversed(chain):
                    blk["pos"] += direction
                return True

            for unit in lane:
                if unit.get("at_tower", False):
                    continue

                moved = False
                target_id = unit.get("target_id")
                target = id_to_unit.get(target_id) if target_id is not None else None

                # Determine movement intent
                if unit.get("behavior", "nearest") != "tower_only" and target is not None:
                    # Validate target compatibility
                    target_type = "air" if target.get("is_air", False) else "ground"
                    if target_type not in unit.get("targets", {"ground"}):
                        target = None
                        unit.pop("target_id", None)

                old = unit["pos"]
                # Do not move on the same turn the unit was spawned
                if unit.get("spawn_turn") == self.turn:
                    continue
                direction = 1 if unit["owner"] == 0 else -1
                bridge_tile = int(self.lane_max_pos // 2)
                if unit.get("behavior", "nearest") == "tower_only" or target is None:
                    # Default marching toward enemy base
                    desired = old + direction
                    if bool(unit.get("is_air", False)):
                        unit["pos"] = desired
                        moved = True
                    else:
                        blocker = ground_unit_at(desired)
                        if blocker is None:
                            unit["pos"] = desired
                            moved = True
                        else:
                            # Bridge rule: only on the bridge tile we attempt push; elsewhere we bypass (no block)
                                if desired == bridge_tile:
                                    if attempt_push(desired, direction, int(unit.get("max_hp", unit.get("hp", 0)))):
                                        unit["pos"] = desired
                                        moved = True
                                else:
                                    # bypass laterally (no block outside bridge)
                                    unit["pos"] = desired
                                    moved = True
                else:
                    # Chase locked target until within range
                    dist = abs(unit["pos"] - target["pos"])
                    rng = int(unit.get("range", 1))
                    if dist > rng:
                        # move 1 step toward target
                        step = 1 if target["pos"] > unit["pos"] else -1
                        desired = old + step
                        if bool(unit.get("is_air", False)):
                            unit["pos"] = desired
                            moved = True
                        else:
                            blocker = ground_unit_at(desired)
                            if blocker is None:
                                unit["pos"] = desired
                                moved = True
                            else:
                                if desired == bridge_tile:
                                    if attempt_push(desired, step, int(unit.get("max_hp", unit.get("hp", 0)))):
                                        unit["pos"] = desired
                                        moved = True
                                else:
                                    # bypass outside bridge
                                    unit["pos"] = desired
                                    moved = True
                    # else engaged, do not move

                # clamp
                unit["pos"] = max(0, min(self.lane_max_pos if hasattr(self, "lane_max_pos") else 5, unit["pos"]))
                if moved and old != unit["pos"]:
                    events.append(f"Lane {lane_id}: Unit#{unit.get('id','?')} {unit.get('name','?')} moved {old}->{unit['pos']}")

            # Sort by position
            lane.sort(key=lambda u: u["pos"])

            # Ranged/air-ground combat with target lock: units keep target until it dies
            remove: List[int] = []
            damage = [0 for _ in lane]
            # Build id->index for current lane after sort
            id_to_idx = {u.get("id"): idx for idx, u in enumerate(lane) if "id" in u}
            for i, u in enumerate(lane):
                if u.get("at_tower", False):
                    continue  # units at tower only hit bases, not troops
                if u.get("behavior", "nearest") == "tower_only":
                    continue  # ignores troops

                # Try to keep locked target
                tgt_obj = None
                tgt_id = u.get("target_id")
                if tgt_id is not None and tgt_id in id_to_idx:
                    v = lane[id_to_idx[tgt_id]]
                    if u["owner"] != v["owner"]:
                        ttype = "air" if v.get("is_air", False) else "ground"
                        if ttype in u.get("targets", {"ground"}):
                            tgt_obj = v
                        else:
                            u.pop("target_id", None)
                    else:
                        u.pop("target_id", None)

                # Acquire new target if needed
                best_j = None
                best_dist = None
                if tgt_obj is None:
                    for j, v in enumerate(lane):
                        if i == j or u["owner"] == v["owner"]:
                            continue
                        ttype = "air" if v.get("is_air", False) else "ground"
                        if ttype not in u.get("targets", {"ground"}):
                            continue
                        dist = abs(u["pos"] - v["pos"]) 
                        if best_dist is None or dist < best_dist:
                            best_dist = dist
                            best_j = j
                    if best_j is not None:
                        tgt_obj = lane[best_j]
                        u["target_id"] = tgt_obj.get("id")
                else:
                    # compute distance to locked target
                    best_j = id_to_idx.get(tgt_obj.get("id"))
                    best_dist = abs(u["pos"] - tgt_obj["pos"]) if best_j is not None else None

                # Deal damage if within range
                if tgt_obj is not None and best_j is not None and best_dist is not None:
                    rng = int(u.get("range", 1))
                    if best_dist <= rng:
                        damage[best_j] += u["dmg"]
                        events.append(
                            f"Lane {lane_id}: Unit#{u.get('id','?')} {u.get('name','?')} hits Unit#{tgt_obj.get('id','?')} {tgt_obj.get('name','?')} for {u['dmg']} (dist {best_dist})"
                        )
            # apply simultaneous damage
            for idx, d in enumerate(damage):
                if d:
                    lane[idx]["hp"] -= d

            # Remove dead units
            for i, u in enumerate(lane):
                if u["hp"] <= 0:
                    remove.append(i)
                    events.append(f"Lane {lane_id}: Unit#{u.get('id','?')} {u.get('name','?')} died")
            for idx in reversed(remove):
                lane.pop(idx)

            # Units reaching tower adjacency: clamp and mark at_tower
            for u in lane:
                adj_enemy = self.enemy_adjacent
                adj_player = self.player_adjacent
                if u["owner"] == 0 and u["pos"] >= adj_enemy:
                    if not u.get("at_tower", False):
                        u["at_tower"] = True
                        u["pos"] = adj_enemy
                        u["arrival_turn"] = self.turn
                        events.append(f"Lane {lane_id}: Unit#{u.get('id','?')} {u.get('name','?')} reached ENEMY TOWER RANGE")
                elif u["owner"] == 1 and u["pos"] <= adj_player:
                    if not u.get("at_tower", False):
                        u["at_tower"] = True
                        u["pos"] = adj_player
                        u["arrival_turn"] = self.turn
                        events.append(f"Lane {lane_id}: Unit#{u.get('id','?')} {u.get('name','?')} reached PLAYER TOWER RANGE")

            # Units at tower deal damage to bases
            for u in lane:
                if not u.get("at_tower", False):
                    continue
                if u["owner"] == 0:
                    # Player unit damages ENEMY tower in this lane
                    self.tower_hp[1][lane_id] -= u["dmg"]
                    reward += u["dmg"]
                    events.append(f"Lane {lane_id}: Unit#{u.get('id','?')} {u.get('name','?')} hits ENEMY TOWER for {u['dmg']}")
                else:
                    # Enemy unit damages PLAYER tower in this lane
                    self.tower_hp[0][lane_id] -= u["dmg"]
                    reward -= u["dmg"]
                    events.append(f"Lane {lane_id}: Unit#{u.get('id','?')} {u.get('name','?')} hits PLAYER TOWER for {u['dmg']}")

            # Towers shoot back within half-field range, persisting target until it dies
            # Define midline as the half tile boundary (e.g., 2.5 for lane 0..5)
            half_pos = (self.lane_max_pos / 2.0)

            # Player tower (left) targets only enemies strictly on left half
            enemy_in_range = [u for u in lane if u["owner"] == 1 and u["pos"] < half_pos]
            if enemy_in_range:
                # Prefer current target if still in range
                cur = self._tower_target_ids[0][lane_id]
                target = None
                if cur is not None:
                    for u in enemy_in_range:
                        if u.get("id") == cur:
                            target = u
                            break
                if target is None:
                    # choose closest to player tower (smallest pos)
                    target = min(enemy_in_range, key=lambda u: u["pos"])
                    self._tower_target_ids[0][lane_id] = target.get("id")
                target["hp"] -= self.tower_dmg
                events.append(f"Lane {lane_id}: PLAYER TOWER hits Unit#{target.get('id','?')} {target.get('name','?')} for {self.tower_dmg}")

            # Enemy tower (right) targets only players strictly on right half
            player_in_range = [u for u in lane if u["owner"] == 0 and u["pos"] > half_pos]
            if player_in_range:
                cur = self._tower_target_ids[1][lane_id]
                target = None
                if cur is not None:
                    for u in player_in_range:
                        if u.get("id") == cur:
                            target = u
                            break
                if target is None:
                    # choose closest to enemy tower (largest pos)
                    target = max(player_in_range, key=lambda u: u["pos"])
                    self._tower_target_ids[1][lane_id] = target.get("id")
                target["hp"] -= self.tower_dmg
                events.append(f"Lane {lane_id}: ENEMY TOWER hits Unit#{target.get('id','?')} {target.get('name','?')} for {self.tower_dmg}")

            # Remove units killed by towers
            remove = []
            for i, u in enumerate(lane):
                if u["hp"] <= 0:
                    remove.append(i)
                    events.append(f"Lane {lane_id}: Unit#{u.get('id','?')} {u.get('name','?')} died")
                    if self._tower_target_ids[0][lane_id] == u.get("id"):
                        self._tower_target_ids[0][lane_id] = None
                    if self._tower_target_ids[1][lane_id] == u.get("id"):
                        self._tower_target_ids[1][lane_id] = None
            for idx in reversed(remove):
                lane.pop(idx)

        # End conditions
        self.turn += 1
        # Win condition: destroy any single enemy tower
        if any(hp <= 0 for hp in self.tower_hp[0]) or any(hp <= 0 for hp in self.tower_hp[1]):
            terminated = True
        if self.turn >= self.max_turns:
            truncated = True

        info = {"events": events, "base_hp": tuple(self.base_hp), "elixir": tuple(self.elixir)}
        if self.debug:
            for e in events:
                print(e)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(11, dtype=np.int32)
        # Per-tower HP by lane and side
        obs[0] = max(0, int(self.tower_hp[0][0]))  # player lane0
        obs[1] = max(0, int(self.tower_hp[0][1]))  # player lane1
        obs[2] = max(0, int(self.tower_hp[1][0]))  # enemy lane0
        obs[3] = max(0, int(self.tower_hp[1][1]))  # enemy lane1
        obs[4], obs[5] = self.elixir
        obs[6] = len([u for u in self.lanes[0] if u["owner"] == 0])
        obs[7] = len([u for u in self.lanes[0] if u["owner"] == 1])
        obs[8] = len([u for u in self.lanes[1] if u["owner"] == 0])
        obs[9] = len([u for u in self.lanes[1] if u["owner"] == 1])
        obs[10] = self.turn
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

    # ------------------------------------------------------------------
    # Helpers for testing/debugging
    # ------------------------------------------------------------------
    def set_debug(self, enabled: bool = True) -> None:
        self.debug = bool(enabled)

    def _select_deploy_pos(self, owner: int, requested: Optional[int]) -> int:
        # Compute default allowed tiles (owner side only) if not provided via config
        mid = int(self.lane_max_pos // 2)
        if owner == 0:
            allowed = getattr(self, "deploy_player_tiles", None)
            if not allowed:
                allowed = set(range(0, mid))  # player's side only
            default_pos = self.player_start
            ally_adjacent = self.player_adjacent
        else:
            allowed = getattr(self, "deploy_enemy_tiles", None)
            if not allowed:
                allowed = set(range(mid + 1, self.lane_max_pos + 1))  # enemy side only
            default_pos = self.enemy_start
            ally_adjacent = self.enemy_adjacent

        # No request: deploy at default (start tile)
        if requested is None:
            return default_pos

        req = int(requested)
        # If request is on opponent side, snap to ally tile next to bridge
        if owner == 0 and req >= mid:
            return max(min(ally_adjacent, mid - 1), 0)
        if owner == 1 and req <= mid:
            return min(max(ally_adjacent, mid + 1), self.lane_max_pos)

        # If request within allowed set, honor it; else clamp to nearest allowed to start
        if req in allowed:
            return req
        return min(allowed, key=lambda p: abs(p - default_pos))

    def _parse_placements(self, action: Any) -> List[Any]:
        """Normalize action into a list of placements preserving optional pos.

        Accepted forms:
        - int: decodes to (card_id, lane_id)
        - list/tuple of ints
        - list/tuple of pairs (card_id, lane_id)
        - list/tuple of triples (card_id, lane_id, pos)
        - list/tuple of dicts {card,lane[,pos]} or {card_id,lane_id[,pos]}
        Returns empty list for pass/no-op. Pos is kept if provided.
        """
        placements: List[Any] = []
        max_single = self.num_cards * 2
        def decode_single(v: int) -> Tuple[int, int] | None:
            if 0 <= v < max_single:
                return (v // 2, v % 2)
            return None

        if isinstance(action, int):
            dec = decode_single(action)
            if dec is not None:
                placements.append(dec)
            return placements

        if isinstance(action, (list, tuple)):
            for item in action:
                if isinstance(item, int):
                    dec = decode_single(item)
                    if dec is not None:
                        placements.append(dec)
                elif isinstance(item, (list, tuple)) and len(item) in (2, 3):
                    try:
                        if len(item) == 2:
                            placements.append((int(item[0]), int(item[1])))
                        else:
                            placements.append((int(item[0]), int(item[1]), int(item[2])))
                    except Exception:
                        continue
                elif isinstance(item, dict):
                    if "card_id" in item and "lane_id" in item:
                        if "pos" in item:
                            placements.append((int(item["card_id"]), int(item["lane_id"]), int(item["pos"])) )
                        else:
                            placements.append((int(item["card_id"]), int(item["lane_id"])) )
                    elif "card" in item and "lane" in item:
                        if "pos" in item:
                            placements.append((int(item["card"]), int(item["lane"]), int(item["pos"])) )
                        else:
                            placements.append((int(item["card"]), int(item["lane"])) )
            return placements

        # Unknown type: treat as pass
        return placements

    def _spawn_unit(
        self,
        card_id: int,
        lane_id: int,
        owner: int,
        pos: Optional[int] = None,
        pay_cost: bool = False,
        events: Optional[List[str]] = None,
    ) -> None:
        card = self.cards[card_id]
        if pay_cost:
            pool = 0 if owner == 0 else 1
            if self.elixir[pool] < card["cost"]:
                if events is not None:
                    events.append(f"Owner {owner} insufficient elixir for {card['name']}")
                return
            self.elixir[pool] -= card["cost"]
        unit = {
            "id": self._next_unit_id,
            "name": card["name"],
            "hp": card["hp"],
            "max_hp": int(card["hp"]),
            "dmg": card["dmg"],
            "pos": (self.player_start if owner == 0 else self.enemy_start) if pos is None else int(pos),
            "owner": owner,
            "behavior": card.get("behavior", "nearest"),
            "range": int(card.get("range", 1)),
            "is_air": bool(card.get("is_air", False)),
            "targets": set(card.get("targets", ["ground"]))
        }
        unit["spawn_turn"] = self.turn
        self._next_unit_id += 1
        self.lanes[lane_id].append(unit)
        if events is not None:
            events.append(f"Spawned Unit#{unit['id']} {unit['name']} at lane {lane_id} pos {unit['pos']} (owner {owner})")

    def spawn_by_name(self, name: str, lane_id: int, owner: int, pos: Optional[int] = None) -> None:
        """Utility for deterministic scenarios in tests/debug. Ignores elixir."""
        # Build mapping lazily if needed
        try:
            idx = self.card_name_to_index  # type: ignore[attr-defined]
        except AttributeError:
            self.card_name_to_index = {c["name"]: i for i, c in enumerate(self.cards)}
        if name not in self.card_name_to_index:
            raise KeyError(f"Unknown card name: {name}")
        self._spawn_unit(self.card_name_to_index[name], lane_id, owner, pos=pos, pay_cost=False, events=None)

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    def _load_full_config(self, config_path: Optional[str]) -> None:
        # Unified config: config/game.json
        if config_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, "config", "game.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            return

        game = cfg.get("game", {})
        self.max_turns = int(game.get("max_turns", self.max_turns))
        self.max_elixir = int(game.get("max_elixir", self.max_elixir))
        self.starting_elixir = int(game.get("starting_elixir", self.starting_elixir))
        base_hp = game.get("base_hp", self.base_hp)
        if isinstance(base_hp, list) and len(base_hp) == 2:
            self.base_hp_init = [int(base_hp[0]), int(base_hp[1])]
        self.tower_hp_init = int(game.get("tower_hp", self.tower_hp_init))
        self.tower_dmg = int(game.get("tower_dmg", self.tower_dmg))
        # Geometry config (preferred over legacy lane_length)
        self.side_tiles = int(game.get("side_tiles", self.side_tiles))
        self.bridge_tiles = int(game.get("bridge_tiles", self.bridge_tiles))
        self.tower_start = int(game.get("tower_start", self.tower_start))
        self.tower_width = int(game.get("tower_width", self.tower_width))
        lane_length = self.side_tiles * 2 + self.bridge_tiles
        self.lane_max_pos = max(3, lane_length - 1)
        # Left (player) tower occupies [tower_start .. tower_start + tower_width - 1]
        self.tower_player_start = self.tower_start
        self.tower_player_end = self.tower_start + self.tower_width - 1
        # Right (enemy) tower symmetrical from the right edge
        self.tower_enemy_end = self.lane_max_pos - self.tower_start
        self.tower_enemy_start = self.tower_enemy_end - (self.tower_width - 1)
        # Spawn positions at start of each side (beginning tiles)
        self.player_start = 0
        self.enemy_start = self.lane_max_pos
        # Keep tower center indices for rendering/reference
        self.tower_player_pos = self.tower_player_start + (self.tower_width // 2)
        self.tower_enemy_pos = self.tower_enemy_start + (self.tower_width // 2)
        # Adjacent tiles toward bridge where units stop and attack towers
        self.player_adjacent = self.tower_player_end + 1
        self.enemy_adjacent = self.tower_enemy_start - 1

        ai = cfg.get("ai", {})
        enemy_ai = bool(ai.get("enemy_ai", not self.disable_enemy_ai))
        self.disable_enemy_ai = not enemy_ai

        # Deploy zones (optional) from config; default to owner's side only
        deploy = cfg.get("deploy", {})
        self.deploy_player_tiles = None
        self.deploy_enemy_tiles = None
        if isinstance(deploy, dict):
            # Player
            if "player_tiles" in deploy and isinstance(deploy["player_tiles"], list):
                self.deploy_player_tiles = set(int(x) for x in deploy["player_tiles"]) or None
            elif "player_from_start" in deploy:
                k = int(deploy["player_from_start"])  # inclusive index
                self.deploy_player_tiles = set(range(0, min(self.lane_max_pos, k) + 1))
            # Enemy
            if "enemy_tiles" in deploy and isinstance(deploy["enemy_tiles"], list):
                self.deploy_enemy_tiles = set(int(x) for x in deploy["enemy_tiles"]) or None
            elif "enemy_from_end" in deploy:
                k = int(deploy["enemy_from_end"])  # inclusive from end
                start = max(0, self.lane_max_pos - k)
                self.deploy_enemy_tiles = set(range(start, self.lane_max_pos + 1))

        # Defaults if not provided: restrict to own side
        mid = int(self.lane_max_pos // 2)
        if not self.deploy_player_tiles:
            self.deploy_player_tiles = set(range(0, mid))
        if not self.deploy_enemy_tiles:
            self.deploy_enemy_tiles = set(range(mid + 1, self.lane_max_pos + 1))
    def _load_cards(self, cards: Optional[Any], cards_path: Optional[str]) -> List[Dict[str, Any]]:
        """Load card definitions from provided object or the unified config.

        Accepted shapes if provided directly:
        - list of card dicts: [{"name", "cost", "hp", "dmg"}, ...]
        - dict of id->card dict: {"0": {...}, "1": {...}}
        Otherwise, reads from `config/game.json` under key `cards`.
        Returns a list indexed by card_id for stable ordering.
        """
        if cards is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            game_cfg_path = os.path.join(base_dir, "config", "game.json")
            with open(game_cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if not (isinstance(cfg, dict) and "cards" in cfg):
                raise FileNotFoundError("Cards not found: include `cards` in config/game.json or pass `cards` param")
            cards = cfg["cards"]

        if isinstance(cards, list):
            lst = [self._validate_card_obj(obj, i) for i, obj in enumerate(cards)]
        elif isinstance(cards, dict):
            items: List[Tuple[int, Dict[str, Any]]] = []
            for k, v in cards.items():
                try:
                    idx = int(k)
                except Exception:
                    raise ValueError(f"Card id must be int-like, got key {k!r}")
                items.append((idx, self._validate_card_obj(v, idx)))
            items.sort(key=lambda x: x[0])
            lst = [v for _, v in items]
        else:
            raise TypeError("cards must be list or dict or JSON content thereof")

        if not lst:
            raise ValueError("Card configuration is empty")
        return lst

    def _validate_card_obj(self, obj: Dict[str, Any], idx: int) -> Dict[str, Any]:
        for key in ("name", "cost", "hp", "dmg"):
            if key not in obj:
                raise ValueError(f"Card {idx} missing field: {key}")
        # Optional fields and normalization
        behavior = str(obj.get("behavior", "nearest"))
        rng = int(obj.get("range", 1))
        is_air = bool(obj.get("is_air", False))
        targets_raw = obj.get("targets", ["ground"])  # default ground-only
        if not isinstance(targets_raw, (list, tuple)):
            raise TypeError(f"Card {idx} field targets must be list[str], got {type(targets_raw).__name__}")
        targets = []
        for t in targets_raw:
            ts = str(t).lower()
            if ts not in ("ground", "air"):
                raise ValueError(f"Card {idx} targets entries must be 'ground' or 'air', got {t!r}")
            targets.append(ts)
        return {
            "name": str(obj["name"]),
            "cost": int(obj["cost"]),
            "hp": int(obj["hp"]),
            "dmg": int(obj["dmg"]),
            "behavior": behavior,
            "range": rng,
            "is_air": is_air,
            "targets": targets,
        }


def make_puffer_env() -> ClashRoyaleEnv:
    """Factory usable by PufferLib trainers: returns a fresh env instance."""
    return ClashRoyaleEnv()


if __name__ == "__main__":
    env = ClashRoyaleEnv()
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    print("Episode finished with reward", total_reward)
