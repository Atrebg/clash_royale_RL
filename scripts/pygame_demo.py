#!/usr/bin/env python3
"""
Pygame demo renderer for ClashRoyaleEnv.

Controls
- 1..9: seleziona carta (id)
- Q/W: seleziona corsia (0/1)
- SPACE: schiera la carta selezionata nella corsia selezionata
- E: abilita/disabilita IA nemica
- P: pausa/continua il tempo
- R: reset partita
- ESC/Close: esci

Nota: devi avere pygame installato: `python3 -m pip install pygame`.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pygame
except Exception as e:  # pragma: no cover
    print("Error: pygame not installed. Install with: python3 -m pip install pygame")
    raise

from clash_royale_env import ClashRoyaleEnv


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class PygameRenderer:
    def __init__(self, width: int = 900, height: int = 500) -> None:
        pygame.init()
        pygame.display.set_caption("ClashRoyaleEnv - Pygame Demo")
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.width = width
        self.height = height

        # Layout
        self.margin = 40
        self.lane_gap = 120
        self.lane_y0 = height // 2 - self.lane_gap // 2
        self.lane_y1 = height // 2 + self.lane_gap // 2
        self.left_x = self.margin
        self.right_x = width - self.margin

        self.font = pygame.font.SysFont("Arial", 16)
        self.font_small = pygame.font.SysFont("Arial", 12)

    def draw_background(self) -> None:
        self.screen.fill((30, 30, 40))
        # Lanes baseline
        pygame.draw.line(self.screen, (60, 120, 60), (self.left_x, self.lane_y0), (self.right_x, self.lane_y0), 2)
        pygame.draw.line(self.screen, (60, 120, 60), (self.left_x, self.lane_y1), (self.right_x, self.lane_y1), 2)
        # Midfield marker (tower range limit)
        half_x = int((self.left_x + self.right_x) / 2)
        pygame.draw.line(self.screen, (200, 200, 80), (half_x, self.lane_y0 - 20), (half_x, self.lane_y0 + 20), 2)
        pygame.draw.line(self.screen, (200, 200, 80), (half_x, self.lane_y1 - 20), (half_x, self.lane_y1 + 20), 2)

    def draw_lane_highlight(self, lane_id: int) -> None:
        # Semi-transparent highlight over selected lane
        lane_y = self.lane_y0 if lane_id == 0 else self.lane_y1
        h = 34
        surf = pygame.Surface((self.width - 2 * self.margin, h), pygame.SRCALPHA)
        surf.fill((200, 200, 80, 60))  # RGBA with alpha
        self.screen.blit(surf, (self.margin, lane_y - h // 2))

    def draw_towers(self, env: ClashRoyaleEnv) -> None:
        # Player tower (left), Enemy tower (right)
        lane_max = getattr(env, "lane_max_pos", 5)
        tower_width_tiles = getattr(env, "tower_width", 3)
        # Tower width spans tower_width_tiles
        tile_px = max(8, int((self.right_x - self.left_x) / float(max(1, lane_max))))
        tower_w, tower_h = tile_px * tower_width_tiles, 60
        # Helper to map pos -> x
        def pos_to_x(pos: int) -> int:
            t = float(np.clip(pos, 0, lane_max)) / float(max(1, lane_max))
            return int(lerp(self.left_x, self.right_x, t))
        def pos_to_x_float(pos: float) -> int:
            t = max(0.0, min(float(pos), float(lane_max))) / float(max(1, lane_max))
            return int(lerp(self.left_x, self.right_x, t))
        # Lane rectangles for towers spanning the full tile width
        for i, y in enumerate([self.lane_y0, self.lane_y1]):
            # left edge from tower start
            left_px_p = pos_to_x_float(getattr(env, "tower_player_start", 4) - 0.5)
            rect_p = pygame.Rect(left_px_p, y - tower_h // 2, tower_w, tower_h)
            pygame.draw.rect(self.screen, (70, 130, 180), rect_p)
            left_px_e = pos_to_x_float(getattr(env, "tower_enemy_start", lane_max - 4) - 0.5)
            rect_e = pygame.Rect(left_px_e, y - tower_h // 2, tower_w, tower_h)
            pygame.draw.rect(self.screen, (180, 70, 70), rect_e)
            # Tower HP labels above
            thp = getattr(env, "tower_hp", [[0, 0], [0, 0]])
            self._blit_text(str(max(0, thp[0][i])), (rect_p.centerx - 8, rect_p.top - 18), (180, 220, 255), small=True)
            self._blit_text(str(max(0, thp[1][i])), (rect_e.centerx - 8, rect_e.top - 18), (255, 180, 180), small=True)

    def draw_bridge(self, env: ClashRoyaleEnv) -> None:
        # Draw a wood-like bridge on the central tile of each lane
        lane_max = getattr(env, "lane_max_pos", 5)
        mid = int(lane_max // 2)
        def x_from_tile_edge(edge_pos: float) -> int:
            t = max(0.0, min(edge_pos, float(lane_max))) / float(max(1, lane_max))
            return int(lerp(self.left_x, self.right_x, t))
        # Tile edges
        left_edge = x_from_tile_edge(mid - 0.5)
        right_edge = x_from_tile_edge(mid + 0.5)
        width = max(6, right_edge - left_edge)
        height = 26
        color = (110, 85, 50)
        plank_color = (140, 110, 70)
        border = (80, 60, 35)
        for y in (self.lane_y0, self.lane_y1):
            rect = pygame.Rect(left_edge, y - height // 2, width, height)
            pygame.draw.rect(self.screen, color, rect)
            # planks
            planks = 4
            for i in range(1, planks):
                px = left_edge + i * width // planks
                pygame.draw.line(self.screen, plank_color, (px, rect.top + 2), (px, rect.bottom - 2), 2)
            pygame.draw.rect(self.screen, border, rect, 2)

    def draw_lane_ticks(self, env: ClashRoyaleEnv) -> None:
        # Draw ticks: small at centers (tile indices), big at borders (half-steps)
        lane_max = getattr(env, "lane_max_pos", 5)
        def pos_to_x_float(pos: float) -> int:
            t = max(0.0, min(float(pos), float(lane_max))) / float(max(1, lane_max))
            return int(lerp(self.left_x, self.right_x, t))

        color_center = (110, 110, 130)
        color_border = (160, 160, 190)
        h_small = 6
        h_big = 14
        for lane_y in (self.lane_y0, self.lane_y1):
            # centers (integers) - small
            for p in range(lane_max + 1):
                x = pos_to_x_float(float(p))
                pygame.draw.line(self.screen, color_center, (x, lane_y - h_small // 2), (x, lane_y + h_small // 2), 1)
            # borders (half-steps) - big
            for b in range(lane_max):
                x = pos_to_x_float(b + 0.5)
                pygame.draw.line(self.screen, color_border, (x, lane_y - h_big // 2), (x, lane_y + h_big // 2), 2)

        # HUD: show Turn and Lane length only (tower HP shown above towers)
        lane_max = getattr(env, "lane_max_pos", 5)
        lane_len = lane_max + 1
        txt = f"Turn: {env.turn}   Lane: {lane_len} (0..{lane_max})"
        self._blit_text(txt, (self.margin, 10), (220, 220, 220))

    def draw_units(self, env: ClashRoyaleEnv) -> None:
        # Convert pos [0..lane_max] to x
        lane_max = getattr(env, "lane_max_pos", 5)
        def pos_to_x(pos: int) -> int:
            t = float(np.clip(pos, 0, lane_max)) / float(max(1, lane_max))
            return int(lerp(self.left_x, self.right_x, t))

        for lane_id, y in enumerate([self.lane_y0, self.lane_y1]):
            for u in env.lanes[lane_id]:
                x = pos_to_x(u.get("pos", 0))
                is_enemy = (u.get("owner", 0) == 1)
                is_air = bool(u.get("is_air", False))
                color = (70, 160, 255) if not is_enemy else (255, 100, 100)
                if is_air:
                    color = (120, 200, 255) if not is_enemy else (255, 150, 150)

                # Draw unit (circle for air, rectangle for ground)
                if is_air:
                    pygame.draw.circle(self.screen, color, (x, y), 10)
                else:
                    pygame.draw.rect(self.screen, color, pygame.Rect(x - 10, y - 10, 20, 20))

                # HP bar and numbers
                max_hp = max(1, int(u.get("max_hp", u.get("hp", 1))))
                hp = max(0, int(u.get("hp", 0)))
                hp_w = 24
                ratio = max(0.0, min(1.0, hp / float(max_hp)))
                bar_bg = pygame.Rect(x - hp_w // 2, y - 18, hp_w, 4)
                pygame.draw.rect(self.screen, (50, 50, 50), bar_bg)
                pygame.draw.rect(self.screen, (80, 220, 80), (bar_bg.x, bar_bg.y, int(hp_w * ratio), 4))
                # HP text
                self._blit_text(f"{hp}/{max_hp}", (x - 16, y - 34), (200, 230, 200), small=True)

                # Name/id label
                name = u.get("name") or "?"
                uid = u.get("id", "?")
                self._blit_text(f"{name}#{uid}", (x - 20, y + 12), (230, 230, 230), small=True)

    def draw_hud(self, env: ClashRoyaleEnv, selected_card: Optional[int], selected_lane: int) -> None:
        # Elixir
        self._blit_text(f"Elixir P:{env.elixir[0]}  E:{env.elixir[1]}  AI:{'ON' if not env.disable_enemy_ai else 'OFF'}",
                        (self.margin, self.height - 28), (220, 220, 180))
        # Cards list
        x = self.margin
        y = self.height - 50
        for i, c in enumerate(env.cards):
            s = f"[{i+1}] {c['name']} (cost {c['cost']})"
            color = (255, 255, 255) if selected_card != i else (255, 220, 120)
            self._blit_text(s, (x, y), color)
            x += 200
        # Lane selection
        self._blit_text(f"Lane: {selected_lane} (Up/Down)", (self.width - 220, self.height - 28), (200, 200, 255))

    def draw_events(self, info_events: Tuple[str, ...]) -> None:
        # Show last ~6 events
        max_lines = 6
        lines = info_events[-max_lines:]
        y = 30
        for line in lines:
            self._blit_text(line, (self.margin, y), (200, 200, 200))
            y += 18

    def _blit_text(self, text: str, pos, color, small: bool = False) -> None:
        surf = (self.font_small if small else self.font).render(text, True, color)
        self.screen.blit(surf, pos)

    def tick(self, fps: int) -> None:
        self.clock.tick(fps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=None, help="override render.width")
    parser.add_argument("--height", type=int, default=None, help="override render.height")
    parser.add_argument("--fps", type=int, default=None, help="override render.fps")
    parser.add_argument("--step_ms", type=int, default=None, help="ms per step (override config)")
    args = parser.parse_args()

    # Load unified config for render defaults
    cfg_path = REPO_ROOT / "config" / "game.json"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    render_cfg = cfg.get("render", {})

    width = int(args.width if args.width is not None else render_cfg.get("width", 900))
    height = int(args.height if args.height is not None else render_cfg.get("height", 500))
    fps = int(args.fps if args.fps is not None else render_cfg.get("fps", 60))
    manual_step = bool(render_cfg.get("manual_step", False))

    env = ClashRoyaleEnv(config_path=str(cfg_path))
    # Apply AI default from config (env._load_full_config handles it); allow toggling in UI
    obs, info = env.reset(seed=0)

    rnd = PygameRenderer(width, height)
    running = True
    paused = False
    selected_card: Optional[int] = 0 if env.num_cards > 0 else None
    selected_lane = 0
    queued_actions: 'List[Tuple[int, int, int]]' = []  # list of (card_id, lane, pos)
    # Player deploy tiles from config (optional)
    deploy_tiles = getattr(env, 'deploy_player_tiles', None)
    selected_tile_idx = 0  # Ensure selected_tile_idx is always initialized
    if deploy_tiles:
        allowed_sorted = sorted(deploy_tiles)
        selected_tile = allowed_sorted[selected_tile_idx]
    else:
        selected_tile = getattr(env, 'player_start', 0)

    # timing for step
    # Load speed from config unless overridden
    default_step_ms = 1000
    if args.step_ms is not None:
        step_ms = max(1, int(args.step_ms))
    else:
        step_ms = int(render_cfg.get("step_ms", default_step_ms))

    accum_ms = 0
    step_requested = False

    # last events
    last_events: Tuple[str, ...] = tuple()

    while running:
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r:
                    obs, info = env.reset(seed=0)
                    last_events = tuple()
                elif event.key == pygame.K_e:
                    env.disable_enemy_ai = not env.disable_enemy_ai
                elif event.key in (pygame.K_q, pygame.K_w):
                    selected_lane = 0 if event.key == pygame.K_q else 1
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    selected_lane = 0 if event.key == pygame.K_UP else 1
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    if deploy_tiles:
                        allowed_sorted = sorted(deploy_tiles)
                        if event.key == pygame.K_LEFT:
                            selected_tile_idx = (selected_tile_idx - 1) % len(allowed_sorted)
                        else:
                            selected_tile_idx = (selected_tile_idx + 1) % len(allowed_sorted)
                        selected_tile = allowed_sorted[selected_tile_idx]
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    idx = event.key - pygame.K_1
                    if 0 <= idx < env.num_cards:
                        selected_card = idx
                elif event.key == pygame.K_SPACE:
                    # Queue a deploy action; multiple allowed per step
                    if selected_card is not None and 0 <= selected_lane <= 1:
                        queued_actions.append((int(selected_card), int(selected_lane), int(selected_tile)))
                elif event.key in (pygame.K_RETURN, pygame.K_n):
                    step_requested = True

        # step timing
        dt = rnd.clock.get_time()
        accum_ms += dt

        action = env.num_cards * 2  # pass by default
        should_step = False
        if not paused:
            if manual_step:
                if step_requested:
                    should_step = True
                    step_requested = False
            else:
                if accum_ms >= step_ms:
                    should_step = True
                    accum_ms = 0

        if should_step:
            # perform all queued actions once, otherwise pass
            if queued_actions:
                action = list(queued_actions)
                queued_actions.clear()
            obs, reward, terminated, truncated, info = env.step(action)
            last_events = tuple(info.get("events", []))
            if terminated or truncated:
                paused = True

        # draw
        rnd.draw_background()
        # Highlight selected lane before drawing units
        rnd.draw_lane_highlight(selected_lane)
        rnd.draw_bridge(env)
        rnd.draw_lane_ticks(env)
        rnd.draw_towers(env)
        rnd.draw_units(env)
        rnd.draw_hud(env, selected_card, selected_lane)
        # Show queued actions and selected tile
        if queued_actions:
            labels = []
            for c,l,p in queued_actions:
                name = env.cards[c]["name"] if 0 <= c < len(env.cards) else str(c)
                labels.append(f"{name}@{l}:{p}")
            rnd._blit_text("Queued: " + ", ".join(labels), (rnd.margin, rnd.height - 70), (240, 220, 150))
        rnd._blit_text(f"Tile: {selected_tile}", (rnd.width - 120, rnd.height - 48), (200, 200, 255))
        rnd.draw_events(last_events)
        # Manual mode indicator
        mode_txt = f"Mode: {'MANUAL' if manual_step else 'AUTO'} (Enter/N to step)"
        rnd._blit_text(mode_txt, (rnd.width - 280, 10), (200, 200, 120))

        pygame.display.flip()
        rnd.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()
