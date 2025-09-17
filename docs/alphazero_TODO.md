AlphaZero TODOs for ClashRoyaleEnv

Goal: enable AlphaZero-style self-play training (MCTS + policy/value network) on this simplified two-lane game.

Engine and API
- Two-player API: expose `current_player`, `legal_actions(player)`, and a deterministic `step(p_action, e_action)`; or an alternation wrapper that produces simultaneous actions from two policies, then calls env once per tick.
- Perspective obs: `obs(player)` that mirrors lanes/targets so the game looks canonical from the acting player; include per-lane tower HP separately (done) and optionally richer lane features.
- Determinism: seedable RNG; no hidden randomness in step; make enemy policy explicit (no random enemy during self-play).
- State clone + hash: implement fast copy (`clone()`) and a stable hash (`key()`) for transposition/MCTS caching; avoid Python dict-heavy state for speed (numpy arrays or tuples).
- Terminal outcome: define `result` in {-1, 0, +1} from the perspective of `current_player` when a tower reaches 0 HP (win/lose) or max turns (draw).

Game model adjustments
- Simultaneity vs turn-based: choose one
  - A) Alternating policy picks: at tick t, collect both players’ actions via their MCTS/policy; call env once with both actions; swap `current_player`.
  - B) True alternating: split each tick into two sub-steps (player first, then enemy) so AlphaZero fits a turn-based interface.
- Legal actions: enumerate only playable cards (enough elixir) per lane + pass; ensure symmetry between players.
- Canonical features: consider 1D planes of length `lane_length` with channels per unit type/owner/HP buckets, tower HP, elixir, turn; keep current 1D vector as minimal baseline.

- Multiple deployments per turn (NEW)
  - Support k≥1 placements per player in the same tick (subject to elixir and per-lane constraints).
  - Action encoding options:
    - Multi-binary over [card×lane] with elixir-cost feasibility mask (preferred for simultaneous play).
    - Variable-length sequence within a turn: sample until elixir exhausted or PASS; collapse to a set for env step.
  - MCTS integration: treat a turn as a single decision over a combinatorial action; use bigram prior factorization or sequential rollout to reduce branching.
  - Env API: extend step to accept a list/set of placements per player; keep deterministic resolution order.
  - Legal mask: compute feasible subsets efficiently (prune by cost, max per-lane spawn, duplicates).

Self-play + MCTS
- MCTS (PUCT): implement node with N, W, Q, P; expand from network policy; select via `Q + c_puct * P * sqrt(N_parent)/(1+N)`; backprop value.
- Dirichlet noise on root prior (α per action count), temperature τ schedule (high early moves, 0 later), move sampling from visit counts.
- Replay: store (canonical_state, pi, z) tuples; z in {-1,0,1}; optionally outcome from the player-to-move perspective at root.
- Resignation (optional): resign when value estimate < threshold; guard with false-positive rate check.

Network + Training
- Framework: PyTorch (preferred). Define policy head (Categorical over legal actions) + value head (tanh scalar) on top of encoder for canonical features.
- Loss: (z − v)^2 − π·log(p) + L2; optimizer AdamW; cosine schedule; gradient clipping.
- Batching: sample from replay with recent priority (windowed buffer); normalize features if needed.
- Checkpointing: save best and latest; export for evaluation.

Evaluation + Gating
- Arena: pit latest vs best with fixed seeds; Elo or win-rate with margin; promote latest→best if threshold met.
- Sanity matches vs scripted bots: random, greedy-elixir, simple counters (e.g., play Musketeer vs air on same lane).

Vectorization + Orchestration
- Parallel self-play: use Gymnasium `VectorEnv` or PufferLib vectorization to generate episodes concurrently.
- Configs: YAML/JSON for MCTS (sims, c_puct, dirichlet, τ), network (layers, lr), self-play (envs, steps), evaluation (games, gating).
- Scripts: `scripts/az_selfplay.py`, `scripts/az_train.py`, `scripts/az_eval.py` with CLI args and logging.

Testing + Reproducibility
- Unit tests: determinism (same seeds → same traces), legal move correctness, symmetry (mirror of state/action is consistent), terminal outcomes.
- Golden traces: extend `trace_demo` with two-player traces; lock seeds.
- Logging: per-episode stats, MCTS diagnostics (visit dists, values), tensorboard support.

Performance
- Optimize env state representation (numpy arrays) and cloning; cache legal actions; avoid Python objects in tight loops.
- Optional: Cython/Numba for MCTS hot paths if needed.

Open Questions / Decisions
- Exact tick model (A or B above) for AlphaZero fit.
- Feature encoder richness vs speed.
- Action masking API for illegal actions (policy head with mask).

Deliverables (minimal viable path)
1) Canonical two-player wrapper + legal actions + deterministic step.
2) Simple MCTS (single-thread) + self-play data collection.
3) Tiny PyTorch net on current vector obs; end-to-end training loop.
4) Arena evaluation + checkpoint gating.
5) Scale up: vectorized self-play, richer features, larger net.

PufferLib Integration (focused TODOs)
- Env factory: expose `make_puffer_env(config_path=...)` returning a fresh Gymnasium-style env (done) and document required packages (`gymnasium`, `pufferlib`).
- Vectorized runners: add `scripts/puffer_vec_demo.py` using PufferLib vectorization (N envs) with seed control and action masking.
- Action masking API: expose `get_action_mask()` or include mask in `info` each step for legal cards/lanes; integrate with PufferLib policies.
- Disable enemy AI during training: read from `config.game.ai.enemy_ai=false` or a constructor flag; training loop controls both players’ actions.
- Observation dtype: ensure obs is `np.float32` (or convertible) and contiguous; provide normalization util if needed.
- Configs: add a `configs/` folder with presets for vector env count, rollout length, batch size, mixed precision, etc.

Env Performance TODOs
- Replace dict/list unit state with fixed-size numpy arrays per lane (e.g., struct-of-arrays: owner, pos, hp, max_hp, dmg, flags) to reduce Python overhead.
- Preallocate buffers for damage, removal indices, and temporary maps each step; avoid per-step allocations.
- Implement `clone()` returning a lightweight copy (views/copies of arrays) and `key()` returning a stable hash (e.g., `tobytes()` + simple hash) for MCTS TT.
- Compute legal moves once per step and cache; reuse for mask and MCTS expansion.
- Make tower/bridge constants scalar ints; avoid repeated `hasattr` checks; hoist invariants.
- Consider JIT (Numba) for hot loops (movement/combat) once arrays are in place; measure before/after.

AlphaZero/MCTS Scaling TODOs
- Parallel MCTS: CPU worker pool (process or thread) with a shared transposition table and virtual loss; batch network evaluations on GPU.
- Root dirichlet noise/temperature schedule configurable via CLI; log visit distributions and root values.
- Replay buffer: prioritized or windowed, sharded across workers; add on-disk spillover.
- Evaluation harness: arena with seeds × lanes; track Elo and promote checkpoints.
