# clash_royale_RL

Semplified text-based environment inspired by Clash Royale for training RL agents.

## Usage

The environment is implemented in `clash_royale_env.py`. It requires the
[`gym`](https://www.gymlibrary.dev/) library and `numpy`.

```bash
pip install gym numpy
python clash_royale_env.py  # runs a random-agent demo
```

The environment exposes a standard Gym interface so it can be used with
libraries such as Stable Baselines3 or RLlib.
