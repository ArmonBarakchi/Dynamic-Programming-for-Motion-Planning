# DoorKey DP Planner

This repository provides an optimal planning method for the MiniGrid DoorKey environment 
using Dynamic Programming. The state space is enumerated using a BFS, then the optimal path is found using
dynamic programming, and then it is executed.


---

## Project Structure

```
.
├── doorkey.py          # Entry point: partA and partB
├── dp.py               # Known-map DP
├── DP_random.py        # Unknown-map DP
├── utils.py            # Env loading, step costs, GIF generation
├── create_env.py       # Generates known and random .env files
├── gif_to_png.py       # Converts GIFs to PNG contact sheets
├── envs/
│   ├── known_envs/     # 7 fixed maps 
│   └── random_envs/    # 36 random 10x10 maps 
├── gif/                # Output GIFs 
├── output/             # Output PNG gif strips sheets 
└── policy/
    └── random_policy.pkl  # Cached policy for Part B 
```

---

## Setup

### Conda 

```bash
conda env create -f environment.yaml
conda activate doorkey-dp
```

###  pip

```bash
pip install -r requirements.txt
```


---

## Running

Both parts are run from `doorkey.py`. Uncomment the desired part at the bottom of the file:

```python
if __name__ == "__main__":
    # partA()   # Known maps
    partB()     # Random maps
```

Then run:

```bash
python doorkey.py
```

### Part A — Known Maps

Runs the DP planner on all 7 known fixed maps. For each map it computes the optimal action sequence, saves a GIF to `gif/test.gif`, and saves a PNG gif strip sheet to `output/<map_name>.png`.



### Part B — Random Maps

Builds a unified state graph over all 36 random 10x10 environments and computes a single policy that works across all of them. The policy is cached to `policy/random_policy.pkl` — subsequent runs load it from disk instead of recomputing.

To force recomputation, delete the cached file:

```bash
rm policy/random_policy.pkl
```



## Output

Each run produces:
- `gif/test.gif` — animated trajectory
- `output/<map_name>.png` — contact sheet of all frames
