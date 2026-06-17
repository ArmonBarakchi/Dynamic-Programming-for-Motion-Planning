import copy
import pickle
from collections import deque

from utils import MF, TL, TR, PK, UD, step, step_cost
from dp import  _door_open_tuple, _is_goal
ACTIONS = [MF, TL, TR, PK, UD]

#state formulation from report for unknown maps

def _state_key(env, info):
    agent_pos = tuple(map(int, env.unwrapped.agent_pos))
    agent_dir = int(env.unwrapped.agent_dir)
    has_key   = env.unwrapped.carrying is not None
    door_open = _door_open_tuple(env, info)
    key_pos   = tuple(map(int, info["key_pos"]))
    goal_pos  = tuple(map(int, info["goal_pos"]))
    return (agent_pos, agent_dir, has_key, door_open, key_pos, goal_pos)

'''different simulation function to account for the fact that the unknown maps don't have a 
perimeter in the environment'''

def _simulate(env, info, action):
    fwd_pos = env.unwrapped.front_pos
    width   = env.unwrapped.width
    height  = env.unwrapped.height
    x, y    = int(fwd_pos[0]), int(fwd_pos[1])
    out_of_bounds = not (0 <= x < width and 0 <= y < height)

    if out_of_bounds and action in [MF, UD]:
        return copy.deepcopy(env), _state_key(env, info), step_cost(action), False

    env_next = copy.deepcopy(env)
    try:
        c, done = step(env_next, action)
        s_next  = _state_key(env_next, info)
    except AssertionError:
        c        = step_cost(action)
        done     = False
        s_next   = _state_key(env, info)
        env_next = copy.deepcopy(env)

    return env_next, s_next, c, done


def _enumerate_reachable_states(envs, infos):

    visited = {}
    succ    = {}
    queue   = deque()

    for env, info in zip(envs, infos):
        s0 = _state_key(env, info)
        if s0 not in visited:
            visited[s0] = (copy.deepcopy(env), info)
            queue.append(s0)

    while queue:
        s = queue.popleft()
        env_s, info_s = visited[s]
        succ[s] = {}

        for a in ACTIONS:
            env_next, s_next, c, _ = _simulate(env_s, info_s, a)
            succ[s][a] = (s_next, c)

            if s_next not in visited:
                visited[s_next] = (env_next, info_s)
                queue.append(s_next)

    return visited, succ


def dp_random(envs, infos):

    visited, succ = _enumerate_reachable_states(envs, infos)
    states = list(visited.keys())

    INF   = float("inf")
    THETA = 1e-8

    V = {}
    for s in states:
        env_s, info_s = visited[s]
        V[s] = 0.0 if _is_goal(env_s, info_s) else INF

    policy    = {s: None for s in states}
    while True:
        delta = 0.0
        V_new = {}

        for s in states:
            env_s, info_s = visited[s]

            if _is_goal(env_s, info_s):
                V_new[s]  = 0.0
                policy[s] = None
                continue

            best_Q   = INF
            best_act = None

            for a in ACTIONS:
                s_next, c = succ[s][a]
                # Q_t(x,u) = l(x,u) + V_{t+1}(f(x,u))
                Q = c + V[s_next]
                if Q < best_Q:
                    best_Q   = Q
                    best_act = a

            # V_t(x) = min_u Q_t(x,u)
            V_new[s]  = best_Q
            # pi_t(x) = argmin_u Q_t(x,u)
            policy[s] = best_act
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new

        if delta < THETA:
            break

    return policy


def run_random_policy(env, info, policy):
    env_roll = copy.deepcopy(env)
    s = _state_key(env_roll, info)
    optim_act_seq = []

    while not _is_goal(env_roll, info):
        a = policy[s]
        if a is None:
            break
        optim_act_seq.append(a)
        env_roll, s, _, _ = _simulate(env_roll, info, a)

    return optim_act_seq


def save_policy(policy, path="./policy/random_policy.pkl"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(policy, f)



def load_policy(path="./policy/random_policy.pkl"):
    with open(path, "rb") as f:
        policy = pickle.load(f)
    return policy