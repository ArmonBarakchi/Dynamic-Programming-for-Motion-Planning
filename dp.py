import copy
from collections import deque

from utils import MF, TL, TR, PK, UD, step

ACTIONS = [MF, TL, TR, PK, UD]


def _door_positions(info):
    dp = info["door_pos"]
    if isinstance(dp, list):
        return [tuple(map(int, p)) for p in dp]
    return [tuple(map(int, dp))]


def _door_open_tuple(env, info):
    door_pos_list = _door_positions(info)
    opens = []
    for (x, y) in door_pos_list:
        door = env.unwrapped.grid.get(x, y)
        opens.append(bool(door.is_open) if door is not None else True)
    return tuple(opens)

#state formulation from report for known maps
def _state_key(env, info):
    agent_pos = tuple(map(int, env.unwrapped.agent_pos))
    agent_dir = int(env.unwrapped.agent_dir)
    has_key   = env.unwrapped.carrying is not None
    door_open = _door_open_tuple(env, info)
    return (agent_pos, agent_dir, has_key, door_open)


def _is_goal(env, info):
    goal_pos  = tuple(map(int, info["goal_pos"]))
    agent_pos = tuple(map(int, env.unwrapped.agent_pos))
    return agent_pos == goal_pos


def _simulate(env, info, action):
    env_next = copy.deepcopy(env)
    c, done = step(env_next, action)
    s_next = _state_key(env_next, info)
    return env_next, s_next, c, done

#run BFS on the state space to create MDP graph
def _enumerate_reachable_states(env, info):
    s0 = _state_key(env, info)
    q = deque([s0])
    visited = {s0}
    state_to_env = {s0: copy.deepcopy(env)}
    succ = {}

    while q:
        s = q.popleft()
        env_s = state_to_env[s]
        succ[s] = {}
        for a in ACTIONS:
            env_next, s_next, c, _ = _simulate(env_s, info, a)
            succ[s][a] = (s_next, c)
            if s_next not in visited:
                visited.add(s_next)
                state_to_env[s_next] = env_next
                q.append(s_next)

    return list(visited), state_to_env, succ


def dp(env, info):

    states, state_to_env, succ = _enumerate_reachable_states(env, info)

    INF   = float("inf")
    THETA = 1e-8

    V = {}
    for s in states:
        V[s] = 0.0 if _is_goal(state_to_env[s], info) else INF

    policy = {s: None for s in states}

    # Sweep backwards in time until convergence
    while True:
        delta = 0.0
        V_new = {}

        for s in states:
            if _is_goal(state_to_env[s], info):
                V_new[s] = 0.0
                policy[s] = None
                continue

            best_Q   = INF
            best_act = None

            for a in ACTIONS:
                s_next, c = succ[s][a]
                Q = c + V[s_next]
                if Q < best_Q:
                    best_Q   = Q
                    best_act = a

            # V_t(x) = min_u Q_t(x,u)
            V_new[s]  = best_Q
            # pi_t(x) = argmin_u Q_t(x,u)
            policy[s] = best_act
            delta = max(delta, abs(V_new[s] - V[s]))

        # V_{t+1} <- V_t
        V = V_new

        if delta < THETA:
            break

    # policy execution
    s = _state_key(env, info)
    env_roll = copy.deepcopy(env)
    optim_act_seq = []

    while not _is_goal(env_roll, info):
        a = policy[s]
        if a is None:
            break
        optim_act_seq.append(a)
        env_roll, s, _, _ = _simulate(env_roll, info, a)

    return optim_act_seq