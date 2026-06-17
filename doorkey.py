from utils import *
from gymnasium.envs.registration import register
from minigrid.envs.doorkey import DoorKeyEnv
from dp import dp
from DP_random import dp_random, run_random_policy, save_policy, load_policy
from gif_to_png import gif_to_full_grid

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

class DoorKey10x10Env(DoorKeyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=10, **kwargs)

register(
    id='MiniGrid-DoorKey-10x10-v0',
    entry_point='__main__:DoorKey10x10Env'
)

def doorkey_problem(env,info):
    """
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Template:
        Replace the placeholder list with the action sequence returned by your
        planner. Minimize the same total stage cost as in utils.step_cost (and
        as defined in your report's MDP). You may branch on env / loaded map if
        needed for Part (A); Part (B) should respect the single-policy requirement.
    """
    # STUDENT: placeholder sequence for wiring; not a solution for all maps.
    optim_act_seq = dp(env,info)
    return optim_act_seq

env_paths = ["./envs/known_envs/doorkey-5x5-normal.env",
             "./envs/known_envs/doorkey-6x6-direct.env",
             "./envs/known_envs/doorkey-6x6-normal.env",
             "./envs/known_envs/doorkey-6x6-shortcut.env",
             "./envs/known_envs/doorkey-8x8-direct.env",
             "./envs/known_envs/doorkey-8x8-normal.env",
             "./envs/known_envs/doorkey-8x8-shortcut.env",
             "./envs/known_envs/example-8x8.env"]
def partA():
    for env_path in env_paths:
        print(env_path)
        env, info = load_env(env_path)
        seq = doorkey_problem(env,info)
        draw_gif_from_seq(seq, load_env(env_path)[0], path="./gif/test.gif")

        #convert gif to gif strips for my report
        base_name = os.path.splitext(os.path.basename(env_path))[0]
        output_path = os.path.join("output", base_name + ".png")
        gif_to_full_grid(
            gif_path="./gif/test.gif",
            output_path=output_path,
            max_height=150,
            cols=5
        )

#loads all the random environments for policy creation
def load_all_random_envs(env_folder):

    env_paths = sorted([
        os.path.join(env_folder, f)
        for f in os.listdir(env_folder)
        if f.endswith(".env")
    ])

    envs, infos = [], []

    for env_path in env_paths:
        with open(env_path, "rb") as f:
            env = pickle.load(f)
        info = {
            "height": env.unwrapped.height,
            "width": env.unwrapped.width,
            "init_agent_pos": env.unwrapped.agent_pos,
            "init_agent_dir": env.unwrapped.dir_vec,
            "door_pos": [],
            "door_open": [],
        }

        for i in range(env.unwrapped.height):
            for j in range(env.unwrapped.width):
                cell = env.unwrapped.grid.get(j, i)
                if isinstance(cell, Key):
                    info["key_pos"] = np.array([j, i])
                elif isinstance(cell, Door):
                    info["door_pos"].append(np.array([j, i]))
                    info["door_open"].append(cell.is_open)
                elif isinstance(cell, Goal):
                    info["goal_pos"] = np.array([j, i])

        envs.append(env)
        infos.append(info)

    return envs, infos


def partB():
    env_folder = "./envs/random_envs"
    policy_path = "./policy/random_policy.pkl"

    # Load policy from disc if it exists otherwise compute and save it
    #To force recomputation delete the .pkl file in ./policy
    if os.path.exists(policy_path):
        policy = load_policy(policy_path)
    else:
        envs, infos = load_all_random_envs(env_folder)
        policy = dp_random(envs, infos)
        save_policy(policy, policy_path)


    env, info, env_path = load_random_env(env_folder)
    seq = run_random_policy(env, info, policy)
    draw_gif_from_seq(seq, load_env(env_path)[0], path="./gif/test.gif")

    # convert gif to gif strips for my submission
    base_name = os.path.splitext(os.path.basename(env_path))[0]
    output_path = os.path.join("output", base_name + ".png")
    gif_to_full_grid(
        gif_path="./gif/test.gif",
        output_path=output_path,
        max_height=150,
        cols=5
        )

if __name__ == "__main__":
   # example_use_of_gym_env()
   #partA()
   partB()

