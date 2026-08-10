'''
Adapted from OGBench
https://github.com/seohongpark/ogbench/blob/master/data_gen_scripts/generate_manipspace.py
'''
import pathlib
import os
import sys
import pickle

os.environ["MUJOCO_GL"] = "egl"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium
import numpy as np
from absl import app, flags
from tqdm import trange
import torch
from gymnasium.envs.registration import registry, register
import ogbench.manipspace  # noqa
from ogbench.manipspace.oracles.markov.button_markov import ButtonMarkovOracle
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle
from ogbench.manipspace.oracles.markov.drawer_markov import DrawerMarkovOracle
from ogbench.manipspace.oracles.markov.window_markov import WindowMarkovOracle
from ogbench.manipspace.oracles.plan.button_plan import ButtonPlanOracle
from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
from ogbench.manipspace.oracles.plan.drawer_plan import DrawerPlanOracle
from ogbench.manipspace.oracles.plan.window_plan import WindowPlanOracle
from envs.cube.cube_env import CubeEnv

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'cube-single-v1', 'Environment name.')
flags.DEFINE_string('dataset_type', 'play', 'Dataset type.')
flags.DEFINE_string('save_path', None, 'Directory to save outputs (will be created).')
flags.DEFINE_float('noise', 0.1, 'Action noise level.')
flags.DEFINE_float('noise_smoothing', 0.5, 'Action noise smoothing level for PlanOracle.')
flags.DEFINE_float('min_norm', 0.4, 'Minimum action norm for MarkovOracle.')
flags.DEFINE_float('p_random_action', 0, 'Probability of selecting a random action.')
flags.DEFINE_integer('num_episodes', 1000, 'Number of train episodes.')
flags.DEFINE_integer('max_episode_steps', None, 'Max steps per episode.')
flags.DEFINE_integer('init_cube', None, 'Force initial target cube index.')


def main(_):
    assert FLAGS.dataset_type in ['play', 'noisy']
    if FLAGS.save_path is None:
        raise ValueError("Please provide --save_path (directory).")

    save_folder = pathlib.Path(FLAGS.save_path)
    save_folder.mkdir(parents=True, exist_ok=True)
    obses_folder = save_folder / "obses"
    obses_folder.mkdir(parents=True, exist_ok=True)

    # Environment
    env_kwargs = {
        "terminate_at_goal": True,
        "mode": "task",
        "reward_task_id": 5,  # lock to task 5
    }

    if FLAGS.max_episode_steps:
        env_kwargs["max_episode_steps"] = FLAGS.max_episode_steps

    env = gymnasium.make(
        FLAGS.env_name,
        **env_kwargs
    )

    # Oracles initialization
    oracle_type = 'plan' if FLAGS.dataset_type == 'play' else 'markov'
    has_button_states = hasattr(env.unwrapped, '_cur_button_states')
    if 'cube' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm)}
        else:
            agents = {'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing)}
    elif 'scene' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {
                'cube': CubeMarkovOracle(env=env, min_norm=FLAGS.min_norm, max_step=100),
                'button': ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                'drawer': DrawerMarkovOracle(env=env, min_norm=FLAGS.min_norm),
                'window': WindowMarkovOracle(env=env, min_norm=FLAGS.min_norm),
            }
        else:
            agents = {
                'cube': CubePlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'button': ButtonPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'drawer': DrawerPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
                'window': WindowPlanOracle(env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing),
            }
    elif 'puzzle' in FLAGS.env_name:
        if oracle_type == 'markov':
            agents = {'button': ButtonMarkovOracle(env=env, min_norm=FLAGS.min_norm, gripper_always_closed=True)}
        else:
            agents = {
                'button': ButtonPlanOracle(
                    env=env, noise=FLAGS.noise, noise_smoothing=FLAGS.noise_smoothing, gripper_always_closed=True
                )
            }
    else:
        raise ValueError(f"Unknown env_name: {FLAGS.env_name}")

    # Counters and accumulators
    successful_episodes = 0
    total_steps = 0
    total_train_steps = 0
    num_train_episodes = FLAGS.num_episodes
    num_val_episodes = FLAGS.num_episodes // 10
    target_episodes = num_train_episodes + num_val_episodes

    seq_lengths = []
    episode_latents = []   # list of (Ti, latent_dim)
    episode_actions = []   # list of (Ti, action_dim)
    episode_index = 0

    # Collection loop with retry logic (keeps original behavior)
    while successful_episodes < target_episodes:
        while True:  # retry loop
            if FLAGS.init_cube is not None and 'cube' in FLAGS.env_name:
                env.unwrapped._target_block = FLAGS.init_cube
            ob, info = env.reset(options={'task_id':5})
            if oracle_type == 'markov':
                xi = np.random.uniform(0, FLAGS.noise)
            agent = agents[info['privileged/target_task']]
            agent.reset(ob, info)

            done = False
            step = 0
            ep_qpos = []

            # Per-episode buffers
            ep_pixels = []   # store only pixels per timestep
            ep_latents = []  # store latents per timestep
            ep_actions = []

            while not done:
                if np.random.rand() < FLAGS.p_random_action:
                    # Sample a random action.
                    action = env.action_space.sample()
                else:
                    # Get an action from the oracle.
                    action = agent.select_action(ob, info)
                    action = np.array(action)
                    if oracle_type == 'markov':
                        # Add Gaussian noise to the action.
                        action += np.random.normal(0, [xi, xi, xi, xi*3, xi*10], action.shape)
                action = np.clip(action, -1, 1)

                next_ob, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Validate observation format
                if not (isinstance(ob, dict) and 'pixels' in ob and 'latent' in ob):
                    raise KeyError("Expected observation dict with keys 'pixels' and 'latent'")

                # Append only pixels to the episode file buffer
                pixels = ob['pixels']
                ep_pixels.append(pixels)

                # Append latents and actions to global lists
                lat = ob['latent']

                ep_latents.append(np.asarray(lat))
                ep_actions.append(np.asarray(action.copy()))

                ep_qpos.append(info['prev_qpos'])

                # Handle subtask completion
                # Handle subtask completion: only switch if the episode hasn't terminated
                # AND there are still unsatisfied blocks to switch to.
                if agent.done and not done:
                    successes = env.unwrapped._compute_successes()
                    num_satisfied = int(sum(bool(s) for s in successes))

                    if num_satisfied < int(env.unwrapped._num_cubes):
                        agent_ob, agent_info = env.unwrapped.switch_target_block()
                        agent = agents[agent_info['privileged/target_task']]
                        agent.reset(agent_ob, agent_info)

                ob = next_ob
                step += 1

            # Scene health check
            if 'scene' in FLAGS.env_name:
                ep_qpos_arr = np.array(ep_qpos)
                block_xyzs = ep_qpos_arr[:, 14:17]
                is_healthy = not ((block_xyzs[:,1] >= 0.29).any() or
                                ((block_xyzs[:,1] <= -0.3) & ((block_xyzs[:,2]<0.06) | (block_xyzs[:,2]>0.08))).any())
                if not is_healthy:
                    print("Unhealthy episode — retrying (no data saved).", flush=True)
                    continue
            # If episode ended because of time limit (truncated) AND the final reward is negative,
            # drop this episode (do not save it and do not increment successful_episodes).
            # `reward` holds the last reward from the loop above.
            if truncated and (reward < 0):
                print(f"Episode ended by time limit with reward {reward:.3f} < 0 — dropping episode and retrying.", flush=True)
                # Do NOT increment successful_episodes or episode_index; restart retry loop
                continue


            # Episode successful -> save per-episode pixels and accumulate latents/actions
            successful_episodes += 1
            total_steps += step
            if successful_episodes <= num_train_episodes:
                total_train_steps += step
                print(f"Train Episode {successful_episodes}/{target_episodes} Steps: {step}", flush=True)
            else:
                print(f"Val Episode {successful_episodes}/{target_episodes} Steps: {step}", flush=True)

            # Save episode pixels only
            ep_file = obses_folder / f"episode_{episode_index:05d}.pth"
            # Convert lists to arrays when sensible (keep dtype uint8 for images if available)
            try:
                # try stack if all same shape
                ep_pixels_arr = np.stack(ep_pixels, axis=0)
            except Exception:
                # fallback to list (torch.save will serialize)
                ep_pixels_arr = ep_pixels
            torch.save(ep_pixels_arr, str(ep_file))

            # Record sequence length
            seq_lengths.append(len(ep_pixels))

            # Append latents and actions to global lists (maintain time order)
            episode_latents.append(
                np.stack(ep_latents, axis=0)
            )
            episode_actions.append(
                np.stack(ep_actions, axis=0)
            )
            episode_index += 1
            break  # exit retry loop and move to next episode

    print("Collected total steps:", total_steps)

    # Concatenate latents and actions into tensors and save
    num_eps = len(seq_lengths)
    max_len = max(seq_lengths)

    latent_dim = episode_latents[0].shape[1]
    action_dim = episode_actions[0].shape[1]

    latents_ep = torch.zeros((num_eps, max_len, latent_dim), dtype=torch.float32)
    actions_ep = torch.zeros((num_eps, max_len, action_dim), dtype=torch.float32)

    for i, (lat_ep, act_ep) in enumerate(zip(episode_latents, episode_actions)):
        L = lat_ep.shape[0]
        latents_ep[i, :L] = torch.from_numpy(lat_ep)
        actions_ep[i, :L] = torch.from_numpy(act_ep)

    torch.save(latents_ep, save_folder / "latents.pth")
    torch.save(actions_ep, save_folder / "actions.pth")

    with open(save_folder / "seq_lengths.pkl", "wb") as f:
        pickle.dump(seq_lengths, f)

    print("Saved:")
    print(" latents:", latents_ep.shape)
    print(" actions:", actions_ep.shape)



if __name__ == "__main__":
    app.run(main)
