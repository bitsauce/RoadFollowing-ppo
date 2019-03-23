import os
import random
import shutil
import time
from collections import deque

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import transform
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from ppo import PPO
from vae.models import ConvVAE, MlpVAE
from replay_buffers import PrioritizedReplayBuffer
from RoadFollowingEnv.car_racing import RoadFollowingEnv
from utils import VideoRecorder, preprocess_frame, compute_gae

def reward1(state):
    # -10 for driving off-track
    if state.num_contacts == 0: return -10 * 0.1
    # + 1 x throttle
    reward = state.velocity * 0.001
    #reward -= 0.01
    return reward

def create_encode_state_fn(model, with_measurements=False, stack=None):
    def encode_state(state):
        frame = preprocess_frame(state.frame)
        encoded_state = model.encode([frame])[0]
        if with_measurements:
            encoded_state = np.append(encoded_state, [state.throttle, state.steering, state.velocity / 30.0])
        if isinstance(stack, int):
            s1 = np.array(encoded_state)
            if not hasattr(state, "stack"):
                state.stack = [np.zeros_like(encoded_state) for _ in range(stack)]
                state.stack_idx = 0
            state.stack[state.stack_idx % stack] = s1
            state.stack_idx += 1
            concat_state = np.concatenate(state.stack)
            return concat_state
        return np.array(encoded_state)
    return encode_state

def make_env(title=None, frame_skip=0, encode_state_fn=None):
    env = RoadFollowingEnv(title=title,
                           encode_state_fn=encode_state_fn,
                           reward_fn=reward1,
                           throttle_scale=0.1,
                           #steer_scale=0.25,
                           max_speed=30,
                           frame_skip=frame_skip)
    env.seed(0)
    return env

def test_agent(test_env, model, video_filename=None):
    # Init test env
    state, terminal, total_reward = test_env.reset(), False, 0
    rendered_frame = test_env.render(mode="rgb_array")

    # Init video recording
    if video_filename is not None:
        video_recorder = VideoRecorder(video_filename, frame_size=rendered_frame.shape)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    # While non-terminal state
    while not terminal:
        # Take deterministic actions at test time (noise_scale=0)
        action = model.predict([state], greedy=True)[0][0]
        state, reward, terminal, _ = test_env.step(action)

        # Add frame
        rendered_frame = test_env.render(mode="rgb_array")
        if video_recorder: video_recorder.add_frame(rendered_frame)
        total_reward += reward

    # Release video
    if video_recorder:
        video_recorder.release()
    
    return total_reward, test_env.reward

def train(params, model_name, save_interval=10, eval_interval=10, record_eval=True, restart=False):
    # Load pre-trained variational autoencoder
    z_dim = 64
    vae = ConvVAE(input_shape=(84, 84, 1),
                  z_dim=z_dim, models_dir="vae",
                  model_name="mse_cnn_zdim64_beta1_data10k",
                  training=False)
    vae.init_session(init_logging=False)
    if not vae.load_latest_checkpoint():
        raise Exception("Failed to load VAE")

    # State encoding fn
    #with_measurements = True
    with_measurements = False
    stack = None
    encode_state_fn = create_encode_state_fn(vae, with_measurements=with_measurements, stack=stack)

    # Create env
    print("Creating environment")
    env      = make_env(model_name, frame_skip=0, encode_state_fn=encode_state_fn)
    test_env = make_env(model_name + " (Test)", encode_state_fn=encode_state_fn)

    # Traning parameters
    initial_lr       = params["initial_lr"]
    discount_factor  = params["discount_factor"]
    gae_lambda       = params["gae_lambda"]
    ppo_epsilon      = params["ppo_epsilon"]
    value_scale      = params["value_scale"]
    entropy_scale    = params["entropy_scale"]
    horizon          = params["horizon"]
    num_epochs       = params["num_epochs"]
    batch_size       = params["batch_size"]

    print("")
    print("Training parameters:")
    for k, v, in params.items(): print(f"  {k}: {v}")
    print("")


    # Environment constants
    input_shape      = np.array([z_dim])
    if with_measurements: input_shape[0] += 3
    if stack is not None: input_shape[0] *= stack
    num_actions      = env.action_space.shape[0]
    action_min       = env.action_space.low
    action_max       = env.action_space.high

    # Create model
    print("Creating model")
    model = PPO(input_shape, num_actions, action_min, action_max, 
                epsilon=ppo_epsilon, value_scale=value_scale, entropy_scale=entropy_scale,
                output_dir=os.path.join("models", model_name))

    # Prompt to load existing model if any
    if not restart:
        if os.path.isdir(model.log_dir) and len(os.listdir(model.log_dir)) > 0:
            answer = input("Model \"{}\" already exists. Do you wish to continue (C) or restart training (R)? ".format(model_name))
            if answer.upper() == "C":
                model.load_latest_checkpoint()
            elif answer.upper() == "R":
                shutil.rmtree(model.output_dir)
                for d in model.dirs:
                    os.makedirs(d)
            else:
                raise Exception("There are already log files for model \"{}\". Please delete it or change model_name and try again".format(model_name))
    else:
        shutil.rmtree(model.output_dir)
        for d in model.dirs:
            os.makedirs(d)
    #model.init_logging()
    model.write_dict_to_summary("hyperparameters", params, 0)

    # For every episode
    episode_counter = 0
    while episode_counter < 100:


        print(f"Episode {episode_counter} (Step {model.step_idx})")

        # Reset environment
        state, terminal_state, total_reward, total_value = env.reset(), False, 0, 0
        
        # While episode not done
        while not terminal_state:
            states, taken_actions, values, rewards, dones = [], [], [], [], []
            for _ in range(horizon):
                action, value = model.predict([state])
                action, value = action[0], value[0][0]

                env.value_label.text = "V(s)={:.2f}".format(value)

                # Perform action
                new_state, reward, terminal_state, _ = env.step(action)
                env.render()
                total_reward += reward

                # Store state, action and reward
                states.append(state)                      # [T, N, 84, 84, 4]
                taken_actions.append(action)              # [T, N, 3]
                values.append(np.squeeze(value, axis=-1)) # [T, N]
                rewards.append(reward)                    # [T, N]
                dones.append(terminal_state)                        # [T, N]
                state = new_state

                if terminal_state:
                    break


            # Calculate last values (bootstrap values)
            last_values = np.squeeze(model.predict([state])[1], axis=-1)[0] # [N]
            
            advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


            # Flatten arrays
            states        = np.array(states).reshape((-1, *input_shape))       # [T x N, 84, 84, 4]
            taken_actions = np.array(taken_actions).reshape((-1, num_actions)) # [T x N, 3]
            returns       = returns.flatten()                                  # [T x N]
            advantages    = advantages.flatten()                               # [T X N]

            T = len(rewards)
            N = 1
            assert states.shape == (T * N, *input_shape)
            assert taken_actions.shape == (T * N, num_actions)
            assert returns.shape == (T * N,)
            assert advantages.shape == (T * N,)

            # Train for some number of epochs
            model.update_old_policy() # θ_old <- θ
            for _ in range(num_epochs):
                num_samples = len(states)
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                for i in range(int(np.ceil(num_samples / batch_size))):
                    # Evaluate model
                    """if model.step_idx % eval_interval == 0:
                        print("Running evaluation...")
                        avg_reward, value_error = evaluate(model, test_env, discount_factor, frame_stack_size, make_video=True)
                        model.write_to_summary("eval_avg_reward", avg_reward)
                        model.write_to_summary("eval_value_error", value_error)"""
                        
                    # Save model
                    if model.step_idx % save_interval == 0:
                        model.save()

                    # Sample mini-batch randomly
                    begin = i * batch_size
                    end   = begin + batch_size
                    if end > num_samples:
                        end = None
                    mb_idx = indices[begin:end]

                    # Optimize network
                    model.train(states[mb_idx], taken_actions[mb_idx],
                                returns[mb_idx], advantages[mb_idx])

        # Write episodic values
        #model.write_value_to_summary("train/episodic/score", env.reward, episode_counter)
        #model.write_value_to_summary("train/episodic/reward", total_reward, episode_counter)
        #model.write_value_to_summary("train/episodic/q_value", total_q, episode_counter)
        #model.write_episodic_summaries(episode_counter)
        episode_counter += 1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trains an agent in a the RoadFollowing environment")

    # Hyper parameters
    parser.add_argument("--initial_lr", type=float, default=3e-4)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epsilon", type=float, default=0.2)
    parser.add_argument("--value_scale", type=float, default=0.5)
    parser.add_argument("--entropy_scale", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)

    # Training vars
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--record_eval", type=bool, default=True)
    parser.add_argument("-restart", action="store_true")

    params = vars(parser.parse_args())

    # Remove non-hyperparameters
    model_name = params["model_name"]; del params["model_name"]
    seed = params["seed"]; del params["seed"]
    save_interval = params["save_interval"]; del params["save_interval"]
    eval_interval = params["eval_interval"]; del params["eval_interval"]
    record_eval = params["record_eval"]; del params["record_eval"]
    restart = params["restart"]; del params["restart"]

    # Reset tf and set seed
    tf.reset_default_graph()
    if isinstance(seed, int):
        tf.random.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(0)

    # Call main func
    train(params, model_name,
          save_interval=save_interval,
          eval_interval=eval_interval,
          record_eval=record_eval,
          restart=restart)
