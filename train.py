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

from ppo import PPO
from vae.models import ConvVAE, MlpVAE
from RoadFollowingEnv.car_racing import RoadFollowingEnv
from utils import VideoRecorder, preprocess_frame, compute_gae

def reward1(state):
    # -10 for driving off-road
    if state.off_road == True: return -10 * 0.1
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
                           max_speed=30,
                           terminate_off_road=True,
                           terminate_when_stopped=True,
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
        action, _ = model.predict([state], greedy=True)
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
    learning_rate    = params["learning_rate"]
    lr_decay         = params["lr_decay"]
    discount_factor  = params["discount_factor"]
    gae_lambda       = params["gae_lambda"]
    ppo_epsilon      = params["ppo_epsilon"]
    value_scale      = params["value_scale"]
    entropy_scale    = params["entropy_scale"]
    horizon          = params["horizon"]
    num_epochs       = params["num_epochs"]
    num_episodes     = params["num_episodes"]
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
                learning_rate=learning_rate, lr_decay=lr_decay,
                epsilon=ppo_epsilon, value_scale=value_scale, entropy_scale=entropy_scale,
                output_dir=os.path.join("models", model_name))

    # Prompt to load existing model if any
    if not restart:
        if os.path.isdir(model.log_dir) and len(os.listdir(model.log_dir)) > 0:
            answer = input("Model \"{}\" already exists. Do you wish to continue (C) or restart training (R)? ".format(model_name))
            if answer.upper() == "C":
                model.load_latest_checkpoint()
            elif answer.upper() == "R":
                restart = True
            else:
                raise Exception("There are already log files for model \"{}\". Please delete it or change model_name and try again".format(model_name))
    if restart:
        shutil.rmtree(model.output_dir)
        for d in model.dirs:
            os.makedirs(d)
    model.init_logging()
    model.write_dict_to_summary("hyperparameters", params, 0)

    # For every episode
    while model.get_episode_idx() < num_episodes:
        episode_idx = model.get_episode_idx()

        # Save model periodically
        if episode_idx % save_interval == 0:
            model.save()
        
        # Run evaluation periodically
        if episode_idx % eval_interval == 0:
            video_filename = os.path.join(model.video_dir, "episode{}.avi".format(episode_idx))
            eval_reward, eval_score = test_agent(test_env, model, video_filename=video_filename)
            model.write_value_to_summary("eval/score",  eval_score,  episode_idx)
            model.write_value_to_summary("eval/reward", eval_reward, episode_idx)

        # Reset environment
        state, terminal_state, total_reward, total_value = env.reset(), False, 0, 0
        
        # While episode not done
        print(f"Episode {episode_idx} (Step {model.get_train_step_idx()})")
        while not terminal_state:
            states, taken_actions, values, rewards, dones = [], [], [], [], []
            for _ in range(horizon):
                action, value = model.predict([state], write_to_summary=True)

                # Show value on-screen
                env.value_label.text = "V(s)={:.2f}".format(value)

                # Perform action
                new_state, reward, terminal_state, _ = env.step(action)
                env.render()
                total_reward += reward

                # Store state, action and reward
                states.append(state)         # [T, *input_shape]
                taken_actions.append(action) # [T,  num_actions]
                values.append(value)         # [T]
                rewards.append(reward)       # [T]
                dones.append(terminal_state) # [T]
                state = new_state

                if terminal_state:
                    break

            # Calculate last value (bootstrap value)
            _, last_values = model.predict([state]) # []
            
            # Compute GAE
            advantages = compute_gae(rewards, values, last_values, dones, discount_factor, gae_lambda)
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Flatten arrays
            states        = np.array(states)
            taken_actions = np.array(taken_actions)
            returns       = np.array(returns)
            advantages    = np.array(advantages)

            T = len(rewards)
            assert states.shape == (T, *input_shape)
            assert taken_actions.shape == (T, num_actions)
            assert returns.shape == (T,)
            assert advantages.shape == (T,)

            # Train for some number of epochs
            model.update_old_policy() # θ_old <- θ
            for _ in range(num_epochs):
                num_samples = len(states)
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                for i in range(int(np.ceil(num_samples / batch_size))):
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
        model.write_value_to_summary("train/score", env.reward, episode_idx)
        model.write_value_to_summary("train/reward", total_reward, episode_idx)
        model.write_value_to_summary("train/value", total_value, episode_idx)
        model.write_episodic_summaries()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trains an agent in a the RoadFollowing environment")

    # Hyper parameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lr_decay", type=float, default=1.0)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epsilon", type=float, default=0.2)
    parser.add_argument("--value_scale", type=float, default=0.5)
    parser.add_argument("--entropy_scale", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)

    # Training vars
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=5)
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
