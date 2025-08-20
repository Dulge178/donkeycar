#!/usr/bin/env python3
"""
Donkey Car AI Training with Stable Baselines3
Trains various RL algorithms on the Waveshare track
"""

import os
import sys
import numpy as np
import gym
import gym_donkeycar
from stable_baselines3 import DQN, PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import matplotlib.pyplot as plt

# Add Donkey Car paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'donkeycar'))
sys.path.append(os.path.join(current_dir, 'gym-donkeycar'))

class DonkeyCarWrapper(gym.Wrapper):
    """Wrapper to handle DonkeyCar specific requirements"""
    
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0]),  # [steering, throttle]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
    def reset(self):
        obs = self.env.reset()
        return self._preprocess_obs(obs)
    
    def step(self, action):
        # Ensure action is in correct format
        if isinstance(action, np.ndarray):
            action = action.tolist()
        
        obs, reward, done, info = self.env.step(action)
        obs = self._preprocess_obs(obs)
        
        # Calculate custom reward
        custom_reward = self._calculate_reward(info)
        
        return obs, custom_reward, done, info
    
    def _preprocess_obs(self, obs):
        """Preprocess observation (image)"""
        if obs is not None:
            # Resize image to standard size
            obs = self._resize_image(obs, (80, 120))
            # Normalize pixel values
            obs = obs.astype(np.float32) / 255.0
        return obs
    
    def _resize_image(self, image, size):
        """Resize image using OpenCV"""
        import cv2
        return cv2.resize(image, size)
    
    def _calculate_reward(self, info):
        """Calculate custom reward based on DonkeyCar info"""
        reward = 0.0
        
        speed = info.get('speed', 0)
        cte = info.get('cte', 0)
        progress = info.get('progress', 0)
        crashed = info.get('crashed', False)
        finished = info.get('finished', False)
        
        if not crashed:
            # Base reward for staying alive
            reward += 1.0
            
            # Speed reward (optimal range for Waveshare)
            if 0.6 <= speed <= 1.2:
                reward += speed * 4.0
            elif speed < 0.6:
                reward += speed * 2.0
            else:
                reward += 1.5
            
            # Track following penalty
            cte_penalty = abs(cte) * 4.0
            reward -= cte_penalty
            
            # Progress reward
            reward += progress * 20.0
        else:
            reward -= 150.0
        
        if finished:
            reward += 300.0
        
        return reward

def create_env(env_name="donkey-waveshare-v0", monitor_dir=None):
    """Create and wrap the DonkeyCar environment"""
    
    # Create base environment
    env = gym.make(env_name, conf={
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "AI_Car",
        "font_size": 100,
        "racer_name": "AI_Agent",
        "country": "USA",
        "bio": "AI Training with Stable Baselines3"
    })
    
    # Wrap with custom wrapper
    env = DonkeyCarWrapper(env)
    
    # Add monitoring if directory specified
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, monitor_dir)
    
    return env

def create_vec_env(env_name="donkey-waveshare-v0", n_envs=1, monitor_dir=None):
    """Create vectorized environment"""
    
    def make_env():
        return create_env(env_name, monitor_dir)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Optional: Stack frames for better temporal understanding
    # env = VecFrameStack(env, n_stack=4)
    
    return env

def train_dqn(env, total_timesteps=1000000, save_path="models/dqn_waveshare"):
    """Train DQN model"""
    
    print("�� Training DQN model...")
    
    # Create DQN model
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0001,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.95,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[512, 256, 128]
        )
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_path,
        name_prefix="dqn_model"
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback
    )
    
    # Save final model
    model.save(f"{save_path}/dqn_final")
    print(f"✅ DQN training completed! Model saved to {save_path}")
    
    return model

def train_ppo(env, total_timesteps=1000000, save_path="models/ppo_waveshare"):
    """Train PPO model"""
    
    print("�� Training PPO model...")
    
    # Create PPO model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=4,
        target_kl=None,
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 256], vf=[512, 256])]
        )
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_path,
        name_prefix="ppo_model"
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback
    )
    
    # Save final model
    model.save(f"{save_path}/ppo_final")
    print(f"✅ PPO training completed! Model saved to {save_path}")
    
    return model

def train_a2c(env, total_timesteps=1000000, save_path="models/a2c_waveshare"):
    """Train A2C model"""
    
    print("�� Training A2C model...")
    
    # Create A2C model
    model = A2C(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        rms_prop_eps=1e-5,
        use_rms_prop=True,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs=dict(
            net_arch=[dict(pi=[512, 256], vf=[512, 256])]
        )
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_path,
        name_prefix="a2c_model"
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback
    )
    
    # Save final model
    model.save(f"{save_path}/a2c_final")
    print(f"✅ A2C training completed! Model saved to {save_path}")
    
    return model

def evaluate_model(model, env, n_eval_episodes=10):
    """Evaluate a trained model"""
    
    print(f"🧪 Evaluating model over {n_eval_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    successful_episodes = 0
    
    for episode in range(n_eval_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if info.get('finished', False):
                successful_episodes += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = (successful_episodes / n_eval_episodes) * 100
    
    print(f"\n�� Evaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Mean Episode Length: {np.mean(episode_lengths):.1f}")
    
    return episode_rewards, episode_lengths, success_rate

def plot_training_results(monitor_dir, save_path="training_results.png"):
    """Plot training results from monitor data"""
    
    try:
        import pandas as pd
        
        # Read monitor data
        monitor_files = [f for f in os.listdir(monitor_dir) if f.startswith('monitor')]
        if not monitor_files:
            print("No monitor files found for plotting")
            return
        
        # Read the first monitor file
        monitor_file = os.path.join(monitor_dir, monitor_files[0])
        data = pd.read_csv(monitor_file, skiprows=1)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot episode rewards
        ax1.plot(data['r'])
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Plot episode lengths
        ax2.plot(data['l'])
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Length')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📈 Training plots saved to {save_path}")
        
    except ImportError:
        print("pandas not available for plotting")

def main():
    """Main training function"""
    
    print("🚗 Donkey Car AI Training with Stable Baselines3")
    print("=" * 60)
    print("⚠️  Make sure simulator is running in a separate terminal!")
    print("   cd DonkeySimLinux && ./donkey_sim.x86_64")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Create environment
    print("�� Creating DonkeyCar environment...")
    env = create_vec_env("donkey-waveshare-v0", n_envs=1, monitor_dir="logs/monitor")
    
    # Choose algorithm to train
    algorithm = input("Choose algorithm (dqn/ppo/a2c/all): ").lower().strip()
    
    if algorithm == "dqn" or algorithm == "all":
        dqn_model = train_dqn(env, total_timesteps=500000)
        evaluate_model(dqn_model, env)
    
    if algorithm == "ppo" or algorithm == "all":
        ppo_model = train_ppo(env, total_timesteps=500000)
        evaluate_model(ppo_model, env)
    
    if algorithm == "a2c" or algorithm == "all":
        a2c_model = train_a2c(env, total_timesteps=500000)
        evaluate_model(a2c_model, env)
    
    # Plot results if monitor data exists
    if os.path.exists("logs/monitor"):
        plot_training_results("logs/monitor")
    
    print("\n�� Training completed!")
    print("📁 Models saved in 'models/' directory")
    print("📊 Logs saved in 'logs/' directory")
    
    env.close()

if __name__ == "__main__":
    main()
