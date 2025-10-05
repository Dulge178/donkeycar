"""
test_twowave_model.py
This is the testing script for the best model from train_twowave.py. It should be
run with test_onewave.py to generate the racing simulation. 
"""

import os
import subprocess
import socket
import time
import signal
import sys

import gym
import gym_donkeycar
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import numpy as np
import myconfig 

# Configuration - matching train_twowave.py
SIM_PATH = os.path.expanduser(myconfig.DONKEY_SIM_PATH)
SIM_PORT = myconfig.DONKEY_GYM_PORT
TIME_SCALE = 3
N_ENVS = 1
CAM_RESOLUTION = (120, 160, 3)
RUN_ID = "slow"  # Matching the training script
MODEL_FILE = "ppo_donkeycar_stable_best_slow_70000.zip"  # Specific model file

# Global variable for cleanup
sim_process = None

def signal_handler(sig, frame):
    print('\nCleaning up right now!')
    if sim_process:
        sim_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def launch_sim():
    """Launch the simulator in non-headless mode"""
    global sim_process
    cmd = [
        SIM_PATH,
        '--port', str(SIM_PORT),
        '--remote',
        '--time-scale', str(TIME_SCALE)
        # Note: No --headless flag to run in visual mode
    ]
    sim_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 1000
    while time.time() < deadline:
        try:
            socket.create_connection(('localhost', SIM_PORT), timeout=1).close()
            print(f"Sim is ready on the sim port: {SIM_PORT}")
            return 
        except OSError:
            time.sleep(1.0)
    raise RuntimeError("SIM did not start in time, issue")

def make_env():
    """Build environment for testing - matching train_twowave.py config"""
    conf = {
        'remote': True,
        'host': 'localhost',
        'port': SIM_PORT,
        'cam_resolution': CAM_RESOLUTION,
        'abort_on_collision': False,
        'frame_skip': 1,
        'start_delay': 5.0,  
        'max_cte': 8.0,
        'steer_limit': 1.0,
        'throttle_min': 0.0,
        'throttle_max': 1.0,
        'log_level': 30,
        'car_name': f'{RUN_ID}_test_car',
        'player_name': f'{RUN_ID}_test_car',
        'racer_name': f'{RUN_ID}_test_car',
        'body_style': 'donkey',
        'font_size': 100,
        'body_rgb': (0, 255, 0),  # Green color matching train_twowave.py         
    }
    
    class ThrottleBiasWrapper(gym.ActionWrapper):
        def __init__(self, env, t_min=0.05):  # Matching train_twowave.py t_min
            super().__init__(env)
            assert 0.0 <= t_min < 1.0, "t_min must be between 0 and 1"
            self.t_min = float(t_min)
        
        def action(self, act):
            a = np.asarray(act, dtype=np.float32).copy()
            thr = float(np.clip(a[1], 0.0, 1.0))
            thr_biased = self.t_min + (1.0 - self.t_min) * thr
            a[1] = thr_biased
            return a

    env = gym.make(myconfig.DONKEY_GYM_ENV_NAME, conf=conf)
    env = ThrottleBiasWrapper(env, t_min=0.05)  # Using same t_min as training
    print("Environment created for testing (twowave config)")
    print("Action space: ", env.action_space)
    print("Low: ", env.action_space.low, " high: ", env.action_space.high)
    time.sleep(2)
    return env

def test_model(model, test_env, n_episodes=5):
    """Test the model for n episodes"""
    total_reward = 0
    successful_episodes = 0
    
    print(f"Starting test run with {n_episodes} episodes!")
    print("=" * 50)
    
    for ep in range(n_episodes):
        try:
            print(f"\nEpisode {ep+1}/{n_episodes}")
            print("-" * 30)
            
            obs = test_env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0
            max_steps = 1000  # Maximum steps per episode
            
            while step_count < max_steps and not done:
                try:
                    # Get action from model (deterministic for testing)
                    action, _ = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = test_env.step(action)
                    
                    info = infos[0] if isinstance(infos, (list, tuple)) else infos
                    reward = float(np.asarray(rewards).squeeze())
                    done = bool(np.asarray(dones).squeeze())
                    
                    # Calculate step penalty based on CTE (matching training evaluation)
                    cte = abs(float(info.get("cte", 0.0)))
                    step_pen = 0.1 * (cte ** 2)
                    episode_reward += reward - step_pen
                    step_count += 1
                    
                    # Print progress every 100 steps
                    if step_count % 100 == 0:
                        print(f"  Step {step_count}: Reward={reward:.3f}, CTE={cte:.3f}")
                    
                    if done:
                        print(f"  Episode completed in {step_count} steps")
                        break
                        
                except Exception as e:
                    print(f"  Error in step {step_count}: {e}")
                    break
            
            if step_count > 0:
                total_reward += episode_reward
                successful_episodes += 1
                avg_reward = episode_reward / step_count if step_count > 0 else 0
                print(f"  Episode {ep+1} completed:")
                print(f"    Total reward: {episode_reward:.3f}")
                print(f"    Average reward per step: {avg_reward:.3f}")
                print(f"    Steps taken: {step_count}")
            
            # Small delay between episodes
            time.sleep(1)
            
        except Exception as e:
            print(f"Error in episode {ep+1}: {e}")
            continue
    
    if successful_episodes > 0:
        avg_total_reward = total_reward / successful_episodes
        print("\n" + "=" * 50)
        print("TEST RESULTS SUMMARY")
        print("=" * 50)
        print(f"Successful episodes: {successful_episodes}/{n_episodes}")
        print(f"Average reward per episode: {avg_total_reward:.3f}")
        print(f"Total reward: {total_reward:.3f}")
        return avg_total_reward
    else:
        print("No successful episodes completed!")
        return float("-inf")

if __name__ == '__main__':
    try:
        print("Starting DonkeyCar Model Test (TwoWave Config)")
        print("=" * 60)
        print(f"Model: {MODEL_FILE}")
        print("=" * 60)
        
        # Launch simulator
        launch_sim()
        time.sleep(2)
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Create test environment
        print("Building test environment (matching twowave training config)...")
        test_env = DummyVecEnv([make_env] * N_ENVS)
        test_env = VecTransposeImage(test_env)
        
        # Load the specific trained model
        print(f"Loading model: {MODEL_FILE}")
        
        try:
            model = PPO.load(MODEL_FILE, device=device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Make sure the model file '{MODEL_FILE}' exists in the current directory")
            sys.exit(1)
        
        # Run test episodes
        print(f"\nStarting test with model: {MODEL_FILE}")
        test_reward = test_model(model, test_env, n_episodes=5)
        
        print(f"\nFinal test reward: {test_reward:.3f}")
        print("Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")
    finally:
        if sim_process:
            sim_process.terminate()
            print("Cleanup completed")
