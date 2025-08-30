"""
test_warehouse_380k.py - Test script for the high-performing 380k step PPO model
Loads the specific warehouse model and runs it in the simulator for evaluation
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

# ---- Constants ----
SIM_PATH = os.path.expanduser(myconfig.DONKEY_SIM_PATH)
SIM_PORT = myconfig.DONKEY_GYM_PORT
TIME_SCALE = 1  # Normal speed for testing
CAM_RESOLUTION = (120, 160, 3)

# Global variable for cleanup
sim_process = None

def signal_handler(sig, frame):
    print('\nCleaning up...')
    if sim_process:
        sim_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def launch_sim():
    """Launch the Donkey Car simulator"""
    global sim_process
    cmd = [
        SIM_PATH,
        '--port', str(SIM_PORT),
        '--remote', '--nogui',
        '--time-scale', str(TIME_SCALE)
    ]
    sim_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            socket.create_connection(('localhost', SIM_PORT), timeout=1).close()
            print(f"✅ Simulator ready on port {SIM_PORT}")
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("❌ Simulator did not start in time")

def make_env():
    """Create the Donkey Car environment"""
    conf = {
        'remote': True,
        'host': 'localhost',
        'port': SIM_PORT,
        'cam_resolution': CAM_RESOLUTION,
        'abort_on_collision': False,
        'frame_skip': 1,
        'start_delay': 2.0,
        'max_cte': 8.0,
        'steer_limit': 1.0,
        'throttle_min': 0.0,
        'throttle_max': 1.0,
        'log_level': 30,
    }
    env = gym.make(myconfig.DONKEY_GYM_ENV_NAME, conf=conf)
    return env

def find_warehouse_model():
    """Find the specific high-performing 380k step warehouse model"""
    # Use the specific high-performing model at 380k steps
    model_path = "ppo_donkeycar_stable_step_380000.zip"
    if os.path.exists(model_path):
        print(" Using high-performing warehouse model: ppo_donkeycar_stable_step_380000.zip")
        print("   (This model showed the highest reward during warehouse training)")
        return model_path
    else:
        # Fallback: look for any warehouse model
        warehouse_models = glob.glob("ppo_donkeycar_stable_step_*.zip")
        if warehouse_models:
            # Sort by step number and use the highest available
            step_models = [(f, int(f.split('_step_')[1].replace('.zip', ''))) for f in warehouse_models]
            best_step_model = max(step_models, key=lambda x: x[1])[0]
            steps = best_step_model.split('_step_')[1].replace('.zip', '')
            print(f"⚠  380k model not found, using highest available: {best_step_model}")
            print(f"   (This model has {steps} training steps)")
            return best_step_model
        else:
            raise FileNotFoundError(" No warehouse models found! Run training first.")

def test_model(model, env, n_episodes=5, max_steps_per_episode=1000):
    """Test the loaded model for multiple episodes"""
    print(f"\n�� Starting warehouse model testing for {n_episodes} episodes...")
    
    episode_rewards = []
    episode_lengths = []
    successful_episodes = 0
    
    for episode in range(n_episodes):
        print(f"\n Episode {episode + 1}/{n_episodes}")
        
        try:
            obs = env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0
            
            # Episode loop
            while not done and step_count < max_steps_per_episode:
                try:
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Take step in environment
                    obs, reward, done, info = env.step(action)
                    
                    # Update episode stats
                    episode_reward += float(np.asarray(reward).squeeze())
                    step_count += 1
                    
                    # Print progress every 100 steps
                    if step_count % 100 == 0:
                        print(f"  Step {step_count}: reward={episode_reward:.2f}")
                    
                    # Brief pause to make it watchable
                    time.sleep(0.01)
                    
                except Exception as e:
                    print(f"  ❌ Step error: {e}")
                    break
            
            # Episode completed
            if step_count > 0:
                episode_rewards.append(episode_reward)
                episode_lengths.append(step_count)
                successful_episodes += 1
                
                print(f" Episode {episode + 1} completed:")
                print(f"   Total reward: {episode_reward:.2f}")
                print(f"   Steps taken: {step_count}")
                print(f"   Final CTE: {info.get('cte', 'N/A') if 'cte' in info else 'N/A'}")
            
            # Brief rest between episodes
            time.sleep(1)
            
        except Exception as e:
            print(f" Episode {episode + 1} failed: {e}")
            continue
    
    # Print summary
    if successful_episodes > 0:
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"\n📊 Warehouse Testing Summary:")
        print(f"   Successful episodes: {successful_episodes}/{n_episodes}")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Average episode length: {avg_length:.1f} steps")
        print(f"   Best episode reward: {max(episode_rewards):.2f}")
        print(f"   Worst episode reward: {min(episode_rewards):.2f}")
        
        return avg_reward
    else:
        print(" All episodes failed!")
        return 0.0

def main():
    """Main testing function"""
    try:
        print(" Starting Donkey Car Warehouse Model Testing")
        print("=" * 60)
        print(" Testing the high-performing 380k step warehouse model")
        print("=" * 60)
        
        # 1. Find the specific 380k step warehouse model
        print("\n🔍 Looking for 380k step warehouse model...")
        model_path = find_warehouse_model()
        
        # 2. Launch simulator
        print("\n🎮 Launching simulator...")
        launch_sim()
        time.sleep(2)  # Wait for simulator to stabilize
        
        # 3. Create test environment
        print("\n�� Creating warehouse test environment...")
        test_env = DummyVecEnv([make_env] * 1)  # Single environment for testing
        test_env = VecTransposeImage(test_env)
        
        # 4. Load the trained warehouse model
        print(f"\n Loading warehouse model: {model_path}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"💻 Using device: {device}")
        
        model = PPO.load(model_path, env=test_env, device=device)
        print(" Warehouse model loaded successfully!")
        
        # 5. Test the model
        print("\n Starting warehouse model testing...")
        test_reward = test_model(model, test_env, n_episodes=5)
        
        print(f"\n Warehouse testing completed!")
        print(f" Final test result: {test_reward:.2f}")
        
        # 6. Performance analysis
        print(f"\n Warehouse Performance Analysis:")
        print(f"   This 380k step warehouse model achieved: {test_reward:.2f} average reward")
        print(f"   Model was trained on warehouse map with 900k total steps")
        print(f"   Uses CNN policy optimized for warehouse navigation")
        
        # 7. Deployment readiness
        if test_reward > 2200:  # Adjust threshold as needed
            print(f"    Model appears ready for warehouse deployment!")
        elif test_reward > 100:
            print(f"     Model shows promise but may need more training")
        else:
            print(f"    Model may need additional training for deployment")
        
    except KeyboardInterrupt:
        print("\n  User interrupted testing")
    except Exception as e:
        print(f" Testing error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n Cleaning up...")
        if sim_process:
            sim_process.terminate()
        print("Cleanup completed")

if __name__ == '__main__':
    main()
