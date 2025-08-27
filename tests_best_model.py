"""
test_best_model.py - Test script for the high-performing 220k step PPO model
Loads the specific model and runs it in the simulator for evaluation
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

def find_best_model():
    """Find the specific high-performing 220k step model"""
    # Use the specific high-performing model at 220k steps
    model_path = "ppo_donkeycar_stable_step_220000.zip"
    if os.path.exists(model_path):
        print("🏆 Using high-performing model: ppo_donkeycar_stable_step_220000.zip")
        print("   (This model showed excellent performance during training)")
        return model_path
    else:
        raise FileNotFoundError("❌ 220k step model not found!")

def test_model(model, env, n_episodes=5, max_steps_per_episode=1000):
    """Test the loaded model for multiple episodes"""
    print(f"\n🚗 Starting model testing for {n_episodes} episodes...")
    
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
                
                print(f"✅ Episode {episode + 1} completed:")
                print(f"   Total reward: {episode_reward:.2f}")
                print(f"   Steps taken: {step_count}")
                print(f"   Final CTE: {info.get('cte', 'N/A') if 'cte' in info else 'N/A'}")
            
            # Brief rest between episodes
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Episode {episode + 1} failed: {e}")
            continue
    
    # Print summary
    if successful_episodes > 0:
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"\n📊 Testing Summary:")
        print(f"   Successful episodes: {successful_episodes}/{n_episodes}")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Average episode length: {avg_length:.1f} steps")
        print(f"   Best episode reward: {max(episode_rewards):.2f}")
        print(f"   Worst episode reward: {min(episode_rewards):.2f}")
        
        return avg_reward
    else:
        print("❌ All episodes failed!")
        return 0.0

def main():
    """Main testing function"""
    try:
        print("🚀 Starting Donkey Car Model Testing")
        print("=" * 50)
        print("🎯 Testing the high-performing 220k step model")
        print("=" * 50)
        
        # 1. Find the specific 220k step model
        print("\n🔍 Looking for 220k step model...")
        model_path = find_best_model()
        
        # 2. Launch simulator
        print("\n🎮 Launching simulator...")
        launch_sim()
        time.sleep(2)  # Wait for simulator to stabilize
        
        # 3. Create test environment
        print("\n🌍 Creating test environment...")
        test_env = DummyVecEnv([make_env] * 1)  # Single environment for testing
        test_env = VecTransposeImage(test_env)
        
        # 4. Load the trained model
        print(f"\n📥 Loading model: {model_path}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"💻 Using device: {device}")
        
        model = PPO.load(model_path, env=test_env, device=device)
        print("✅ Model loaded successfully!")
        
        # 5. Test the model
        print("\n🧪 Starting model testing...")
        test_reward = test_model(model, test_env, n_episodes=5)
        
        print(f"\n🎉 Testing completed!")
        print(f" Final test result: {test_reward:.2f}")
        
        # 6. Performance comparison
        print(f"\n📈 Performance Analysis:")
        print(f"   This 220k step model achieved: {test_reward:.2f} average reward")
        print(f"   Previous 120k step model achieved: 157.32 average reward")
        if test_reward > 157.32:
            improvement = ((test_reward - 157.32) / 157.32) * 100
            print(f"   🚀 Improvement: +{improvement:.1f}%")
        else:
            print(f"   ⚠️  Performance similar to previous model")
        
    except KeyboardInterrupt:
        print("\n⏹️  User interrupted testing")
    except Exception as e:
        print(f"❌ Testing error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if sim_process:
            sim_process.terminate()
        print("✅ Cleanup completed")

if __name__ == '__main__':
    main()
