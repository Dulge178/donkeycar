"""
train_multiagent.py 
Multi-agent competitive racing training script for two cars.
Uses self-play where main agent trains against frozen copies of itself.
"""

import os
import subprocess
import socket
import time
import signal
import sys
import copy

import gym
import gym_donkeycar
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import numpy as np
import myconfig

"""Configuration"""
SIM_PATH = os.path.expanduser(myconfig.DONKEY_SIM_PATH)
SIM_PORT = myconfig.DONKEY_GYM_PORT  # Both cars connect to same port
TIME_SCALE = 3
N_ENVS = 1
TOTAL_STEPS = 400000
SEGMENT = 10000
CAM_RESOLUTION = (120, 160, 3)
RUN_ID = "competitive"

# Opponent management
OPPONENT_UPDATE_FREQ = 20000  # Save new opponent every 20k steps
MAX_OPPONENT_POOL = 5  # Keep last 5 versions
CURRICULUM_STEPS = 50000  # Use random opponent for first 50k steps

"""Global variables"""
sim_process = None
opponent_pool = []

"""Signal handler for cleanup"""
def signal_handler(sig, frame):
    print('\nCleaning up!')
    if sim_process:
        sim_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def launch_sim():
    """Launch simulator that supports 2 cars on same port"""
    global sim_process
    cmd = [
        SIM_PATH,
        '--port', str(SIM_PORT),
        '--remote',
        '--time-scale', str(TIME_SCALE)
    ]
    sim_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 1000
    while time.time() < deadline:
        try:
            socket.create_connection(('localhost', SIM_PORT), timeout=1).close()
            print(f"Sim ready on port: {SIM_PORT}")
            return
        except OSError:
            time.sleep(1.0)
    raise RuntimeError("Simulator did not start in time")

class ThrottleBiasWrapper(gym.ActionWrapper):
    """Same throttle bias from your original code"""
    def __init__(self, env, t_min=0.15):
        super().__init__(env)
        assert 0.0 <= t_min < 1.0, "t_min must be between 0 and 1"
        self.t_min = float(t_min)
    
    def action(self, act):
        a = np.asarray(act, dtype=np.float32).copy()
        thr = float(np.clip(a[1], 0.0, 1.0))
        thr_biased = self.t_min + (1.0 - self.t_min) * thr
        a[1] = thr_biased
        return a

class MultiAgentRacingEnv(gym.Env):
    """
    Wrapper that manages two cars competing on SAME port.
    - Main car (learning agent)
    - Opponent car (frozen policy)
    Both connect to same simulator port; simulator spawns both cars.
    """
    def __init__(self, port, opponent_policy=None):
        super().__init__()
        
        self.port = port
        self.opponent_policy = opponent_policy
        
        # Create environments for both cars - SAME PORT
        self.main_env = self._make_single_env(port, "main")
        self.opp_env = self._make_single_env(port, "opponent")
        
        # Observation space: camera + relative position info
        # [camera(120x160x3), distance_to_opp, am_i_ahead, speed_diff]
        self.observation_space = self.main_env.observation_space
        self.action_space = self.main_env.action_space
        
        # Track state
        self.main_obs = None
        self.opp_obs = None
        self.main_info = {}
        self.opp_info = {}
        self.step_count = 0
        
    def _make_single_env(self, port, car_name):
        """Create a single car environment"""
        # Stagger starting positions to prevent collision
        # Main car starts slightly ahead, opponent starts behind        
        # Opponent gets delayed start to prevent ramming
        
        conf = {
            'remote': True,
            'host': 'localhost',
            'port': port,
            'cam_resolution': CAM_RESOLUTION,
            'abort_on_collision': False,
            'frame_skip': 1,
            'start_delay': 1.0, 
            'max_cte': 8.0,
            'steer_limit': 1.0,
            'throttle_min': 0.0,
            'throttle_max': 1.0,
            'log_level': 30,
            'car_name': f'{RUN_ID}_{car_name}',
            'player_name': f'{RUN_ID}_{car_name}',
            'racer_name': f'{RUN_ID}_{car_name}',
            'body_style': 'donkey',
            'font_size': 100,
            'body_rgb': (255, 0, 0) if car_name == "main" else (0, 0, 255),
        }
        env = gym.make(myconfig.DONKEY_GYM_ENV_NAME, conf=conf)
        env = ThrottleBiasWrapper(env, t_min=0.15)
        return env
    
    def reset(self):
        """Reset both cars"""
        self.main_obs = self.main_env.reset()
        self.opp_obs = self.opp_env.reset()
        self.step_count = 0
        
        # Get initial info
        self.main_info = {}
        self.opp_info = {}
        
        return self.main_obs
    
    def step(self, main_action):
        """
        Step both cars simultaneously.
        Main car uses provided action.
        Opponent uses its frozen policy.
        """
        if self.step_count < 20:
            opp_action = np.array([0.0, 0.0])  
        # Get opponent action
        elif self.opponent_policy is None:
            # Random opponent for curriculum learning
            opp_action = self.opp_env.action_space.sample()
        else:
            # Use frozen policy (no gradients)
            with torch.no_grad():
                opp_action, _ = self.opponent_policy.predict(self.opp_obs, deterministic=True)
        
        # Step both environments
        self.main_obs, main_reward, main_done, self.main_info = self.main_env.step(main_action)
        self.opp_obs, opp_reward, opp_done, self.opp_info = self.opp_env.step(opp_action)
        
        self.step_count += 1
        
        # Calculate competitive reward
        competitive_reward = self._calculate_competitive_reward()
        
        # IMPORTANT: Only end episode if MAIN car is done
        # Don't let opponent crashes ruin main car's training
        done = main_done or self.step_count >= 500
        
        # If opponent crashes but main doesn't, give big win bonus
        if opp_done and not main_done and self.step_count < 500:
            competitive_reward += 20.0  # Opponent crashed, you win!
        
        # Add competitive info to main_info
        self.main_info['competitive_reward'] = competitive_reward
        self.main_info['step_count'] = self.step_count
        self.main_info['opponent_crashed'] = opp_done
        
        return self.main_obs, competitive_reward, done, self.main_info
    
    def _calculate_competitive_reward(self):
        """
        Calculate reward based on racing performance.
        Combines base driving reward with competitive bonuses.
        """
        # Base reward from environment
        base_reward = 0.0
        
        # CTE penalty (staying on track)
        main_cte = abs(float(self.main_info.get('cte', 0.0)))
        cte_penalty = 0.1 * (main_cte ** 2)
        base_reward -= cte_penalty
        
        # Speed reward (encourage going fast)
        main_speed = float(self.main_info.get('speed', 0.0))
        speed_reward = main_speed * 0.01
        base_reward += speed_reward
        
        # Competitive component
        competitive_bonus = 0.0
        
        # Get positions if available
        main_pos = self.main_info.get('pos', None)
        opp_pos = self.opp_info.get('pos', None)
        
        if main_pos is not None and opp_pos is not None:
            # Calculate distance between cars
            distance = np.sqrt(
                (main_pos[0] - opp_pos[0])**2 + 
                (main_pos[2] - opp_pos[2])**2  # Use x,z (y is height)
            )
            
            # Simple heuristic: if main car is ahead in x-direction
            # (assumes track progresses in +x direction, may need adjustment)
            main_progress = main_pos[0]
            opp_progress = opp_pos[0]
            
            if main_progress > opp_progress:
                # Ahead bonus
                competitive_bonus += 2.0
                # Bigger lead = more bonus
                lead = main_progress - opp_progress
                competitive_bonus += min(lead * 0.5, 5.0)  # Cap at +5
            else:
                # Behind penalty
                competitive_bonus -= 1.0
        
        # Collision penalty
        if self.main_info.get('hit', 'none') != 'none':
            competitive_bonus -= 10.0
        
        total_reward = base_reward + competitive_bonus
        return total_reward
    
    def close(self):
        """Close both environments"""
        self.main_env.close()
        self.opp_env.close()
    
    def set_opponent_policy(self, policy):
        """Update the opponent policy"""
        self.opponent_policy = policy

def make_multiagent_env(opponent_policy=None):
    """Factory function to create multi-agent environment"""
    return MultiAgentRacingEnv(SIM_PORT, opponent_policy)

class TqdmCallback(BaseCallback):
    """Progress bar from your original code"""
    def __init__(self, total_steps, freq=500):
        super().__init__()
        self.bar = tqdm(total=total_steps, desc='Train', unit='step')
        self.freq = freq
        self.count = 0

    def _on_step(self) -> bool:
        self.count += 1
        if self.count % self.freq == 0:
            self.bar.update(self.freq)
        return True
    
    def _on_training_end(self):
        rem = self.count % self.freq
        if rem:
            self.bar.update(rem)
        self.bar.close()

def save_opponent_snapshot(model, steps_done):
    """
    Create a frozen copy of the model for opponent pool.
    Returns a copy that won't be trained.
    """
    # Save model to disk
    model_path = f'opponent_snapshot_{RUN_ID}_{steps_done}.zip'
    model.save(model_path)
    
    # Load it back as opponent (frozen)
    opponent = PPO.load(model_path)
    opponent.policy.eval()  # Set to eval mode
    
    print(f"Saved opponent snapshot at {steps_done} steps")
    return opponent

def select_opponent(pool, steps_done):
    """
    Select opponent based on curriculum strategy.
    Early training: random opponent
    Later: mix of recent and older opponents
    """
    # Curriculum: use random opponent for first N steps
    if steps_done < CURRICULUM_STEPS:
        return None  # None = random opponent
    
    # No opponents yet
    if len(pool) == 0:
        return None
    
    # 70% chance to use recent opponent, 30% any opponent
    if np.random.random() < 0.7:
        # Use one of the last 3 opponents
        recent_pool = pool[-min(3, len(pool)):]
        return np.random.choice(recent_pool)
    else:
        # Use any opponent
        return np.random.choice(pool)

if __name__ == '__main__':
    try:
        launch_sim()
        time.sleep(2)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        print("Building multi-agent environment")
        
        # Create environment with no opponent initially (random)
        def env_fn():
            return make_multiagent_env(opponent_policy=None)
        
        train_env = DummyVecEnv([env_fn] * N_ENVS)
        train_env = VecTransposeImage(train_env)
        
        print("Building the model")
        model = PPO(
            'CnnPolicy',
            train_env,
            learning_rate=5e-4,
            n_epochs=20,
            vf_coef=0.8,
            ent_coef=0.01,
            clip_range=0.25,
            tensorboard_log='./runs_competitive',
            device=device,
            verbose=0
        )
        
        total_bar = tqdm(total=TOTAL_STEPS, desc='Total', unit='step')
        steps_done = 0
        opponent_pool = []
        
        print("\n=== Starting Multi-Agent Competitive Training ===")
        print(f"Curriculum learning: Random opponent for first {CURRICULUM_STEPS} steps")
        print(f"Opponent snapshots saved every {OPPONENT_UPDATE_FREQ} steps")
        
        # Training loop
        while steps_done < TOTAL_STEPS:
            try:
                seg = min(SEGMENT, TOTAL_STEPS - steps_done)
                
                # Select opponent based on curriculum
                current_opponent = select_opponent(opponent_pool, steps_done)
                
                if current_opponent is None:
                    print(f"\nSegment {steps_done}: Training against RANDOM opponent")
                else:
                    print(f"\nSegment {steps_done}: Training against opponent from pool (skill level: {len(opponent_pool)} snapshots)")
                
                # CRITICAL: Update the environment's opponent policy!
                # Access the base environment through the wrapper layers
                # DummyVecEnv -> VecTransposeImage -> MultiAgentRacingEnv
                try:
                    # Get through the VecTransposeImage wrapper
                    base_env = train_env.venv.envs[0]  # This is the MultiAgentRacingEnv
                    base_env.set_opponent_policy(current_opponent)
                    print(f"Opponent policy updated in environment")
                except AttributeError:
                    # Fallback: try different wrapper structure
                    base_env = train_env.envs[0]
                    base_env.set_opponent_policy(current_opponent)
                    print(f"Opponent policy updated in environment (fallback method)")
                
                print(f"Training for {seg} steps...")
                model.learn(total_timesteps=seg, callback=TqdmCallback(seg, freq=500))
                steps_done += seg
                total_bar.update(seg)
                
                # Save checkpoint
                model.save(f'ppo_competitive_{RUN_ID}_{steps_done}')
                print(f"Checkpoint saved at {steps_done} steps")
                
                # Save opponent snapshot
                if steps_done % OPPONENT_UPDATE_FREQ == 0 and steps_done >= CURRICULUM_STEPS:
                    opponent = save_opponent_snapshot(model, steps_done)
                    opponent_pool.append(opponent)
                    
                    # Manage pool size
                    if len(opponent_pool) > MAX_OPPONENT_POOL:
                        print(f"Opponent pool full, removing oldest")
                        opponent_pool.pop(0)
                    
                    print(f"Opponent pool size: {len(opponent_pool)}")
                
            except Exception as e:
                print(f"Training segment error: {e}")
                print("Attempting to restart...")
                
                if sim_process:
                    sim_process.terminate()
                    time.sleep(3)
                
                launch_sim()
                time.sleep(2)
                
                # Recreate environment
                train_env.close()
                train_env = DummyVecEnv([env_fn] * N_ENVS)
                train_env = VecTransposeImage(train_env)
                model.set_env(train_env)
        
        total_bar.close()
        model.save(f'ppo_competitive_final_{RUN_ID}')
        print("\n=== Training Complete ===")
        print(f"Final model saved")
        print(f"Opponent pool size: {len(opponent_pool)}")
        
    except KeyboardInterrupt:
        print("\nUser interrupted training")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sim_process:
            sim_process.terminate()
            print("Cleanup completed")
