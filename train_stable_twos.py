"""
train_stable_twos.py
Bener Dulger
9/15/2025
While I am still trying to fix the hardware I wanted to improve our
model as much as possible so I added in a function for stricter reward penalties
as we discussed because I noticed our model sometimes still making mistakes with the
testing script. 
"""
# Imports all the libraries that are necessary: 
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
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import numpy as np

import myconfig

#Just setting up constants for the environemnt
#number of setps and the segment lengths
SIM_PATH       = os.path.expanduser(myconfig.DONKEY_SIM_PATH)
SIM_PORT       = myconfig.DONKEY_GYM_PORT
TIME_SCALE     = 2
N_ENVS         = 1  
TOTAL_STEPS    = 300_000
SEGMENT        = 10_000  
CAM_RESOLUTION = (120, 160, 3)

sim_process = None

def signal_handler(sig, frame):
    print('\nCleaning up...')
    if sim_process:
        sim_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def launch_sim():
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
            print(f"Sim ready on port {SIM_PORT}")
            return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("Sim did not start in time")

#wrapper for reward penalties to improve model overall
class StrictPenaltyWrapper(gym.Wrapper):

    def __init__(
        self,
        env,
        hard_cte=3.0,            
        hard_cte_penalty=2.0,    
        collision_penalty=5.0,   
        offtrack_penalty=3.0,    
        max_cte=8.0              
    ):
        super().__init__(env)
        self.hard_cte = float(hard_cte)
        self.hard_cte_penalty = float(hard_cte_penalty)
        self.collision_penalty = float(collision_penalty)
        self.offtrack_penalty = float(offtrack_penalty)
        self.max_cte = float(max_cte)
        self._prev_cte = None

    # -------- Function One: Stricter penalties --------
    def _apply_strict_penalties(self, reward, info):
        """
        Make penalties bite harder for bad events.
        - Collisions, off-track
        - Large CTE (far from center line)
        - CTE spikes (oscillation)
        """
        r = float(reward)

        # Read values from info dict with safe defaults
        cte = float(info.get('cte', 0.0))
        collision = bool(info.get('collision', False))
        off_track = bool(info.get('off_track', False))

        # 1) Strong, immediate penalties for crash/off-track
        if collision:
            r -= self.collision_penalty

        if off_track:
            r -= self.offtrack_penalty

        # 2) Penalty that grows if you're far from center (beyond hard_cte)
        #    Capped by max_cte to avoid exploding values.
        if abs(cte) > self.hard_cte:
            over = min(abs(cte), self.max_cte) - self.hard_cte
            r -= self.hard_cte_penalty * (1.0 + over)

        # 3) Small penalty for zig-zagging (big jumps in CTE between steps)
        if self._prev_cte is not None:
            delta = abs(cte - self._prev_cte)
            r -= 0.05 * delta   

        # Save for next step's oscillation check
        self._prev_cte = cte
        return r

    # Intercept each step: apply Function One to the base reward
    def reset(self, **kwargs):
        self._prev_cte = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        shaped = float(np.asarray(reward).squeeze())
        shaped = self._apply_strict_penalties(shaped, info)
        return obs, shaped, done, info

def make_env():
    # Simplified configuration to reduce complexity
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

    # Wrap with the strict penalty shaper (Function One)
    env = StrictPenaltyWrapper(
        env,
        hard_cte=3.0,           
        hard_cte_penalty=2.0,     
        collision_penalty=5.0,    
        offtrack_penalty=3.0,    
        max_cte=conf['max_cte'],  #
    )
    return env

class TqdmCallback(BaseCallback):
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

def simple_eval(model, eval_env, n_episodes=2):
    """Simplified evaluation function"""
    total_reward = 0
    successful_episodes = 0

    print(f"Starting evaluation of {n_episodes} episodes...")

    for ep in range(n_episodes):
        try:
            print(f"Evaluating episode {ep+1}/{n_episodes}")

            obs = eval_env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0

            while not done and step_count < 500:  # Limit steps
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = eval_env.step(action)
                    episode_reward += float(np.asarray(reward).squeeze())
                    step_count += 1
                except Exception as e:
                    print(f"  Evaluation step error: {e}")
                    break

            if step_count > 0:
                total_reward += episode_reward
                successful_episodes += 1
                print(f"Episode {ep+1}: reward={episode_reward:.2f}, steps={step_count}")

            time.sleep(0.5)  # Brief rest

        except Exception as e:
            print(f"Evaluation episode {ep+1} error: {e}")
            continue

    if successful_episodes > 0:
        avg_reward = total_reward / successful_episodes
        print(f"Evaluation result: average reward={avg_reward:.2f} ({successful_episodes}/{n_episodes} episodes successful)")
        return avg_reward
    else:
        print("All evaluation episodes failed")
        return 0.0

if __name__ == '__main__':
    try:
        # 1) Launch simulator
        launch_sim()
        time.sleep(2)  # Wait for simulator to stabilize

        # Fixed device selection for Windows (CUDA first, then CPU)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', device)

        # 2) Create training environment
        print("Creating training environment...")
        train_env = DummyVecEnv([make_env] * N_ENVS)
        train_env = VecTransposeImage(train_env)

        # 3) Create model
        print("Creating model...")
        model = PPO(
            'CnnPolicy',
            train_env,
            learning_rate=5e-4,
            n_epochs=10,  # Reduced number of epochs
            vf_coef=0.8,
            ent_coef=0.01,
            clip_range=0.25,
            tensorboard_log='./runs',
            device=device,
            verbose=0
        )

        total_bar = tqdm(total=TOTAL_STEPS, desc='Total', unit='step')
        steps_done = 0
        best_eval_reward = -float('inf')

        # 4) Training loop
        while steps_done < TOTAL_STEPS:
            try:
                seg = min(SEGMENT, TOTAL_STEPS - steps_done)
                print(f"\nStarting training segment: {seg} steps")

                model.learn(total_timesteps=seg, callback=TqdmCallback(seg, freq=500))
                steps_done += seg
                total_bar.update(seg)

                # Save intermediate model
                model.save(f'ppo_donkeycar_stable_step_{steps_done}')
                print(f"Completed {steps_done}/{TOTAL_STEPS} steps, model saved")

                # Simplified evaluation
                if steps_done % 20000 == 0:  # Evaluate every 20k steps
                    print("\nStarting evaluation...")
                    eval_reward = simple_eval(model, train_env, n_episodes=2)

                    # Ensure we have a Python float (avoid ndarray formatting issues)
                    try:
                        eval_reward_float = float(np.asarray(eval_reward).squeeze())
                    except Exception:
                        eval_reward_float = float(np.mean(eval_reward))

                    if eval_reward_float > best_eval_reward:
                        best_eval_reward = eval_reward_float
                        model.save(f'ppo_donkeycar_stable_best_{best_eval_reward:.2f}')
                        print(f"New best model saved. Evaluation reward: {best_eval_reward:.2f}")

                    print(f"Current best evaluation reward: {best_eval_reward:.2f}")

            except Exception as e:
                print(f"Training segment error: {e}")
                print("Attempting to restart...")

                # Restart simulator
                if sim_process:
                    sim_process.terminate()
                    time.sleep(3)

                launch_sim()
                time.sleep(2)

                # Recreate environment
                train_env.close()
                train_env = DummyVecEnv([make_env] * N_ENVS)
                train_env = VecTransposeImage(train_env)
                model.set_env(train_env)

        total_bar.close()
        model.save('ppo_donkeycar_stable_final')
        print("Training completed!")
        print(f"Final best evaluation reward: {best_eval_reward:.2f}")

    except KeyboardInterrupt:
        print("\nUser interrupted training")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        # Cleanup
        if sim_process:
            sim_process.terminate()
        print("Cleanup completed")
