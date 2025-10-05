"""
train_twowave.py 
This is the best training script for training the slow car
that can be used in the two car racing simulation. It goes
along with test_twowave.py.
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
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import numpy as np
import myconfig 

"""Add in import myconfig before 
putting into donkeycar simulator
"""

"""Basic setup of environment"""

SIM_PATH = os.path.expanduser(myconfig.DONKEY_SIM_PATH)
SIM_PORT = myconfig.DONKEY_GYM_PORT
TIME_SCALE = 3
N_ENVS = 1
TOTAL_STEPS = 200000
SEGMENT = 10000
CAM_RESOLUTION = (120, 160, 3)
RUN_ID = "slow"

"""Global variable for cleanup"""
sim_process = None

"""Checking for keyboard interrupt"""
def signal_handler(sig, frame):
    print('\nCleaning up right now!')
    if sim_process:
        sim_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
"""
Tells computer to call function when 
a keyboard interrupt occurs!
"""

def launch_sim():
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
            print(f"Sim is ready on the sim port: {SIM_PORT}")
            return 
        except OSError:
            time.sleep(1.0)
    raise RuntimeError("SIM did not start in time, issue")
    
def make_env():
    """Building a basic environment"""
    conf = {
        'remote':True,
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
        'car_name': f'{RUN_ID}_car',
        'player_name': f'{RUN_ID}_car',
        'racer_name':  f'{RUN_ID}_car',
        'body_style': 'donkey',   # some builds require a style for labels to init
        'font_size': 100,
        'body_rgb': (0, 255, 0),         
    }
    class ThrottleBiasWrapper(gym.ActionWrapper):
        def __init__(self, env, t_min=0.05):
            super().__init__(env)
            assert 0.0 <= t_min < 1.0, """States t_min must be between 0 and 1"""
            self.t_min = float(t_min)
        
        def action(self, act):
            a = np.asarray(act, dtype=np.float32).copy()
            thr = float(np.clip(a[1], 0.0, 1.0))
            thr_biased =  self.t_min + (1.0 - self.t_min) * thr
            a[1] = thr_biased
            return a

    env = gym.make(myconfig.DONKEY_GYM_ENV_NAME, conf=conf)
    env = ThrottleBiasWrapper(env, t_min=0.05)
    print("IMPORTANT PARAMS FOR REMAP")
    print("Action space: ", env.action_space)
    print("Low: ", env.action_space.low, " high: ", env.action_space.high)
    a = env.action_space.sample()
    print("Shape: ", env.action_space.shape)
    print("Sample action: ", a)
    time.sleep(5)
    return env

"""Creates progress bar inherits from StableBaselines3 library"""
class TqdmCallback(BaseCallback):
    def __init__(self, total_steps, freq=500):
        super().__init__()
        self.bar = tqdm(total=total_steps, desc = 'Train', unit = 'step')
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

"""
Function for evaluating the model
"""

def simple_eval(model, eval_env, n_episodes=5):
    total_reward = 0
    successful_episodes = 0

    print(f"Starting evaluation of 5 episodes now!!!")

    for ep in range(n_episodes):
        try:
            print(f"Evaluating episode {ep+1}/{n_episodes}")

            obs = eval_env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0

            while step_count < 500:
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = eval_env.step(action)
                    info = infos[0] if isinstance(infos, (list, tuple)) else infos

                    

                    

                    

                    reward = float(np.asarray(rewards).squeeze())
                    done = bool(np.asarray(dones).squeeze())
                    cte = abs(float(info.get("cte", 0.0)))
                    step_pen = 0.1 * (cte ** 2)
                    episode_reward += reward - step_pen
                    step_count += 1
                    if done:
                        break
                except Exception as e:
                    print(f" Evaluation step error: {e}")
            
            if step_count > 0:
                total_reward += episode_reward
                successful_episodes += 1

            if successful_episodes > 0:
                return total_reward/ successful_episodes
            else:
                return float("-inf")

            time.sleep(0.5)

        except Exception as e:
            print(f"Evaluation episode {ep+1} error: {e}")
            continue

if __name__ == '__main__':
  try:
      launch_sim()
      time.sleep(2)

      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      print("Using device: ", device)

      print("Building environment for training")

      train_env = DummyVecEnv([make_env]* N_ENVS)
      train_env = VecTransposeImage(train_env)

      """Generating actual model"""
      print("Building the model")

      model = PPO(
          'CnnPolicy',
          train_env,
          learning_rate=5e-4,
          n_epochs=20,
          vf_coef=0.8,
          ent_coef=0.01,
          clip_range=0.25,
          tensorboard_log='./runs',
          device=device,
          verbose=0
      )
      
      total_bar = tqdm(total=TOTAL_STEPS, desc = 'Total', unit = 'step')
      steps_done = 0
      best_eval_reward = -float('inf')

      """ Training Loop beng used """

      while steps_done < TOTAL_STEPS: 
            try: 
              seg = min(SEGMENT, TOTAL_STEPS - steps_done)
              print(f"\nStarting training segment: {seg} steps")

              model.learn(total_timesteps=seg, callback=TqdmCallback(seg, freq = 500))
              steps_done += seg
              total_bar.update(seg)

              model.save(f'ppo_donkeycar_stable_best_{RUN_ID}_{steps_done}')
              print(f"Completed {steps_done}/{TOTAL_STEPS} steps, model saved")

              if steps_done % 20000 == 0:
                print("Ts evaluation happening")
                eval_reward = simple_eval(model, train_env, n_episodes=5)


                try:
                    eval_reward_float = float(np.asarray(eval_reward).squeeze())
                except Exception:
                    eval_reward_float = float(np.mean(eval_reward)) 
                if eval_reward_float > best_eval_reward:
                    best_eval_reward = eval_reward_float
                    model.save(f'ppo_donkeycar_stable_best_{RUN_ID}_{best_eval_reward:.2f}')
                    print(f"new best model saved, here is the evaluation reward: {best_eval_reward:.2f}")
                print(f"Current best reward: {best_eval_reward:.2f}")
            except Exception as e:
                        print(f"Training segment error: {e}")
                        print("Lemme try to restart")
                    
                        if sim_process:
                            sim_process.terminate()
                            time.sleep(3)
                        
                        launch_sim()
                        time.sleep(2)

                        train_env.close()
                        train_env = DummyVecEnv([make_env] * N_ENVS)
                        train_env = VecTransposeImage(train_env)
                        model.set_env(train_env)
      total_bar.close()
      model.save(f'ppo_donkeycar_stable_final_{RUN_ID}')
      print("The training for this model is completed")
      print(f"Best model eval reward: {best_eval_reward:.2f}")

  except KeyboardInterrupt:
    print("\nUser interrupted training")
  except Exception as e:
    print(f"training error: {e}")
  finally:
      if sim_process:
        sim_process.terminate()
        print("Cleanup completed")
