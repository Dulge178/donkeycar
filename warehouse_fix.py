"""
train_warehouse.py - fine-tune from a saved PPO model with stricter rewards and smoother steering
"""
import os, subprocess, socket, time, signal, sys
import gym
import gym_donkeycar
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import numpy as np

import myconfig

### SIM / TRAIN CONSTANTS ###
SIM_PATH = os.path.expanduser(myconfig.DONKEY_SIM_PATH)
SIM_PORT = myconfig.DONKEY_GYM_PORT
TIME_SCALE = 2
N_ENVS = 1
TOTAL_STEPS = 900_000
SEGMENT = 20_000
CAM_RESOLUTION = (120, 160, 3)
sim_process = None

### REWARD / SMOOTHING HYPERPARAMS (tune these) ###
# Steering rate limit per step (absolute change). Lower = smoother.
STEER_RATE_LIMIT = 0.08
# Penalize change in steering (steer jerk) each step.
W_STEER_DELTA = 1.2
# Penalize absolute steering magnitude (keeps car straighter).
W_STEER_ABS = 0.25
# Penalize cross-track error if env provides info['cte'] (centerline distance).
W_CTE = 0.25
# Extra penalty when 'done' due to off-track/collision (if info indicates)
OFFTRACK_PENALTY = 5.0

# When continuing training, slightly stricter PPO (safer fine-tune)
FT_LEARNING_RATE = 2e-4
FT_CLIP_RANGE = 0.2
FT_ENT_COEF = 0.02
FT_VF_COEF = 0.9
FT_EPOCHS = 15

### graceful ctrl+c ###
def signal_handler(sig, frame):
    print('\nCleaning up ')
    if sim_process:
        sim_process.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

### launch simulator (headless) ###
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
    raise RuntimeError("Simulation didn't start")

##############################
# Wrappers for smoothing & reward shaping
##############################

class SmoothActionWrapper(gym.ActionWrapper):
    """
    Rate-limit steering to prevent sudden jumps.
    Assumes action = [steering, throttle] within [-1,1] ranges typically.
    """
    def __init__(self, env, steer_rate_limit=0.1):
        super().__init__(env)
        self.prev_action = None
        self.steer_rate_limit = float(steer_rate_limit)

    def action(self, action):
        # make sure it's a 1D np array
        a = np.array(action, dtype=np.float32).copy()
        if self.prev_action is None:
            self.prev_action = np.zeros_like(a)
        # Rate limit steering (idx 0)
        delta = np.clip(a[0] - self.prev_action[0],
                        -self.steer_rate_limit, self.steer_rate_limit)
        a[0] = self.prev_action[0] + delta
        self.prev_action = a
        return a

    def reset(self, **kwargs):
        self.prev_action = None
        return super().reset(**kwargs)


class RewardShaper(gym.Wrapper):
    """
    Adds penalties for:
      - Large change in steering per step (jerk)
      - Large absolute steering (wander)
      - Large cross-track error (cte), if provided by env info
      - Early termination due to off-track/collision
    """
    def __init__(self, env,
                 w_delta=W_STEER_DELTA,
                 w_abs=W_STEER_ABS,
                 w_cte=W_CTE,
                 offtrack_penalty=OFFTRACK_PENALTY):
        super().__init__(env)
        self.w_delta = float(w_delta)
        self.w_abs = float(w_abs)
        self.w_cte = float(w_cte)
        self.offtrack_penalty = float(offtrack_penalty)
        self.prev_steer = 0.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # steer component assumed at index 0
        steer = float(np.asarray(action)[0])
        steer_delta = abs(steer - self.prev_steer)
        self.prev_steer = steer

        shaped = float(reward)

        # Penalize steering jerk and large steering
        shaped -= self.w_delta * steer_delta
        shaped -= self.w_abs * abs(steer)

        # Penalize cross-track error if available
        cte = float(info.get('cte', 0.0)) if isinstance(info, dict) else 0.0
        shaped -= self.w_cte * abs(cte)

        # If episode ended due to crash/off-track, apply extra penalty
        # Common donkeycar infos: 'off_track', 'hit', 'collision', etc.
        if done and isinstance(info, dict):
            if info.get('off_track') or info.get('hit') or info.get('collision'):
                shaped -= self.offtrack_penalty

        return obs, shaped, done, info

    def reset(self, **kwargs):
        self.prev_steer = 0.0
        return super().reset(**kwargs)

##############################

def make_env():
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
    base = gym.make(myconfig.DONKEY_GYM_ENV_NAME, conf=conf)
    # Monitor first so episode stats reflect shaped reward
    env = Monitor(base)
    # Smooth actions BEFORE they reach the env
    env = SmoothActionWrapper(env, steer_rate_limit=STEER_RATE_LIMIT)
    # Shape reward AFTER env computes base reward
    env = RewardShaper(env)
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
    total_reward = 0.0
    successful_episodes = 0
    print(f"Starting evaluation of {n_episodes} episodes...")
    for ep in range(n_episodes):
        try:
            print(f"Evaluating episode {ep+1}/{n_episodes}")
            obs = eval_env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0
            while not done and step_count < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += float(np.asarray(reward).squeeze())
                step_count += 1
            if step_count > 0:
                total_reward += episode_reward
                successful_episodes += 1
                print(f"✅ Episode {ep+1}: reward={episode_reward:.2f}, steps={step_count}")
            time.sleep(0.3)
        except Exception as e:
            print(f"Evaluation episode {ep+1} error: {e}")
    if successful_episodes > 0:
        avg_reward = total_reward / successful_episodes
        print(f"Evaluation result: average reward={avg_reward:.2f} ({successful_episodes}/{n_episodes} episodes successful)")
        return avg_reward
    else:
        print("All evaluation episodes failed")
        return 0.0

if __name__ == '__main__':
    # === Choose which model to load ===
    # If you already saved a "best" model with a score suffix, set that path here.
    # Otherwise this defaults to your noted checkpoint at 380k steps.
    BEST_MODEL_PATH = os.environ.get("BEST_MODEL_PATH", "ppo_donkeycar_stable_step_380000.zip")

    try:
        # 1) Sim
        launch_sim()
        time.sleep(2)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('using device:', device)

        print("Building training environment")
        train_env = DummyVecEnv([make_env] * N_ENVS)
        train_env = VecTransposeImage(train_env)

        print("Loading / Building Model")
        if os.path.exists(BEST_MODEL_PATH):
            print(f"Loading existing model from: {BEST_MODEL_PATH}")
            model = PPO.load(BEST_MODEL_PATH, env=train_env, device=device, print_system_info=False)
            # Optionally tighten hyperparams for fine-tuning:
            model.learning_rate = FT_LEARNING_RATE
            model.clip_range = FT_CLIP_RANGE
            model.ent_coef = FT_ENT_COEF
            model.vf_coef = FT_VF_COEF
            model.n_epochs = FT_EPOCHS
        else:
            print(f"⚠️  {BEST_MODEL_PATH} not found. Creating a new model.")
            model = PPO(
                'CnnPolicy',
                train_env,
                learning_rate=FT_LEARNING_RATE,
                n_epochs=FT_EPOCHS,
                vf_coef=FT_VF_COEF,
                ent_coef=FT_ENT_COEF,
                clip_range=FT_CLIP_RANGE,
                tensorboard_log='./runs',
                device=device,
                verbose=0
            )

        total_bar = tqdm(total=TOTAL_STEPS, desc='Total', unit='step')
        steps_done = 0
        best_eval_reward = -float('inf')

        while steps_done < TOTAL_STEPS:
            try:
                seg = min(SEGMENT, TOTAL_STEPS - steps_done)
                model.learn(total_timesteps=seg, callback=TqdmCallback(seg, freq=500))
                steps_done += seg
                total_bar.update(seg)

                ckpt_name = f'ppo_donkeycar_stable_step_{steps_done}'
                model.save(ckpt_name)
                print(f"✅ Completed {steps_done}/{TOTAL_STEPS} steps, model saved to {ckpt_name}")

                if steps_done % 20000 == 0:
                    print("\n starting evaluation: ")
                    # Use a fresh eval env so state isn't shared
                    eval_env = DummyVecEnv([make_env])
                    eval_env = VecTransposeImage(eval_env)
                    eval_reward = simple_eval(model, eval_env, n_episodes=2)
                    try:
                        eval_reward_float = float(np.asarray(eval_reward).squeeze())
                    except Exception:
                        eval_reward_float = float(np.mean(eval_reward))
                    if eval_reward_float > best_eval_reward:
                        best_eval_reward = eval_reward_float
                        best_name = f'ppo_donkeycar_stable_best_{best_eval_reward:.2f}'
                        model.save(best_name)
                        print(f"🎯 New best model: {best_name} (eval {best_eval_reward:.2f})")
                    print(f"Current best evaluation reward: {best_eval_reward:.2f}")
                    eval_env.close()

            except Exception as e:
                print(f" Training segment error: {e}")
                print("Attempting to restart")
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
        model.save('ppo_donkeycar_stable_final')
        print("Training completed!")
        print(f"Final best evaluation reward: {best_eval_reward:.2f}")

    except KeyboardInterrupt:
        print("\nUser interrupted training")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        if sim_process:
            sim_process.terminate()
        print("Cleanup completed")
