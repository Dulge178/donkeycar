#!/usr/bin/env python3
"""
Simple Donkey Car AI Training Script for Waveshare Track
Trains a DQN model that can drive around the Waveshare track
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from collections import deque
import random

# Add Donkey Car paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'donkeycar'))
sys.path.append(os.path.join(current_dir, 'gym-donkeycar'))

import gym
import gym_donkeycar

class SimpleDQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.model = self._build_model()
        
    def _build_model(self):
        """Build neural network"""
        model = tf.keras.Sequential([
            # Convolutional layers
            tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size),
            tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        """Train on experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1-dones)
        targets_full = self.model.predict_on_batch(states)
        
        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets
        
        self.model.fit(states, targets_full, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def preprocess_image(image):
    """Preprocess camera image"""
    image = tf.image.resize(image, (80, 120))
    image = image / 255.0
    return image

def calculate_reward(speed, cte, progress, crashed, finished):
    """Calculate reward for Waveshare track"""
    reward = 0.0
    
    if not crashed:
        # Base reward for staying alive
        reward += 1.0
        
        # Speed reward (adjusted for Waveshare track characteristics)
        if 0.6 <= speed <= 1.2:  # Lower speed range for Waveshare
            reward += speed * 4.0
        elif speed < 0.6:
            reward += speed * 2.0
        else:
            reward += 1.5
            
        # Track following (Waveshare has tighter turns)
        cte_penalty = abs(cte) * 4.0  # Increased penalty for tight track
        reward -= cte_penalty
        
        # Progress reward
        reward += progress * 20.0  # Higher progress reward for challenging track
            
    else:
        reward -= 150.0  # Higher crash penalty for challenging track
        
    if finished:
        reward += 300.0  # Higher completion reward
        
    return reward

def main():
    print("🚗 Donkey Car AI Training - Waveshare Track")
    print("=" * 50)
    print("⚠️  Make sure simulator is running in a separate terminal!")
    print("   cd DonkeySimLinux && ./donkey_sim.x86_64")
    print("=" * 50)
    
    # Create Waveshare environment
    env = gym.make("donkey-waveshare-v0", conf={
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "AI_Car",
        "font_size": 100,
        "racer_name": "AI_Agent",
        "country": "USA",
        "bio": "AI Training on Waveshare"
    })
    
    # Setup
    state_size = (80, 120, 3)
    action_size = 5  # 0=straight, 1=left, 2=right, 3=accelerate, 4=brake
    
    # Action mapping: convert discrete action to [steering, throttle]
    def map_action(action):
        if action == 0:  # straight
            return [0.0, 0.4]  # Lower throttle for Waveshare
        elif action == 1:  # left
            return [-0.6, 0.4]  # More aggressive steering
        elif action == 2:  # right
            return [0.6, 0.4]   # More aggressive steering
        elif action == 3:  # accelerate
            return [0.0, 0.8]   # Moderate acceleration
        elif action == 4:  # brake
            return [0.0, 0.0]
        else:
            return [0.0, 0.4]
    
    agent = SimpleDQN(state_size, action_size)
    
    # Training parameters (adjusted for Waveshare difficulty)
    episodes = 150  # More episodes for challenging track
    max_steps = 1200  # More steps per episode
    
    # Statistics
    successful_episodes = 0
    best_reward = float('-inf')
    
    print(f"Starting training for {episodes} episodes on Waveshare track...")
    print("Action space: 0=straight, 1=left, 2=right, 3=accelerate, 4=brake")
    print("Track: Waveshare (challenging with tight turns)")
    print("-" * 50)
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        state = preprocess_image(state)
        state = np.expand_dims(state, axis=0)
        
        total_reward = 0
        step_count = 0
        
        print(f"Episode {episode}/{episodes} starting on Waveshare track...")
        
        for step in range(max_steps):
            # Choose action
            action = agent.act(state)
            
            # Map discrete action to continuous [steering, throttle]
            continuous_action = map_action(action)
            
            # Take action
            next_state, reward, done, info = env.step(continuous_action)
            next_state = preprocess_image(next_state)
            next_state = np.expand_dims(next_state, axis=0)
            
            # Get info
            speed = info.get('speed', 0)
            cte = info.get('cte', 0)
            progress = info.get('progress', 0)
            crashed = info.get('crashed', False)
            finished = info.get('finished', False)
            
            # Calculate reward
            advanced_reward = calculate_reward(speed, cte, progress, crashed, finished)
            
            # Store experience
            agent.remember(state, action, advanced_reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += advanced_reward
            step_count += 1
            
            # Print progress
            if step % 150 == 0:  # Less frequent updates for longer episodes
                print(f"  Step {step}: Action={action}, Speed={speed:.2f}, CTE={cte:.2f}")
            
            if done:
                if finished:
                    successful_episodes += 1
                    print(f"  🎉 Episode {episode} completed on Waveshare!")
                elif crashed:
                    print(f"  💥 Episode {episode} crashed on Waveshare!")
                break
        
        # Train agent
        if len(agent.memory) > agent.batch_size:
            agent.replay()
        
        # Save model every 25 episodes (adjusted for longer training)
        if episode % 25 == 0:
            model_path = f"mysim/models/ai_model_waveshare_episode_{episode}.h5"
            agent.model.save(model_path)
            print(f"  💾 Model saved: {model_path}")
        
        # Save best model
        if episode == 1 or total_reward > best_reward:
            best_reward = total_reward
            best_model_path = "mysim/models/ai_best_model_waveshare.h5"
            agent.model.save(best_model_path)
            print(f"  🏆 New best model saved!")
        
        # Print results
        success_rate = (successful_episodes / episode) * 100
        print(f"Episode {episode} completed: Steps={step_count}, Reward={total_reward:.2f}, Success Rate={success_rate:.1f}%")
        print("-" * 50)
    
    # Save final model
    final_model_path = "mysim/models/ai_final_model_waveshare.h5"
    agent.model.save(final_model_path)
    
    # Print summary
    print("\n" + "=" * 50)
    print("�� Training completed on Waveshare track!")
    print("=" * 50)
    print(f"Total episodes: {episodes}")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Success rate: {(successful_episodes/episodes)*100:.1f}%")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    
    env.close()
    
    if (successful_episodes/episodes)*100 >= 50:  # Lower threshold for challenging track
        print("\n🎉 Your AI can drive the Waveshare track!")
    else:
        print("\n⚠️  Your AI needs more training on this challenging track!")
    
    print("\n🚗 Happy driving on Waveshare!")

if __name__ == "__main__":
    main()
