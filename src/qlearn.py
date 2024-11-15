import numpy as np
import pandas as pd

class StockTradingEnv:
    def __init__(self, df):
        self.df = df
        self.reset()
        
        self.action_space = 3
        
        # State space: position + 5 market features
        self.state_space = 6
        
        self.initial_balance = 10000
        self.transaction_fee = 0.001

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.current_price = self.df['Close'].iloc[0]
        self.done = False
        return self._get_state()

    def _get_state(self):
        # Normalize the market features
        market_features = np.array([
            self.df['Open'].iloc[self.current_step],
            self.df['High'].iloc[self.current_step],
            self.df['Low'].iloc[self.current_step],
            self.df['Close'].iloc[self.current_step],
            self.df['Volume'].iloc[self.current_step]
        ])
        
        # Add position information
        position = np.array([self.shares_held])
        
        return np.concatenate([position, market_features])

    def step(self, action):
        previous_value = self.balance + (self.shares_held * self.current_price)
        self.current_step += 1
        self.current_price = self.df['Close'].iloc[self.current_step]
        
        # Execute action
        if action == 1:  # Buy
            shares_to_buy = self.balance // self.current_price
            cost = shares_to_buy * self.current_price * (1 + self.transaction_fee)
            if cost <= self.balance:
                self.shares_held += shares_to_buy
                self.balance -= cost
        elif action == 2:  # Sell
            if self.shares_held > 0:
                sale_value = self.shares_held * self.current_price * (1 - self.transaction_fee)
                self.balance += sale_value
                self.shares_held = 0

        # Calculate reward as percentage change in portfolio value
        current_value = self.balance + (self.shares_held * self.current_price)
        reward = ((current_value - previous_value) / previous_value) * 100  # percentage return
        
        # Check if episode is done
        self.done = self.current_step >= len(self.df) - 1
        
        return self._get_state(), reward, self.done

class QLearningAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = {}
        self.learning_rate = 0.01
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def get_action(self, state):
        state_key = tuple(state.round(4))
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
            
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state):
        state_key = tuple(state.round(4))
        next_state_key = tuple(next_state.round(4))
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space)
            
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(df, episodes=1000):
    env = StockTradingEnv(df)
    agent = QLearningAgent(env.state_space, env.action_space)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while not env.done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            
    return agent
