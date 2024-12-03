# File: multi_etf_dqn_trading_agent.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Larger network for more complex patterns
        self.fc1 = nn.Linear(self.state_size, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, self.action_size)
        
        # Adjusted hyperparameters for more exploration
        self.memory = deque(maxlen=20000)  # Larger memory
        self.batch_size = 64  # Increased batch size
        self.gamma = 0.85  # Lower gamma for more immediate rewards
        self.epsilon = 1.0  # Start with 100% exploration
        self.epsilon_min = 0.1  # Higher minimum exploration
        self.epsilon_decay = 0.997  # Slower decay
        self.learning_rate = 0.0005  # Lower learning rate
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Track training progress
        self.training_step = 0
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Add more randomness early in training
        exploration_threshold = self.epsilon
        if self.training_step < 1000:  # Extra exploration early on
            exploration_threshold = max(self.epsilon, 0.8)
            
        if np.random.rand() <= exploration_threshold:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.forward(state)
            # More noise early in training
            noise_scale = 0.3 if self.training_step < 1000 else 0.1
            noise = torch.randn_like(q_values) * noise_scale
            return torch.argmax(q_values + noise).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.training_step += 1
        
        # More updates early in training
        num_updates = 8 if self.training_step < 1000 else 4
        
        for _ in range(num_updates):
            mini_batch = random.sample(self.memory, self.batch_size)
            
            states = np.array([sample[0] for sample in mini_batch])
            actions = np.array([sample[1] for sample in mini_batch])
            rewards = np.array([sample[2] for sample in mini_batch])
            next_states = np.array([sample[3] for sample in mini_batch])
            dones = np.array([sample[4] for sample in mini_batch])
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            current_q = self.forward(states).gather(1, actions)
            next_q = self.forward(next_states).max(1)[0].detach()
            target = rewards + (1 - dones) * self.gamma * next_q
            
            # Add some reward scaling for more aggressive learning
            loss = self.criterion(current_q.squeeze(), target)
            
            self.optimizer.zero_grad()
            loss.backward()
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
        
        # Decay epsilon more slowly
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class StockTradingEnv:
    def __init__(self, data_dict, initial_balance=10000):
        self.data_dict = data_dict  # Dictionary of dataframes keyed by stock symbol
        self.initial_balance = initial_balance
    
        # Define action space: -100%, -50%, 0%, +50%, +100%
        self.action_space = np.array([-1, -0.5, 0, 0.5, 1])
        self.action_size = len(self.action_space)
    
        self.reset()
    
    def reset(self):
        self.balances = {stock: self.initial_balance / len(self.data_dict) for stock in self.data_dict}
        self.net_worths = self.balances.copy()
        self.positions = {stock: 0 for stock in self.data_dict}  # Position in percentage
        self.current_steps = {stock: 0 for stock in self.data_dict}
        self.trade_histories = {stock: [] for stock in self.data_dict}
        self.done = False
        return self._get_states()
    
    def _get_states(self):
        states = {}
        for stock, data in self.data_dict.items():
            if self.current_steps[stock] >= len(data) - 1:
                self.done = True
                continue
            
            row = data.iloc[self.current_steps[stock]]
    
            # State features: Using normalized continuous values
            state = np.array([
                (row['Close'] - row['MA5']) / row['Close'],
                (row['Close'] - row['MA20']) / row['Close'],
                row['RSI'] / 100,
                row['MACD'] / 100,  # Normalizing MACD
                row['Signal_Line'] / 100, 
                self.positions[stock]  # Position as continuous value between -1 and 1
            ])
            states[stock] = state
        return states
    
    def step(self, actions):
        rewards = {}
        next_states = {}
        for stock, action_idx in actions.items():
            action = self.action_space[action_idx]
            prev_net_worth = self.net_worths[stock]
            data = self.data_dict[stock]
    
            # Update position
            self.positions[stock] = action
    
            # Move to next step
            self.current_steps[stock] += 1
    
            # Check if at the end of data
            if self.current_steps[stock] >= len(data):
                self.done = True
                continue
    
            # Calculate new net worth
            current_price = data['Close'].iloc[self.current_steps[stock]]
            prev_price = data['Close'].iloc[self.current_steps[stock] -1]
            price_change = (current_price - prev_price) / prev_price
    
            # Update net worth based on position and price change
            self.net_worths[stock] = self.balances[stock] * (1 + self.positions[stock] * price_change)
    
            # Calculate reward (net worth change)
            reward = self.net_worths[stock] - prev_net_worth
            rewards[stock] = reward
    
            # Save trade history
            self.trade_histories[stock].append({
                'step': self.current_steps[stock],
                'position': self.positions[stock],
                'net_worth': self.net_worths[stock],
                'price': current_price,
            })
        next_states = self._get_states()
        return next_states, rewards, self.done
    
    def get_total_net_worth(self):
        return sum(self.net_worths.values())

def train_agents(env, episodes=200):
    agents = {stock: DQNAgent(state_size=6, action_size=env.action_size) for stock in env.data_dict}
    training_history = {
        stock: {
            'portfolio_values': [],
            'actions': [],
            'prices': []
        } for stock in env.data_dict
    }

    for episode in range(episodes):
        states = env.reset()
        total_rewards = {stock: 0 for stock in env.data_dict}
    
        while not env.done:
            actions = {}
            for stock in env.data_dict:
                if stock in states and states[stock] is not None:
                    state = states[stock]
                    action = agents[stock].act(state)
                    actions[stock] = action
                    
                    # Record actions and prices
                    training_history[stock]['actions'].append(action)
                    training_history[stock]['prices'].append(env.data_dict[stock]['Close'].iloc[env.current_steps[stock]])
            
            next_states, rewards, done = env.step(actions)
            
            for stock in env.data_dict:
                if (stock in states and states[stock] is not None and 
                    stock in next_states and next_states[stock] is not None):
                    state = states[stock]
                    action = actions.get(stock)
                    reward = rewards.get(stock, 0)
                    next_state = next_states[stock]
                    done_flag = int(env.done)
                    
                    agents[stock].remember(state, action, reward, next_state, done_flag)
                    agents[stock].replay()
                    total_rewards[stock] = total_rewards.get(stock, 0) + reward
                    
                    # Record portfolio value
                    portfolio_value = env.balances[stock] + (
                        env.positions[stock] * env.data_dict[stock]['Close'].iloc[env.current_steps[stock]]
                    )
                    training_history[stock]['portfolio_values'].append(portfolio_value)
            
            states = next_states
            
        if (episode + 1) % 10 == 0:
            total_net_worth = env.get_total_net_worth()
            print(f"Episode {episode +1}/{episodes}, Total Net Worth: {total_net_worth:.2f}")
    
    return agents, training_history

def evaluate_agents(env, agents):
    states = env.reset()
    total_rewards = {stock: 0 for stock in env.data_dict}
    for stock in agents:
        agents[stock].epsilon = 0  # Disable exploration for evaluation
    
    while not env.done:
        actions = {}
        for stock in env.data_dict:
            state = states[stock]
            action = agents[stock].act(state)
            actions[stock] = action
        next_states, rewards, done = env.step(actions)
        for stock in env.data_dict:
            total_rewards[stock] += rewards[stock]
        states = next_states
    print("Evaluation Results:")
    for stock in env.data_dict:
        print(f"{stock} Final Net Worth: {env.net_worths[stock]:.2f}")
    return env.trade_histories

def plot_trade_histories(trade_histories):
    for stock, trades in trade_histories.items():
        steps = [trade['step'] for trade in trades]
        net_worths = [trade['net_worth'] for trade in trades]
        positions = [trade['position'] for trade in trades]
        prices = [trade['price'] for trade in trades]

        plt.figure(figsize=(12,6))

        plt.subplot(2,1,1)
        plt.plot(steps, net_worths, label='Net Worth')
        plt.title(f'{stock} Net Worth Over Time')
        plt.xlabel('Step')
        plt.ylabel('Net Worth')
        plt.legend()

        plt.subplot(2,1,2)
        plt.plot(steps, prices, label='Price', color='orange')
        # Plot positions as colors
        cmap = plt.get_cmap('bwr')
        norm = plt.Normalize(-1, 1)
        plt.scatter(steps, prices, c=positions, cmap=cmap, norm=norm, label='Position')
        plt.title(f'{stock} Price and Position Over Time')
        plt.xlabel('Step')
        plt.ylabel('Price')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Main Execution
if __name__ == "__main__":
    # Load your data using your existing functions
    from load_data import load_data
    from split import split_data

    # Load data
    qqq, spy, voo = load_data()

    # Split data into training and testing sets
    (train_qqq, test_qqq), (train_spy, test_spy), (train_voo, test_voo) = split_data(qqq, spy, voo)
    train_data = {'qqq': train_qqq, 'spy': train_spy, 'voo': train_voo}
    test_data = {'qqq': test_qqq, 'spy': test_spy, 'voo': test_voo}

    # Ensure that the data includes the necessary indicators
    for df in train_data.values():
        # Calculate technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        delta = df['Close'].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        RS = roll_up / roll_down
        df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
        df['RSI'].fillna(50, inplace=True)
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df.dropna(inplace=True)

    for df in test_data.values():
        # Calculate technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        delta = df['Close'].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        RS = roll_up / roll_down
        df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
        df['RSI'].fillna(50, inplace=True)
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        print(df.head())
        print(df.columns)
        df.dropna(inplace=True)

    # Initialize environment and agents
    env = StockTradingEnv(train_data)
    agents = {stock: DQNAgent(state_size=6, action_size=env.action_size) for stock in train_data}

    # Train the agents
    print("Training agents...")
    train_agents(env, agents, episodes=50)

    # Evaluate the agents
    print("\nEvaluating agents...")
    test_env = StockTradingEnv(test_data)
    trade_histories = evaluate_agents(test_env, agents)

    # Plot the trade histories
    plot_trade_histories(trade_histories)
