# qlearn.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Online network
        self.online_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, action_size)
        )
        
        # Target network
        self.target_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, action_size)
        )
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        
        # Hyperparameters
        self.memory = deque(maxlen=50000)
        self.batch_size = 128
        self.gamma = 0.99  # encourage long-term rewards
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.learning_rate = 0.0005
        self.epsilon_decay_step = (1.0 - self.epsilon_min) / 100  # linear decay over 100 episodes
        
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.criterion = nn.SmoothL1Loss()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.online_network.to(self.device)
        self.target_network.to(self.device)
        self.train_steps = 0
        self.update_target_every = 200  # Update target network every N steps
        
        # TensorBoard writer
        self.writer = SummaryWriter('runs/dqn_training')
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_step
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
    
    def forward(self, state):
        return self.online_network(state)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def act(self, state, is_eval=False):
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.online_network.eval()  # Set to evaluation mode for action selection
        with torch.no_grad():
            action_values = self.online_network(state_t)
        self.online_network.train()  # Revert back to training mode
        return torch.argmax(action_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a mini-batch from memory
        mini_batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays
        states = np.array([sample[0] for sample in mini_batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in mini_batch], dtype=np.int64)
        rewards = np.array([sample[2] for sample in mini_batch], dtype=np.float32)
        next_states = np.array([sample[3] for sample in mini_batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in mini_batch], dtype=np.float32)
        
        # Convert to tensors
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        
        # Current Q values
        current_q = self.online_network(states).gather(1, actions).squeeze()
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.criterion(current_q, target)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Log loss to TensorBoard
        self.writer.add_scalar('Loss', loss.item(), self.train_steps)
        
        # Update target network periodically
        if self.train_steps % self.update_target_every == 0:
            self.update_target_network()
        
        self.train_steps += 1
    
    def close_writer(self):
        self.writer.close()

class StockTradingEnv:
    def __init__(self, df):
        self.data_dict = df  # Dictionary of DataFrames for each stock
        self.action_space = 5  # Simplified actions: Buy 25%, Buy 50%, Sell 25%, Sell 50%, Hold
        self.action_size = self.action_space
        self.initial_balance = 3333.33
        self.state_space = 5  # [Close_scaled, Volume_scaled, RSI_scaled, MACD_scaled, Signal_Line_scaled]
        
        self.transaction_fee = 0.001
        self.holding_penalty = 0.0002
        self.reset()
    
    def step(self, stock, action):
        if self.current_steps[stock] >= len(self.data_dict[stock]) - 1:
            self.done = True
            return None, 0, self.done
        
        current_price = self.data_dict[stock].iloc[self.current_steps[stock]]['Close']  # Use original Close price
        current_price_scaled = self.data_dict[stock].iloc[self.current_steps[stock]]['Close_scaled']  # Use scaled price for state
        
        # Safety check for zero or negative prices
        if current_price <= 0:
            return self._get_state(stock), 0, self.done
        
        previous_portfolio = self.balances[stock] + self.positions[stock] * current_price
        reward = 0
        
        # Simplified Action Mapping
        if action == 0:  # Buy 25%
            percentage = 0.25
            max_purchase = percentage * self.balances[stock]
            shares_to_buy = max_purchase / current_price
            cost = shares_to_buy * current_price * (1 + self.transaction_fee)
            if cost <= self.balances[stock] and shares_to_buy > 0:
                self.positions[stock] += shares_to_buy
                self.balances[stock] -= cost
                reward -= self.transaction_fee  # Transaction fee as penalty
        
        elif action == 1:  # Buy 50%
            percentage = 0.50
            max_purchase = percentage * self.balances[stock]
            shares_to_buy = max_purchase / current_price
            cost = shares_to_buy * current_price * (1 + self.transaction_fee)
            if cost <= self.balances[stock] and shares_to_buy > 0:
                self.positions[stock] += shares_to_buy
                self.balances[stock] -= cost
                reward -= self.transaction_fee
        
        elif action == 2:  # Sell 25%
            percentage = 0.25
            shares_to_sell = percentage * self.positions[stock]
            sale_value = shares_to_sell * current_price * (1 - self.transaction_fee)
            if shares_to_sell > 0 and shares_to_sell <= self.positions[stock]:
                self.positions[stock] -= shares_to_sell
                self.balances[stock] += sale_value
                reward -= self.transaction_fee
        
        elif action == 3:  # Sell 50%
            percentage = 0.50
            shares_to_sell = percentage * self.positions[stock]
            sale_value = shares_to_sell * current_price * (1 - self.transaction_fee)
            if shares_to_sell > 0 and shares_to_sell <= self.positions[stock]:
                self.positions[stock] -= shares_to_sell
                self.balances[stock] += sale_value
                reward -= self.transaction_fee
        
        elif action == 4:  # Hold
            reward -= self.holding_penalty  # Optional holding penalty
        
        # Move to the next step
        self.current_steps[stock] += 1
        
        if self.current_steps[stock] < len(self.data_dict[stock]):
            next_price = self.data_dict[stock].iloc[self.current_steps[stock]]['Close']  # Use original Close price
            # Update portfolio value
            current_portfolio = self.balances[stock] + (self.positions[stock] * next_price)
            # Reward is the percentage change in portfolio
            reward += (current_portfolio - previous_portfolio) / self.initial_balance
        else:
            self.done = True
            # Final portfolio change
            current_portfolio = self.balances[stock] + (self.positions[stock] * current_price)
            reward += (current_portfolio - self.initial_balance) / self.initial_balance
        
        next_state = self._get_state(stock)
        return next_state, reward, self.done
    
    def step_all(self, actions):
        """Handle multiple actions for all stocks simultaneously"""
        next_states = {}
        rewards = {}
        
        for stock, action in actions.items():
            next_state, reward, _ = self.step(stock, action)
            next_states[stock] = next_state
            rewards[stock] = reward
        
        return next_states, rewards, self.done
    
    def reset(self):
        self.balances = {stock: self.initial_balance for stock in self.data_dict}
        self.positions = {stock: 0 for stock in self.data_dict}
        self.current_steps = {stock: 0 for stock in self.data_dict}
        self.done = False
        states = {stock: self._get_state(stock) for stock in self.data_dict}
        return states
    
    def _get_state(self, stock):
        row = self.data_dict[stock].iloc[self.current_steps[stock]]
        return np.array([
            row['Close_scaled'], 
            row['Volume_scaled'], 
            row['RSI_scaled'],
            row['MACD_scaled'], 
            row['Signal_Line_scaled'],
        ], dtype=np.float32)

def train_agents(env, episodes=200):
    # Initialize a single agent for all stocks
    state_size = env.state_space
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    
    # TensorBoard writer
    writer = SummaryWriter('runs/dqn_training')
    
    training_history = {stock: {'net_worth': [], 'positions': []} for stock in env.data_dict}
    
    for episode in range(1, episodes + 1):
        states = env.reset()
        total_rewards = {stock: 0 for stock in env.data_dict}
        
        while not env.done:
            actions = {}
            for stock in env.data_dict:
                if states[stock] is not None:
                    actions[stock] = agent.act(states[stock])
            
            next_states, rewards, done = env.step_all(actions)
            
            for stock in env.data_dict:
                if states[stock] is not None and next_states[stock] is not None:
                    agent.remember(states[stock], actions[stock], rewards[stock], next_states[stock], done)
                    agent.replay()
                    total_rewards[stock] += rewards[stock]
                    
                    # Record training history
                    portfolio_value = env.balances[stock] + (env.positions[stock] * env.data_dict[stock].iloc[env.current_steps[stock]]['Close'])
                    training_history[stock]['net_worth'].append(portfolio_value)
                    training_history[stock]['positions'].append(actions[stock])
            
            states = next_states
        
        # Decay epsilon at the end of each episode
        agent.decay_epsilon()
        
        # Log rewards to TensorBoard
        for stock in env.data_dict:
            writer.add_scalar(f'Reward/{stock}', total_rewards[stock], episode)
            writer.add_scalar(f'Portfolio_Value/{stock}', training_history[stock]['net_worth'][-1], episode)
            writer.add_scalar(f'Epsilon', agent.epsilon, episode)
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{episodes}")
            for stock in env.data_dict:
                portfolio_value = env.balances[stock] + (env.positions[stock] * env.data_dict[stock].iloc[env.current_steps[stock]]['Close'])
                pct_change = ((portfolio_value - env.initial_balance) / env.initial_balance) * 100
                print(f"{stock} - Portfolio: ${portfolio_value:.2f} ({pct_change:+.2f}%), Epsilon: {agent.epsilon:.3f}")
    
    agent.close_writer()
    return agent, training_history

def evaluate_agents(env, agent):
    """
    Evaluates the performance of the trained agent in the environment without exploration.
    """
    states = env.reset()
    portfolio_values = {stock: [env.initial_balance] for stock in env.data_dict}
    actions_taken = {stock: [] for stock in env.data_dict}
    
    # Disable exploration during evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    while not env.done:
        actions = {}
        for stock in env.data_dict:
            if states[stock] is not None:
                actions[stock] = agent.act(states[stock], is_eval=True)
                actions_taken[stock].append(actions[stock])
        
        next_states, rewards, done = env.step_all(actions)
        
        # Update portfolio values
        for stock in env.data_dict:
            if states[stock] is not None:
                current_price = env.data_dict[stock].iloc[env.current_steps[stock]]['Close']
                portfolio_value = env.balances[stock] + (env.positions[stock] * current_price)
                portfolio_values[stock].append(portfolio_value)
        
        states = next_states
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Print evaluation results
    print("\nEvaluation Results:")
    for stock in env.data_dict:
        final_value = portfolio_values[stock][-1]
        pct_change = ((final_value - env.initial_balance) / env.initial_balance) * 100
        print(f"{stock} Final Portfolio Value: ${final_value:.2f} ({pct_change:+.2f}%)")
    
    return portfolio_values, actions_taken

def plot_trade_histories(trade_histories, data):
    """
    Plots the trade histories for each stock, including portfolio value, stock price, and trading actions.
    """
    for stock, trades in trade_histories.items():
        net_worths = trades["net_worth"]
        positions = trades["positions"]
        steps = list(range(len(net_worths)))
        
        # Ensure that stock_prices are aligned with the net_worths length
        stock_prices_full = data[stock]['Close'].values
        stock_prices = stock_prices_full[:len(net_worths)]
        
        # If stock_prices are shorter, pad with the last available price
        if len(stock_prices) < len(net_worths):
            pad_length = len(net_worths) - len(stock_prices)
            stock_prices = np.concatenate([stock_prices, np.full(pad_length, stock_prices[-1])])
        
        plt.figure(figsize=(12, 8))

        # Plot portfolio net worth
        plt.subplot(2, 1, 1)
        plt.plot(steps, net_worths, label='Net Worth', color="blue")
        plt.title(f'{stock} Portfolio Value Over Time')
        plt.xlabel('Trading Steps')
        plt.ylabel('Net Worth ($)')
        plt.legend()
        plt.grid(True)

        # Plot stock price and actions
        plt.subplot(2, 1, 2)
        plt.plot(stock_prices, label='Stock Price ($)', color="orange", alpha=0.7)

        # Overlay buy and sell actions
        buy_steps = [i for i, action in enumerate(positions) if action in [0, 1]]  # Buy 25%, Buy 50%
        sell_steps = [i for i, action in enumerate(positions) if action in [2, 3]]  # Sell 25%, Sell 50%
        
        plt.scatter(buy_steps, stock_prices[buy_steps], color="green", marker="^", label="Buy")
        plt.scatter(sell_steps, stock_prices[sell_steps], color="red", marker="v", label="Sell")

        plt.title(f'{stock} Stock Price and Trading Actions')
        plt.xlabel('Trading Steps')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        # Alternatively, save the plot
        # plt.savefig(f'{stock}_trade_history.png')
        # plt.close()

# Main execution
if __name__ == "__main__":
    from load_data import load_data
    
    # Load and preprocess data
    qqq, spy, voo = load_data()
    
    # Split data into training and testing sets
    split_ratio = 0.8
    split_index = int(len(qqq) * split_ratio)
    
    train_qqq = qqq.iloc[:split_index].copy()
    test_qqq = qqq.iloc[split_index:].copy()
    
    train_spy = spy.iloc[:split_index].copy()
    test_spy = spy.iloc[split_index:].copy()
    
    train_voo = voo.iloc[:split_index].copy()
    test_voo = voo.iloc[split_index:].copy()
    
    # Prepare training and testing data dictionaries
    train_data = {
        'QQQ': train_qqq,
        'SPY': train_spy,
        'VOO': train_voo
    }
    test_data = {
        'QQQ': test_qqq,
        'SPY': test_spy,
        'VOO': test_voo
    }
    
    # Create environments
    env = StockTradingEnv(train_data)
    test_env = StockTradingEnv(test_data)
    
    # Train agents
    print("\nTraining agents...")
    agent, training_history = train_agents(env, episodes=200)
    
    # Evaluate agents
    print("\nEvaluating agents...")
    portfolio_values, actions_taken = evaluate_agents(test_env, agent)
    
    # Plot training history with train_data
    print("\nPlotting Training History...")
    plot_trade_histories(training_history, train_data)
    
    # Prepare and plot evaluation history with test_data
    evaluation_history = {}
    for stock in portfolio_values:
        evaluation_history[stock] = {
            'net_worth': portfolio_values[stock],
            'positions': actions_taken[stock]
        }
    
    print("\nPlotting Evaluation History...")
    plot_trade_histories(evaluation_history, test_data)
    
    # Print final performance metrics for evaluation
    print("\nFinal Performance Metrics (Evaluation):")
    for stock in portfolio_values:
        final_value = portfolio_values[stock][-1]
        initial_value = test_env.initial_balance
        total_return = ((final_value - initial_value) / initial_value) * 100
        print(f"{stock}:")
        print(f"  Initial Balance: ${initial_value:.2f}")
        print(f"  Final Value: ${final_value:.2f}")
        print(f"  Total Return: {total_return:+.2f}%")
