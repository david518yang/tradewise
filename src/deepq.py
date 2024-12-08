"""Deep Q-Network implementation for stock trading."""

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
import itertools

class DQNAgent(nn.Module):
    """Deep Q-Network agent for stock trading."""
    
    def __init__(self, state_size, action_size, hidden_size=256):
        """Initialize DQN agent with neural network architecture.
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            hidden_size (int): Number of hidden units in neural network
        """
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Enhanced network with market pattern recognition layers
        self.online_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.01),  # Better gradient flow
            nn.Dropout(p=0.1),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.1),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.01),
            
            nn.Linear(hidden_size // 2, action_size)
        )
        
        # Target network with same architecture
        self.target_network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.1),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.1),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.LeakyReLU(0.01),
            
            nn.Linear(hidden_size // 2, action_size)
        )
        
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        
        # Hyperparameters tuned for trading
        self.memory = deque(maxlen=50000)
        self.batch_size = 128
        self.gamma = 0.95  # Higher discount factor for longer-term patterns
        self.epsilon = 1.0
        self.epsilon_min = 0.10
        self.learning_rate = 0.001
        self.epsilon_decay_step = 0.995
        
        # Huber loss for robustness to outliers in market data
        self.criterion = nn.SmoothL1Loss()
        
        # Adam with lower learning rate for stability
        self.optimizer = optim.Adam(
            self.online_network.parameters(), 
            lr=self.learning_rate,
            weight_decay=1e-5  # L2 regularization
        )
        
        # Learning rate scheduler for adaptive learning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            cooldown=2
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.online_network.to(self.device)
        self.target_network.to(self.device)
        
        self.train_steps = 0
        self.update_target_every = 200
        self.writer = SummaryWriter('runs/dqn_training')
        self.running_loss = 0
        self.loss_history = []
    
    def decay_epsilon(self):
        """Decay epsilon value."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay_step
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
    
    def forward(self, state):
        """Forward pass of online network.
        
        Args:
            state: Current state
        
        Returns:
            torch.Tensor: Q-values for each action
        """
        return self.online_network(state)
    
    def update_target_network(self):
        """Update target network weights with current network weights."""
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def act(self, state, is_eval=False):
        """Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            is_eval (bool): Whether in evaluation mode
        
        Returns:
            int: Selected action
        """
        if not is_eval and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Validate state
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not np.all(np.isfinite(state)):
            state = np.nan_to_num(state, 0)
        
        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Set networks to eval mode for inference
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(state)
            action = q_values.argmax().item()
        self.online_network.train()
        
        # Validate action
        action = max(0, min(action, self.action_size - 1))
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience tuple in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Validate inputs
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        
        # Handle NaN values
        state = np.nan_to_num(state, 0)
        next_state = np.nan_to_num(next_state, 0)
        
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Sample from experience replay buffer and train the network."""
        if len(self.memory) < self.batch_size:
            return
        
        # Get random minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Convert lists to numpy arrays first for better performance
        states_array = np.array([t[0] for t in minibatch])
        next_states_array = np.array([t[3] for t in minibatch])
        
        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states_array).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        
        # Get other batch elements
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)
        
        # Current Q values
        curr_q_values = self.online_network(states)
        curr_q = curr_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q values with target network for stability
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            expected_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss and update
        loss = self.criterion(curr_q, expected_q.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % self.update_target_every == 0:
            self.update_target_network()
        
        # Track loss
        self.running_loss = 0.95 * self.running_loss + 0.05 * loss.item()
        self.loss_history.append(loss.item())
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', loss.item(), self.train_steps)
        self.writer.add_scalar('Q_value/average', curr_q.mean().item(), self.train_steps)
    
    def step(self, state, is_eval=False):
        """Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            is_eval (bool): Whether in evaluation mode
        
        Returns:
            int: Selected action
        """
        if not is_eval and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Validate state
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not np.all(np.isfinite(state)):
            state = np.nan_to_num(state, 0)
        
        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Set networks to eval mode for inference
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(state)
            action = q_values.argmax().item()
        self.online_network.train()
        
        # Validate action
        action = max(0, min(action, self.action_size - 1))
        return action
    
    def evaluate(self, state):
        """Choose action using greedy policy.
        
        Args:
            state: Current state
        
        Returns:
            int: Selected action
        """
        # Validate state
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not np.all(np.isfinite(state)):
            state = np.nan_to_num(state, 0)
        
        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Set networks to eval mode for inference
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(state)
            action = q_values.argmax().item()
        self.online_network.train()
        
        # Validate action
        action = max(0, min(action, self.action_size - 1))
        return action

class StockTradingEnv:
    """Stock trading environment implementing gym-like interface."""
    
    def __init__(self, df):
        """Initialize trading environment.
        
        Args:
            df (dict): Dictionary of stock dataframes
        """
        self.data_dict = df  # Dictionary of DataFrames for each stock
        self.action_space = 9  # Expanded actions: Buy/Sell 100%, 75%, 50%, 25%, Hold
        self.action_size = self.action_space
        self.initial_balance = 3333.33
        self.state_space = 9  # [Close_scaled, Returns, Volume_trend, RSI_zone, MACD_signal, MA5_trend, MA20_trend, MA50_trend, Position_zone]
        
        # Verify features
        required_columns = ['Close_scaled', 'Returns', 'Volume_scaled', 'MA20_Volume', 'RSI_scaled', 'MACD_scaled', 'Signal_Line_scaled', 'MA5', 'MA20', 'MA50']
        for stock, data in df.items():
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing scaled columns in data for {stock}")
        
        self.transaction_fee = 0.001
        self.holding_penalty = 0.0001  # Reduced penalty
        self.min_trade_interval = 3  # Minimum steps between trades
        
        self.reset()
    
    def get_portfolio_value(self, stock):
        """Get current portfolio value for a stock.
        
        Args:
            stock (str): Stock symbol
            
        Returns:
            float: Current portfolio value (cash + stock position value)
        """
        if stock not in self.data_dict:
            raise ValueError(f"Stock {stock} not found in environment")
            
        current_price = self.data_dict[stock]['Close'].iloc[self.current_steps[stock]]
        stock_value = self.positions[stock] * current_price
        return self.balances[stock] + stock_value
    
    def reset(self):
        """Reset environment."""
        self.done = False  # Reset done flag
        self.current_steps = {stock: 0 for stock in self.data_dict.keys()}
        self.positions = {stock: 0 for stock in self.data_dict.keys()}
        self.balances = {stock: self.initial_balance for stock in self.data_dict.keys()}
        self.last_trade_step = {stock: 0 for stock in self.data_dict.keys()}
        self.max_values = {stock: self.initial_balance for stock in self.data_dict.keys()}
        
        # Return initial states for each stock
        states = {}
        for stock in self.data_dict.keys():
            if len(self.data_dict[stock]) > 0:  # Only get state if data exists
                try:
                    states[stock] = self._get_state(stock)
                except Exception as e:
                    print(f"Warning: Could not get initial state for {stock}: {e}")
                    states[stock] = None
            else:
                states[stock] = None
        
        return states
    
    def step(self, stock, action):
        """Take a step in the environment.
        
        Args:
            stock (str): Stock symbol
            action (int): Action to take
        
        Returns:
            tuple: Next state, reward, done
        """
        # Check if stock exists and has data
        if stock not in self.data_dict or len(self.data_dict[stock]) == 0:
            return None, 0, True
            
        # Check if this stock has reached its end
        if self.current_steps[stock] >= len(self.data_dict[stock]) - 1:
            return None, 0, True
        
        try:
            current_price = self.data_dict[stock].iloc[self.current_steps[stock]]['Close']
            current_value = self.balances[stock] + (self.positions[stock] * current_price)
            
            # Calculate position size based on action
            shares_to_trade = 0
            if action == 0:  # Buy 100%
                max_shares = (self.balances[stock] * 0.99) // current_price
                shares_to_trade = max_shares if max_shares > 0 else 0
            elif action == 1:  # Buy 75%
                max_shares = (self.balances[stock] * 0.74) // current_price
                shares_to_trade = max_shares if max_shares > 0 else 0
            elif action == 2:  # Buy 50%
                max_shares = (self.balances[stock] * 0.49) // current_price
                shares_to_trade = max_shares if max_shares > 0 else 0
            elif action == 3:  # Buy 25%
                max_shares = (self.balances[stock] * 0.24) // current_price
                shares_to_trade = max_shares if max_shares > 0 else 0
            elif action == 4:  # Hold
                shares_to_trade = 0
            elif action == 5:  # Sell 25%
                shares_to_trade = -int(self.positions[stock] * 0.25)
            elif action == 6:  # Sell 50%
                shares_to_trade = -int(self.positions[stock] * 0.50)
            elif action == 7:  # Sell 75%
                shares_to_trade = -int(self.positions[stock] * 0.75)
            else:  # Sell 100%
                shares_to_trade = -self.positions[stock]
            
            reward = 0
            trade_executed = False
            
            # Execute trade if valid
            if shares_to_trade > 0:  # Buying
                cost = shares_to_trade * current_price * (1 + self.transaction_fee)
                if cost <= self.balances[stock]:
                    self.positions[stock] += shares_to_trade
                    self.balances[stock] -= cost
                    trade_executed = True
                    
            elif shares_to_trade < 0:  # Selling
                if abs(shares_to_trade) <= self.positions[stock]:
                    proceeds = abs(shares_to_trade) * current_price * (1 - self.transaction_fee)
                    self.positions[stock] += shares_to_trade
                    self.balances[stock] += proceeds
                    trade_executed = True
            
            # Move to next step for this stock
            self.current_steps[stock] += 1
            
            # Calculate reward based on portfolio value change
            next_price = self.data_dict[stock].iloc[self.current_steps[stock]]['Close']
            new_value = self.balances[stock] + (self.positions[stock] * next_price)
            
            # Update max value seen for this stock
            self.max_values[stock] = max(self.max_values[stock], new_value)
            
            # Calculate reward components
            value_change = (new_value - current_value) / current_value
            drawdown = (self.max_values[stock] - new_value) / self.max_values[stock]
            position_size = self.positions[stock] * next_price / new_value if new_value > 0 else 0
            
            # Combine reward components
            reward = value_change
            if drawdown > 0.1:  # Penalize significant drawdowns
                reward -= drawdown * 0.5
            if position_size > 0.8:  # Penalize over-concentration
                reward -= 0.1 * (position_size - 0.8)
            if trade_executed:  # Small penalty for trading
                reward -= self.transaction_fee
            
            # Get next state
            try:
                next_state = self._get_state(stock)
            except Exception as e:
                print(f"Warning: Could not get next state for {stock}: {e}")
                next_state = None
                return next_state, reward, True
            
            # Check if this stock is done
            stock_done = self.current_steps[stock] >= len(self.data_dict[stock]) - 1
            
            return next_state, reward, stock_done
            
        except Exception as e:
            print(f"Error in step for {stock}: {e}")
            return None, 0, True
    
    def step_all(self, actions):
        """Take a step in the environment for all stocks.
        
        Args:
            actions (dict): Dictionary of actions for each stock
        
        Returns:
            tuple: Next states, rewards, done
        """
        next_states = {}
        rewards = {}
        stocks_done = {}
        active_stocks = 0
        total_stocks = len(self.data_dict)
        
        for stock in self.data_dict.keys():
            if stock in actions:
                next_state, reward, done = self.step(stock, actions[stock])
                next_states[stock] = next_state
                rewards[stock] = reward
                stocks_done[stock] = done
                
                if not done:
                    active_stocks += 1
        
        # Environment is done when all stocks are done
        self.done = active_stocks == 0
        
        return next_states, rewards, self.done
    
    def _get_state(self, stock):
        """Get current state for a stock.
        
        Args:
            stock (str): Stock symbol
        
        Returns:
            ndarray: State vector
        """
        data = self.data_dict[stock]
        current_idx = self.current_steps[stock]
        
        # Price features
        close = data['Close_scaled'].iloc[current_idx]
        returns = data['Returns'].iloc[current_idx]
        
        # Volume
        volume = data['Volume_scaled'].iloc[current_idx]
        vol_ma = data['MA20_Volume'].iloc[current_idx] / data['Volume'].iloc[current_idx]
        
        # Technical Indicators
        rsi = data['RSI_scaled'].iloc[current_idx]
        macd = data['MACD_scaled'].iloc[current_idx]
        signal = data['Signal_Line_scaled'].iloc[current_idx]
        
        # Moving Average Crossovers (discretized into trends)
        ma5 = data['MA5'].iloc[current_idx] / data['Close'].iloc[current_idx]
        ma20 = data['MA20'].iloc[current_idx] / data['Close'].iloc[current_idx]
        ma50 = data['MA50'].iloc[current_idx] / data['Close'].iloc[current_idx]
        
        # Discretize MA trends into -1 (downtrend), 0 (sideways), 1 (uptrend)
        ma5_trend = 1 if ma5 > 1.01 else (-1 if ma5 < 0.99 else 0)
        ma20_trend = 1 if ma20 > 1.01 else (-1 if ma20 < 0.99 else 0)
        ma50_trend = 1 if ma50 > 1.01 else (-1 if ma50 < 0.99 else 0)
        
        # RSI zones: oversold (0), neutral (1), overbought (2)
        rsi_zone = 0 if rsi < 0.3 else (2 if rsi > 0.7 else 1)
        
        # MACD signal: -1 (bearish), 0 (neutral), 1 (bullish)
        macd_signal = 1 if macd > signal else (-1 if macd < signal else 0)
        
        # Volume trend: -1 (decreasing), 0 (stable), 1 (increasing)
        vol_trend = 1 if vol_ma > 1.1 else (-1 if vol_ma < 0.9 else 0)
        
        # Portfolio state (discretized position ratio)
        position = self.positions[stock] * data['Close'].iloc[current_idx]
        total_value = self.balances[stock] + position
        position_ratio = position / total_value if total_value > 0 else 0
        position_zone = int(position_ratio * 4)  # Discretize into 0-3 zones
        
        return np.array([
            close,          # Current price (scaled)
            returns,        # Recent returns
            vol_trend,      # Volume trend (-1, 0, 1)
            rsi_zone,       # RSI zone (0, 1, 2)
            macd_signal,    # MACD signal (-1, 0, 1)
            ma5_trend,      # MA5 trend (-1, 0, 1)
            ma20_trend,     # MA20 trend (-1, 0, 1)
            ma50_trend,     # MA50 trend (-1, 0, 1)
            position_zone   # Position zone (0-3)
        ], dtype=np.float32)

def train_agents(env, episodes=200):
    """Train DQN agents on environment.
    
    Args:
        env: Trading environment
        episodes (int): Number of training episodes
    
    Returns:
        tuple: Dictionary of trained agents and training history
    """
    print("\nInitializing DQN Agents...")
    agents = {}
    training_history = {}
    
    # Initialize agents and history for each stock
    for stock in env.data_dict.keys():
        agents[stock] = DQNAgent(env.state_space, env.action_space)
        training_history[stock] = {
            'net_worth': [],
            'positions': [],
            'best_value': float('-inf'),
            'patience_counter': 0,
            'train_start': None,
            'train_end': None,
            'train_duration': None
        }
        
        # Calculate and store training period for each stock
        train_data = env.data_dict[stock]
        training_history[stock]['train_start'] = train_data.index[0]
        training_history[stock]['train_end'] = train_data.index[-1]
        training_history[stock]['train_duration'] = (train_data.index[-1] - train_data.index[0]).days
        
        print(f"\n{stock} Training Period:")
        print(f"Start: {training_history[stock]['train_start'].strftime('%Y-%m-%d')}")
        print(f"End: {training_history[stock]['train_end'].strftime('%Y-%m-%d')}")
        print(f"Duration: {training_history[stock]['train_duration']} days")
    
    print("\nStarting Training...")
    for episode in range(episodes):
        states = env.reset()
        total_rewards = {stock: 0 for stock in env.data_dict.keys()}
        active_stocks = set(env.data_dict.keys())
        
        while not env.done and active_stocks:
            actions = {}
            
            # Get actions for each stock
            for stock in active_stocks.copy():
                if states[stock] is None:
                    active_stocks.remove(stock)
                    continue
                    
                action = agents[stock].act(states[stock])
                actions[stock] = action
            
            # Take step in environment
            next_states, rewards, done = env.step_all(actions)
            
            # Learn from experiences
            for stock in active_stocks.copy():
                if states[stock] is not None and next_states[stock] is not None:
                    # Store experience and learn
                    agents[stock].remember(states[stock], actions[stock], rewards[stock], next_states[stock], done)
                    if len(agents[stock].memory) > agents[stock].batch_size:
                        agents[stock].replay()
                    
                    total_rewards[stock] += rewards[stock]
                    
                    # Record training history
                    portfolio_value = env.get_portfolio_value(stock)
                    training_history[stock]['net_worth'].append(portfolio_value)
                    training_history[stock]['positions'].append(actions[stock])
                else:
                    active_stocks.remove(stock)
            
            states = next_states
            
            # Update target networks periodically
            for stock in active_stocks:
                if agents[stock].train_steps % agents[stock].update_target_every == 0:
                    agents[stock].update_target_network()
        
        # Early stopping check and epsilon decay for each agent
        all_converged = True
        for stock in env.data_dict.keys():
            if len(training_history[stock]['net_worth']) > 0:  # Only update if we have data for this stock
                current_value = training_history[stock]['net_worth'][-1]
                
                # Update best value and patience counter
                if current_value > training_history[stock]['best_value']:
                    training_history[stock]['best_value'] = current_value
                    training_history[stock]['patience_counter'] = 0
                else:
                    training_history[stock]['patience_counter'] += 1
                
                # Decay epsilon
                agents[stock].decay_epsilon()
                
                # Check if this agent has converged
                if training_history[stock]['patience_counter'] < 20 or agents[stock].epsilon > 0.1:
                    all_converged = False
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{episodes}")
            for stock in env.data_dict.keys():
                if len(training_history[stock]['net_worth']) > 0:  # Only print if we have data
                    portfolio_value = training_history[stock]['net_worth'][-1]
                    pct_change = ((portfolio_value - env.initial_balance) / env.initial_balance) * 100
                    print(f"{stock} - Portfolio: ${portfolio_value:.2f} ({pct_change:+.2f}%), "
                          f"Epsilon: {agents[stock].epsilon:.3f}, "
                          f"Steps: {len(training_history[stock]['net_worth'])}")
        
        # Early stopping if all agents have converged
        if all_converged and episode >= 50:  # Ensure minimum training episodes
            print("\nEarly stopping triggered for all agents!")
            break
    
    # Print final training summary
    print("\nTraining Summary:")
    for stock in env.data_dict.keys():
        print(f"\n{stock}:")
        print(f"Training Period: {training_history[stock]['train_start'].strftime('%Y-%m-%d')} to "
              f"{training_history[stock]['train_end'].strftime('%Y-%m-%d')} "
              f"({training_history[stock]['train_duration']} days)")
        if len(training_history[stock]['net_worth']) > 0:
            final_value = training_history[stock]['net_worth'][-1]
            pct_change = ((final_value - env.initial_balance) / env.initial_balance) * 100
            print(f"Final Portfolio Value: ${final_value:.2f} ({pct_change:+.2f}%)")
            print(f"Best Portfolio Value: ${training_history[stock]['best_value']:.2f}")
            print(f"Total Training Steps: {len(training_history[stock]['net_worth'])}")
    
    return agents, training_history

def evaluate_agents(env, agents):
    """Evaluate the performance of the trained agents in the environment without exploration.
    
    Args:
        env: Trading environment
        agents (dict): Dictionary of trained DQN agents
        
    Returns:
        tuple: Portfolio values and actions taken for each stock
    """
    portfolio_values = {stock: [] for stock in env.data_dict.keys()}
    actions_taken = {stock: [] for stock in env.data_dict.keys()}
    
    # Save original epsilon and set to 0 for evaluation
    original_epsilon = {stock: agent.epsilon for stock, agent in agents.items()}
    for agent in agents.values():
        agent.epsilon = 0
    
    # Reset environment
    states = env.reset()
    done = False
    
    while not done:
        # Get actions for each stock
        actions = {}
        for stock in env.data_dict.keys():
            state = states[stock]
            action = agents[stock].evaluate(state)  # Use evaluate instead of act for deterministic actions
            actions[stock] = action
            actions_taken[stock].append(action)
        
        # Take step in environment
        next_states, rewards, done = env.step_all(actions)
        
        # Store portfolio values
        for stock in env.data_dict.keys():
            portfolio_values[stock].append(env.get_portfolio_value(stock))
        
        states = next_states
    
    # Restore original epsilon
    for stock, agent in agents.items():
        agent.epsilon = original_epsilon[stock]
    
    return portfolio_values, actions_taken

def plot_trade_histories(trade_histories, data):
    """Plot the trade histories for each stock, including portfolio value, stock price, and trading actions.
    
    Args:
        trade_histories (dict): Dictionary of trade histories for each stock
        data (dict): Dictionary of stock dataframes
    """
    for stock, trades in trade_histories.items():
        net_worths = trades["net_worth"]
        positions = trades["positions"]
        steps = list(range(len(net_worths)))
        
        # Use scaled prices for consistency
        stock_prices_full = data[stock]['Close_scaled'].values * data[stock]['Close'].mean()  # Denormalize for interpretability
        stock_prices = stock_prices_full[:len(net_worths)]
        
        # If stock_prices are shorter, pad with the last available price
        if len(stock_prices) < len(net_worths):
            pad_length = len(net_worths) - len(stock_prices)
            stock_prices = np.concatenate([stock_prices, np.full(pad_length, stock_prices[-1])])
        
        plt.figure(figsize=(15, 10))

        # Plot portfolio net worth
        plt.subplot(2, 1, 1)
        plt.plot(steps, net_worths, label='Net Worth', color="blue", linewidth=1.5)
        plt.title(f'{stock} Portfolio Value Over Time')
        plt.xlabel('Trading Steps')
        plt.ylabel('Net Worth ($)')
        plt.legend()
        plt.grid(True)

        # Plot stock price and actions
        plt.subplot(2, 1, 2)
        plt.plot(stock_prices, label='Stock Price ($)', color="orange", linewidth=2)

        # Only plot markers every N steps to reduce clutter
        marker_interval = 100  # Increased interval to reduce markers further
        
        # Overlay buy and sell actions with reduced frequency
        buy_steps = [i for i, action in enumerate(positions) if action in [0, 1, 2, 3]]  # Buy actions
        sell_steps = [i for i, action in enumerate(positions) if action in [5, 6, 7, 8]]  # Sell actions (excluding Hold)
        
        # Filter markers to reduce density
        buy_steps = buy_steps[::marker_interval]
        sell_steps = sell_steps[::marker_interval]
        
        if buy_steps:
            plt.scatter(buy_steps, stock_prices[buy_steps], color="green", marker="^", 
                       label="Buy", s=100, alpha=0.7)
        if sell_steps:
            plt.scatter(sell_steps, stock_prices[sell_steps], color="red", marker="v", 
                       label="Sell", s=100, alpha=0.7)

        plt.title(f'{stock} Stock Price and Trading Actions')
        plt.xlabel('Trading Steps')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    from load_data import load_data
    from split import split_data_individual
    
    # Load data
    qqq, spy, voo = load_data()
    
    # Split data using individual full durations
    split_data = split_data_individual(qqq, spy, voo, test_size=0.2)
    
    # Create training data dictionary
    train_data = {
        'QQQ': split_data['QQQ']['train'],
        'SPY': split_data['SPY']['train'],
        'VOO': split_data['VOO']['train']
    }
    
    # Create test data dictionary with individual periods
    test_data = {
        'QQQ': split_data['QQQ']['test'],
        'SPY': split_data['SPY']['test'],
        'VOO': split_data['VOO']['test']
    }
    
    # Print training and test periods for each stock
    for stock in ['QQQ', 'SPY', 'VOO']:
        print(f"\n{stock} Periods:")
        print(f"Training: {train_data[stock].index[0]} to {train_data[stock].index[-1]}")
        print(f"Testing:  {test_data[stock].index[0]} to {test_data[stock].index[-1]}")
    
    # Create environments
    env = StockTradingEnv(train_data)
    test_env = StockTradingEnv(test_data)
    
    # Train agents
    print("\nTraining agents...")
    agents, training_history = train_agents(env, episodes=200)
    
    # Evaluate agents
    print("\nEvaluating agents...")
    portfolio_values, actions_taken = evaluate_agents(test_env, agents)
    
    # Plot results similar to qlearn.py
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 8))
    for stock in env.data_dict.keys():
        dates = test_data[stock].index
        values = portfolio_values[stock]
        plt.plot(dates, values, label=f'{stock} Portfolio Value')
    
    plt.title('Portfolio Values During Test Period')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
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
