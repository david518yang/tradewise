"""Q-Learning implementation for stock trading."""

import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

class QLearningAgent:
    """Q-learning agent for stock trading."""
    
    def __init__(self, state_space, action_space, learning_rate=0.001, 
                 discount_factor=0.95, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.995, n_states=10):
        """Initialize Q-learning agent."""
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_states = n_states
        
        # Initialize Q-table
        self.q_table = {}
    
    def discretize_state(self, state):
        """Convert continuous state to discrete state."""
        if isinstance(state, dict):
            raise ValueError("State cannot be a dictionary. Expected a numeric array.")
            
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        elif not isinstance(state, np.ndarray):
            state = np.array([state], dtype=np.float32)
        
        # Ensure state is 1D
        state = state.ravel()
        
        # Discretize each feature into n_states bins
        discrete = []
        for val in state:
            try:
                if np.isnan(val) or not np.isfinite(val):
                    discrete.append(0)  # Handle NaN and infinite values
                else:
                    # Scale the value between 0 and n_states-1
                    scaled_val = max(0, min(val, 1))  # Clip between 0 and 1
                    discrete.append(int(scaled_val * (self.n_states - 1)))
            except (TypeError, ValueError):
                discrete.append(0)
        
        return tuple(discrete)  # Return tuple for hashability
    
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        discrete_state = self.discretize_state(state)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        
        # Get Q-values for this state
        q_values = self.get_q_values(discrete_state)
        return np.argmax(q_values)
    
    def get_q_values(self, discrete_state):
        """Get Q-values for a discrete state."""
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space)
        return self.q_table[discrete_state]
    
    def learn(self, state, action, reward, next_state):
        """Update Q-value for state-action pair."""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        current_q = self.get_q_values(discrete_state)[action]
        next_max_q = np.max(self.get_q_values(discrete_next_state))
        
        # Q-learning update rule
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[discrete_state][action] = new_q

class StockTradingEnv:
    """Stock trading environment implementing gym-like interface."""
    
    def __init__(self, df, initial_balance=3333.33):
        """Initialize the trading environment.
        
        Args:
            df (dict): Dictionary of dataframes for each stock
            initial_balance (float): Initial cash balance
        """
        self.df = df
        self.initial_balance = initial_balance
        self.stocks = list(df.keys())
        
        # Define action and state space
        self.action_space = 3  # hold (0), buy (1), sell (2)
        
        # State space: [normalized_close, returns, volume, rsi, macd, ma_ratios, position]
        self.state_space = 9
        
        # Initialize agent data
        self.reset()
    
    def _get_state(self, stock):
        """Get current state for a stock.
        
        Args:
            stock (str): Stock symbol
        
        Returns:
            np.array: Current state vector
        """
        current_step = self.agent_data[stock]['current_step']
        df = self.df[stock]
        
        try:
            # Get current price and normalize it
            current_close = df['Close'].iloc[current_step]
            avg_price = df['Close'].iloc[max(0, current_step-10):current_step+1].mean()
            normalized_close = (current_close - avg_price) / avg_price if avg_price != 0 else 0
            
            # Get other features
            returns = df['Returns'].iloc[current_step]
            
            # Volume relative to its moving average
            volume = df['Volume'].iloc[current_step]
            ma20_volume = df['MA20_Volume'].iloc[current_step]
            normalized_volume = volume / ma20_volume if ma20_volume != 0 else 1
            
            # Technical indicators
            rsi = df['RSI'].iloc[current_step] / 100  # Normalize RSI to [0,1]
            macd = df['MACD'].iloc[current_step]
            signal = df['Signal_Line'].iloc[current_step]
            macd_signal_ratio = (macd - signal) / abs(signal) if abs(signal) > 0 else 0
            
            # Moving average ratios
            ma5 = df['MA5'].iloc[current_step]
            ma20 = df['MA20'].iloc[current_step]
            ma50 = df['MA50'].iloc[current_step]
            
            ma5_ratio = (current_close / ma5 - 1) if ma5 != 0 else 0
            ma20_ratio = (current_close / ma20 - 1) if ma20 != 0 else 0
            ma50_ratio = (current_close / ma50 - 1) if ma50 != 0 else 0
            
            # Current position
            position = self.agent_data[stock]['shares_held'] / self.initial_balance
            
            state = np.array([
                normalized_close,    # Price relative to recent average
                returns,            # Returns
                normalized_volume,  # Volume relative to MA
                rsi,               # RSI (normalized)
                macd_signal_ratio, # MACD/Signal ratio
                ma5_ratio,         # Price relative to MA5
                ma20_ratio,        # Price relative to MA20
                ma50_ratio,        # Price relative to MA50
                position,          # Current position
            ], dtype=np.float32)
            
            # Replace any remaining NaN or infinite values
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state
            
        except (KeyError, IndexError) as e:
            print(f"Error getting state for {stock} at step {current_step}: {str(e)}")
            # Return a zero state vector if there's an error
            return np.zeros(self.state_space, dtype=np.float32)
    
    def reset(self):
        """Reset environment."""
        self.done = False
        self.agent_data = {}
        
        # Initialize agent data for each stock
        for stock in self.stocks:
            self.agent_data[stock] = {
                'balance': self.initial_balance,
                'shares_held': 0,
                'current_step': 0,
                'max_value': self.initial_balance
            }
            
            # Verify data length
            if len(self.df[stock]) == 0:
                raise ValueError(f"Empty DataFrame for stock {stock}")
        
        return {stock: self._get_state(stock) for stock in self.stocks}

    def step(self, stock, action):
        """Take a step in the environment.
        
        Args:
            stock (str): Stock symbol
            action (int): Action to take (0: hold, 1: buy, 2: sell)
            
        Returns:
            tuple: (new_state, reward, done)
        """
        agent_data = self.agent_data[stock]
        df = self.df[stock]
        
        # Get current price and move to next step
        current_idx = agent_data['current_step']
        current_price = df['Close'].iloc[current_idx]
        
        # Calculate previous portfolio value
        prev_value = self.get_portfolio_value(stock)
        
        # Execute action
        if action == 1:  # Buy
            if agent_data['balance'] >= current_price:  # Check if we can buy
                shares_to_buy = agent_data['balance'] // current_price
                agent_data['shares_held'] += shares_to_buy
                agent_data['balance'] -= shares_to_buy * current_price
        elif action == 2:  # Sell
            if agent_data['shares_held'] > 0:  # Check if we can sell
                agent_data['balance'] += agent_data['shares_held'] * current_price
                agent_data['shares_held'] = 0
        
        # Move to next step
        agent_data['current_step'] += 1
        
        # Check if we've reached the end of this stock's data
        if agent_data['current_step'] >= len(df) - 1:
            agent_data['current_step'] = len(df) - 1  # Prevent going beyond bounds
            self.done = True
            
        # Calculate new portfolio value
        new_value = self.get_portfolio_value(stock)
        
        # Update max value seen
        agent_data['max_value'] = max(agent_data['max_value'], new_value)
        
        # Calculate returns-based reward
        returns = (new_value - prev_value) / prev_value if prev_value > 0 else 0
        
        # Base reward on returns
        reward = returns
        
        # Add position-based rewards/penalties
        position_size = agent_data['shares_held'] * current_price / new_value if new_value > 0 else 0
        
        # Penalize very large positions (over 80% of portfolio)
        if position_size > 0.8:
            reward -= 0.1 * (position_size - 0.8)
        
        # Penalize frequent trading
        if action != 0:  # If not holding
            reward -= 0.001  # Small transaction cost
        
        # Penalize drawdowns
        drawdown = (agent_data['max_value'] - new_value) / agent_data['max_value'] if agent_data['max_value'] > 0 else 0
        if drawdown > 0.1:  # Only penalize significant drawdowns
            reward -= drawdown * 0.5  # Scale down the drawdown penalty
        
        # Scale reward to be in a reasonable range
        reward = np.clip(reward, -1, 1)
        
        # Check if done
        all_stocks_done = all(
            self.agent_data[s]['current_step'] >= len(self.df[s]) - 1 
            for s in self.stocks
        )
        if all_stocks_done:
            self.done = True
            
            # Add terminal reward based on overall performance
            final_return = (new_value - self.initial_balance) / self.initial_balance
            if final_return > 0:
                reward += 0.5  # Bonus for positive overall return
            
        return self._get_state(stock), reward, self.done

    def get_portfolio_value(self, stock):
        """Get current portfolio value for a stock."""
        agent_data = self.agent_data[stock]
        df = self.df[stock]
        current_idx = agent_data['current_step']
        current_price = df['Close'].iloc[current_idx]
        return agent_data['balance'] + (agent_data['shares_held'] * current_price)

def train_agents(env, episodes=500, early_stopping_patience=100, min_epsilon_for_stopping=0.2):
    """Train Q-learning agents for each stock."""
    agents = {}
    history = {}
    
    # Initialize agents for each stock
    for stock in env.stocks:
        agents[stock] = QLearningAgent(
            state_space=env.state_space,
            action_space=env.action_space,
            learning_rate=0.001,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            n_states=10
        )
        history[stock] = {
            'portfolio_values': [],
            'rewards': [],
            'best_value': float('-inf'),
            'patience_counter': 0,
            'initial_value': env.initial_balance
        }
    
    for episode in range(episodes):
        states = env.reset()  # Get initial states for all stocks
        total_rewards = {stock: 0 for stock in env.stocks}
        done = False
        
        while not done:
            actions = {}
            new_states = {}
            rewards = {}
            dones = {}
            
            # Take actions for each stock
            for stock in env.stocks:
                state = states[stock]
                action = agents[stock].act(state)
                
                # Take step in environment
                new_state, reward, done = env.step(stock, action)
                
                # Store results
                actions[stock] = action
                new_states[stock] = new_state
                rewards[stock] = reward
                dones[stock] = done
                
                # Update Q-table
                agents[stock].learn(state, action, reward, new_state)
                
                # Update total rewards
                total_rewards[stock] += reward
            
            # Update states
            states = new_states
            done = all(dones.values())
        
        # After episode ends, update history
        for stock in env.stocks:
            current_value = env.get_portfolio_value(stock)
            history[stock]['portfolio_values'].append(current_value)
            history[stock]['rewards'].append(total_rewards[stock])
            
            # Early stopping check
            if current_value > history[stock]['best_value']:
                history[stock]['best_value'] = current_value
                history[stock]['patience_counter'] = 0
            else:
                history[stock]['patience_counter'] += 1
            
            # Decay epsilon once per episode
            agents[stock].epsilon = max(
                agents[stock].epsilon_min,
                agents[stock].epsilon * agents[stock].epsilon_decay
            )
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"\nEpisode {episode + 1}/{episodes}")
            for stock in env.stocks:
                current_value = history[stock]['portfolio_values'][-1]
                initial_value = history[stock]['initial_value']
                pct_change = ((current_value - initial_value) / initial_value) * 100
                print(f"{stock} - Portfolio: ${current_value:.2f} ({pct_change:+.2f}%), "
                      f"Epsilon: {agents[stock].epsilon:.3f}")
        
        # Check if all agents should stop early
        if all(agents[stock].epsilon <= min_epsilon_for_stopping for stock in env.stocks) and \
           all(history[stock]['patience_counter'] >= early_stopping_patience for stock in env.stocks):
            print("\nEarly stopping triggered for all agents")
            print(f"Training stopped with epsilon values: " + 
                  ", ".join([f"{stock}: {agents[stock].epsilon:.3f}" for stock in env.stocks]))
            break
    
    return agents, history

def evaluate_agents(env, agents):
    """Evaluate trained agents on test data."""
    results = {}
    
    for stock in env.stocks:
        print(f"\nEvaluating {stock} agent...")
        
        # Create evaluation environment for this stock
        states = env.reset()
        state = states[stock]  # Get state for current stock
        
        portfolio_values = []
        dates = []
        actions_taken = []
        done = False
        
        while not done:
            # Get action from agent (no exploration during evaluation)
            action = agents[stock].act(state)
            
            # Take action in environment
            new_state, reward, done = env.step(stock, action)
            
            # Record portfolio value and action
            portfolio_value = env.get_portfolio_value(stock)
            portfolio_values.append(portfolio_value)
            actions_taken.append(action)
            dates.append(env.df[stock].index[env.agent_data[stock]['current_step']])
            
            # Update state
            state = new_state
        
        results[stock] = {
            'dates': dates,
            'portfolio_values': portfolio_values,
            'actions': actions_taken
        }
        
        # Print evaluation results
        initial_value = env.initial_balance
        final_value = portfolio_values[-1]
        total_return = ((final_value - initial_value) / initial_value) * 100
        print(f"{stock} Final Portfolio Value: ${final_value:.2f} ({total_return:+.2f}%)")
    
    return results

if __name__ == "__main__":
    import pandas as pd
    from split import split_data_individual
    
    print("\nLoading data...")
    # Load and preprocess data
    qqq = pd.read_csv('../data/qqq.csv', index_col='Date', parse_dates=True)
    spy = pd.read_csv('../data/spy.csv', index_col='Date', parse_dates=True)
    voo = pd.read_csv('../data/voo.csv', index_col='Date', parse_dates=True)
    
    # Split data into train and test sets
    split_data = split_data_individual(qqq, spy, voo)
    
    # Extract train and test data
    train_data = {
        'QQQ': split_data['QQQ']['train'],
        'SPY': split_data['SPY']['train'],
        'VOO': split_data['VOO']['train']
    }
    
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
    
    print("\nInitializing training environment...")
    # Create training environment
    env = StockTradingEnv(train_data)
    
    print("\nTraining agents...")
    # Train agents
    agents, history = train_agents(env, episodes=500, early_stopping_patience=100, min_epsilon_for_stopping=0.2)
    
    print("\nEvaluating agents on test data...")
    # Create test environment
    test_env = StockTradingEnv(test_data)
    
    # Evaluate agents
    results = evaluate_agents(test_env, agents)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 8))
    for stock in env.stocks:
        plt.plot(results[stock]['dates'], 
                results[stock]['portfolio_values'], 
                label=f'{stock} Portfolio Value')
    
    plt.title('Portfolio Values During Test Period')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()