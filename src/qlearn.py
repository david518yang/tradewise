"""Q-Learning implementation for stock trading."""

import numpy as np
import pandas as pd

class QLearningAgent:
    """Q-Learning agent for stock trading."""
    
    def __init__(self, state_space, action_space, learning_rate=0.1, gamma=0.99, epsilon_start=1.0):
        """Initialize Q-Learning Agent.
        
        Args:
            state_space (int): Size of state space
            action_space (int): Size of action space
            learning_rate (float): Learning rate for Q-value updates
            gamma (float): Discount factor for future rewards
            epsilon_start (float): Initial exploration rate
        """
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = {}
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.best_reward = float('-inf')
        self.best_q_table = None

    def get_action(self, state, is_training=True):
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            is_training (bool): Whether to use exploration
        
        Returns:
            int: Selected action
        """
        state_key = tuple(state.round(4))
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
        
        if is_training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        
        return np.argmax(self.q_table[state_key])

    def update(self, state, action, reward, next_state):
        """Update Q-values using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_key = tuple(state.round(4))
        next_state_key = tuple(next_state.round(4))
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space)
        
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action] = new_q

    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_if_better(self, episode_reward):
        """Save Q-table if performance improves.
        
        Args:
            episode_reward (float): Total reward for the episode
        """
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.best_q_table = self.q_table.copy()

    def load_best(self):
        """Load the best performing Q-table."""
        if self.best_q_table is not None:
            self.q_table = self.best_q_table.copy()

    def reset(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.epsilon_start

class StockTradingEnv:
    """Stock trading environment implementing gym-like interface."""
    
    def __init__(self, df):
        """Initialize trading environment.
        
        Args:
            df (dict): Dictionary of stock dataframes
        """
        self.df = df
        self.position_sizes = np.array([-1.0, -0.5, 0, 0.5, 1.0])
        self.action_space = len(self.position_sizes)
        self.state_space = 9 + 5 + (len(df) - 1)
        self.initial_balance = 10000
        self.transaction_fee = 0.001
        self.min_trade_pct = 0.2
        
        self.agent_data = {
            key: {
                'balance': self.initial_balance / len(df),
                'shares_held': 0,
                'current_step': 0,
                'entry_price': None,
                'last_action_step': 0,
                'last_position': 0,
                'max_value': self.initial_balance / len(df)
            } for key in df.keys()
        }
        self.done = False
        self.reset()

    def reset(self):
        """Reset environment."""
        for key, agent_data in self.agent_data.items():
            agent_data['balance'] = self.initial_balance / len(self.df)
            agent_data['shares_held'] = 0
            agent_data['current_step'] = 0
            agent_data['entry_price'] = None
            agent_data['last_action_step'] = 0
            agent_data['last_position'] = 0
            agent_data['max_value'] = self.initial_balance / len(self.df)
        self.done = False
        return {key: self._get_state(key) for key in self.df.keys()}

    def _calculate_technical_indicators(self, data, current_step):
        """Calculate technical indicators.
        
        Args:
            data: Stock data
            current_step: Current step
        
        Returns:
            tuple: RSI, MACD, MACD histogram
        """
        try:
            delta = data['Close'].diff()
            up_days = delta.copy()
            down_days = delta.copy()
            up_days[delta <= 0] = 0
            down_days[delta >= 0] = 0
            
            RS_up = up_days.ewm(com=13, min_periods=1, adjust=False).mean()
            RS_down = abs(down_days).ewm(com=13, min_periods=1, adjust=False).mean()
            
            RS = RS_up / RS_down
            RSI = 100.0 - (100.0 / (1.0 + RS))
            
            if 0 <= current_step < len(RSI):
                current_rsi = RSI.iloc[current_step]
                if np.isnan(current_rsi):
                    current_rsi = 50.0
            else:
                current_rsi = 50.0

            exp1 = data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
            macd_hist = macd - signal
            
            if 0 <= current_step < len(macd):
                current_macd = macd.iloc[current_step]
                current_macd_hist = macd_hist.iloc[current_step]
                if np.isnan(current_macd):
                    current_macd = 0.0
                if np.isnan(current_macd_hist):
                    current_macd_hist = 0.0
            else:
                current_macd = 0.0
                current_macd_hist = 0.0

            return current_rsi, current_macd, current_macd_hist
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return 50.0, 0.0, 0.0  

    def _calculate_etf_features(self, data, current_step):
        """Calculate ETF features.
        
        Args:
            data: Stock data
            current_step: Current step
        
        Returns:
            np.array: ETF features
        """
        try:
            if current_step >= len(data):
                return np.zeros(5)  
            
            volume = data['Volume'].iloc[current_step]
            start_idx = max(0, current_step-20)
            avg_volume_20d = data['Volume'].iloc[start_idx:current_step+1].mean()
            relative_volume = volume / avg_volume_20d if avg_volume_20d > 0 else 1.0

            returns_20d = 0.0
            volatility_20d = 0.0
            if current_step >= 20:
                returns_20d = (data['Close'].iloc[current_step] / data['Close'].iloc[current_step-20]) - 1
                volatility_20d = data['Close'].pct_change().iloc[current_step-20:current_step+1].std()
            
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            money_flow = typical_price * data['Volume']
            
            delta_money_flow = money_flow.diff()
            pos_flow = delta_money_flow.copy()
            neg_flow = delta_money_flow.copy()
            pos_flow[delta_money_flow <= 0] = 0
            neg_flow[delta_money_flow >= 0] = 0
            
            pos_sum = pos_flow.rolling(window=14, min_periods=1).sum()
            neg_sum = abs(neg_flow.rolling(window=14, min_periods=1).sum())
            
            mfi = 100 - (100 / (1 + (pos_sum / neg_sum)))
            current_mfi = mfi.iloc[current_step] if not np.isnan(mfi.iloc[current_step]) else 50.0

            rolling_mean = data['Close'].rolling(window=20, min_periods=1).mean()
            rolling_std = data['Close'].rolling(window=20, min_periods=1).std()
            current_price = data['Close'].iloc[current_step]
            
            bb_position = 0.0
            if not np.isnan(rolling_mean.iloc[current_step]) and not np.isnan(rolling_std.iloc[current_step]):
                current_mean = rolling_mean.iloc[current_step]
                current_std = rolling_std.iloc[current_step]
                if current_std != 0:
                    bb_position = (current_price - current_mean) / (2 * current_std)

            return np.array([
                float(relative_volume - 1),  
                float(returns_20d),          
                float(volatility_20d),       
                float((current_mfi - 50) / 50),  
                float(bb_position),          
            ])
            
        except Exception as e:
            print(f"Error calculating ETF features: {e}")
            return np.zeros(5)  

    def _calculate_correlation_features(self, stock):
        """Calculate correlation features.
        
        Args:
            stock: Stock symbol
        
        Returns:
            np.array: Correlation features
        """
        current_step = self.agent_data[stock]['current_step']
        lookback = 20  
        
        if current_step < lookback:
            return np.zeros(len(self.df) - 1)  
        
        correlations = []
        current_returns = self.df[stock]['Close'].pct_change().iloc[current_step-lookback:current_step]
        
        for other_stock in self.df.keys():
            if other_stock != stock:
                other_returns = self.df[other_stock]['Close'].pct_change().iloc[current_step-lookback:current_step]
                corr = current_returns.corr(other_returns)
                correlations.append(float(corr) if not np.isnan(corr) else 0.0)
        
        return np.array(correlations)

    def _get_state(self, stock):
        """Get state for the given stock.
        
        Args:
            stock: Stock symbol
        
        Returns:
            np.array: State
        """
        data = self.df[stock]
        agent_data = self.agent_data[stock]
        current_step = agent_data['current_step']
        
        returns = 0.0
        momentum_5 = 0.0
        momentum_10 = 0.0
        
        if current_step >= 10:
            current_price = data['Close'].iloc[current_step]
            prev_price = data['Close'].iloc[current_step - 1]
            price_5_ago = data['Close'].iloc[current_step - 5]
            price_10_ago = data['Close'].iloc[current_step - 10]
            
            returns = (current_price - prev_price) / prev_price
            momentum_5 = (current_price - price_5_ago) / price_5_ago
            momentum_10 = (current_price - price_10_ago) / price_10_ago
        
        rsi, macd, macd_hist = self._calculate_technical_indicators(data, current_step)
        
        etf_features = self._calculate_etf_features(data, current_step)
        
        correlation_features = self._calculate_correlation_features(stock)
        
        position_size = 0.0
        position_pnl = 0.0
        if agent_data['shares_held'] != 0:
            current_price = data['Close'].iloc[current_step]
            position_value = agent_data['shares_held'] * current_price
            total_value = position_value + agent_data['balance']
            position_size = position_value / total_value
            
            if agent_data['entry_price'] is not None:
                position_pnl = (current_price - agent_data['entry_price']) / agent_data['entry_price']
        
        normalized_rsi = (rsi - 50) / 50
        
        market_features = np.concatenate([
            [
                float(returns),              
                float(momentum_5),           
                float(momentum_10),          
                float(position_size),        
                float(position_pnl),         
                float(agent_data['balance'] / self.initial_balance),  
                float(normalized_rsi),       
                float(macd),                
                float(macd_hist),           
            ],
            etf_features,                   
            correlation_features            
        ])
        
        market_features = np.nan_to_num(market_features, nan=0.0)
        
        return market_features

    def step(self, stock, action):
        """Take a step in the environment.
        
        Args:
            stock: Stock symbol
            action: Action to take
        
        Returns:
            tuple: Next state, reward, done
        """
        agent_data = self.agent_data[stock]
        data = self.df[stock]

        if agent_data['current_step'] >= len(data) - 1:
            self.done = True
            return None, 0, self.done

        previous_value = agent_data['balance'] + (agent_data['shares_held'] * data['Close'].iloc[agent_data['current_step']])
        agent_data['current_step'] += 1
        agent_data['current_price'] = data['Close'].iloc[agent_data['current_step']]

        current_stock_value = agent_data['shares_held'] * agent_data['current_price']
        target_percentage = self.position_sizes[action]
        total_value = agent_data['balance'] + current_stock_value
        
        target_stock_value = total_value * target_percentage if target_percentage > 0 else 0
        
        shares_to_trade = (target_stock_value - current_stock_value) / agent_data['current_price']
        
        transaction_cost = 0
        
        if shares_to_trade > 0:  
            cost = shares_to_trade * agent_data['current_price'] * (1 + self.transaction_fee)
            if cost <= agent_data['balance']:
                agent_data['shares_held'] += shares_to_trade
                agent_data['balance'] -= cost
                agent_data['entry_price'] = agent_data['current_price']
                transaction_cost = cost * self.transaction_fee
        elif shares_to_trade < 0:  
            shares_to_sell = min(abs(shares_to_trade), agent_data['shares_held'])
            sale_value = shares_to_sell * agent_data['current_price'] * (1 - self.transaction_fee)
            agent_data['balance'] += sale_value
            agent_data['shares_held'] -= shares_to_sell
            transaction_cost = sale_value * self.transaction_fee
            if agent_data['shares_held'] == 0:
                agent_data['entry_price'] = None

        current_value = agent_data['balance'] + (agent_data['shares_held'] * agent_data['current_price'])
        
        agent_data['max_value'] = max(agent_data['max_value'], current_value)
        
        value_change = (current_value - previous_value) / previous_value
        drawdown = (agent_data['max_value'] - current_value) / agent_data['max_value']
        
        reward = value_change - (transaction_cost / previous_value) - (drawdown * 0.1)
        reward *= 100  
        
        self.done = agent_data['current_step'] >= len(data) - 1

        return self._get_state(stock), reward, self.done

def train_agent(df, episodes=200, early_stop_patience=50):
    """Train Q-learning agents with early stopping.
    
    Args:
        df (dict): Dictionary of stock dataframes
        episodes (int): Maximum number of episodes
        early_stop_patience (int): Episodes to wait before early stopping
    
    Returns:
        tuple: Trained agents, training history
    """
    env = StockTradingEnv(df)
    
    agents = {
        stock: QLearningAgent(env.state_space, env.action_space) 
        for stock in df.keys()
    }
    
    history = {
        'episode_rewards': [],
        'portfolio_values': [],
        'epsilon_values': {stock: [] for stock in df.keys()}
    }
    
    best_rewards = {stock: float('-inf') for stock in df.keys()}
    episodes_without_improvement = {stock: 0 for stock in df.keys()}
    max_episodes_without_improvement = 0
    
    for episode in range(episodes):
        states = env.reset()
        total_rewards = {stock: 0 for stock in df.keys()}
        
        while not env.done:
            actions = {}
            for stock in df.keys():
                actions[stock] = agents[stock].get_action(states[stock])
            
            next_states = {}
            rewards = {}
            for stock in df.keys():
                next_state, reward, done = env.step(stock, actions[stock])
                next_states[stock] = next_state
                rewards[stock] = reward
                total_rewards[stock] += reward
            
            for stock in df.keys():
                if next_states[stock] is not None:  
                    agents[stock].update(states[stock], actions[stock], rewards[stock], next_states[stock])
            
            states = next_states
        
        portfolio_values = {}
        returns = {}
        for stock in df.keys():
            portfolio_value = env.agent_data[stock]['balance'] + (
                env.agent_data[stock]['shares_held'] * 
                env.df[stock]['Close'].iloc[env.agent_data[stock]['current_step']]
            )
            initial_value = env.initial_balance / len(df)
            return_pct = ((portfolio_value - initial_value) / initial_value) * 100
            portfolio_values[stock] = portfolio_value
            returns[stock] = return_pct
        
        history['episode_rewards'].append(sum(total_rewards.values()))
        history['portfolio_values'].append(sum(portfolio_values.values()))
        for stock in df.keys():
            history['epsilon_values'][stock].append(agents[stock].epsilon)
        
        for stock in df.keys():
            if total_rewards[stock] > best_rewards[stock]:
                best_rewards[stock] = total_rewards[stock]
                agents[stock].save_if_better(total_rewards[stock])
                episodes_without_improvement[stock] = 0
            else:
                episodes_without_improvement[stock] += 1
            
            agents[stock].decay_epsilon()
        
        max_episodes_without_improvement = max(episodes_without_improvement.values())
        if max_episodes_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping at episode {episode}")
            break
        
        if episode % 10 == 0:
            print(f"\nEpisode {episode}/{episodes}")
            for stock in df.keys():
                print(f"{stock} - Portfolio: ${portfolio_values[stock]:.2f} ({returns[stock]:+.2f}%), "
                      f"Epsilon: {agents[stock].epsilon:.3f}")
    
    for stock in df.keys():
        agents[stock].load_best()
    
    return agents, history

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    from load_data import load_data
    from split import split_data
    
    def plot_training_history(history):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        ax1.plot(history['episode_rewards'])
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True)
        
        ax2.plot(history['portfolio_values'])
        ax2.set_title('Portfolio Value')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.grid(True)
        
        ax3.plot(history['epsilon_values']['QQQ'], label='QQQ')
        ax3.plot(history['epsilon_values']['SPY'], label='SPY')
        ax3.plot(history['epsilon_values']['VOO'], label='VOO')
        ax3.set_title('Exploration Rate (Epsilon)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

    def evaluate_agent(agents, env, plot=True):
        states = env.reset()
        total_reward = 0
        portfolio_values = []
        actions_taken = []
        
        while not env.done:
            actions = {}
            for stock in env.df.keys():
                actions[stock] = agents[stock].get_action(states[stock], is_training=False)
                
            next_states = {}
            for stock in env.df.keys():
                next_state, reward, done = env.step(stock, actions[stock])
                next_states[stock] = next_state
                total_reward += reward
                
                portfolio_value = sum(env.agent_data[s]['balance'] + 
                                   env.agent_data[s]['shares_held'] * 
                                   env.df[s]['Close'].iloc[env.agent_data[s]['current_step']]
                                   for s in env.df.keys())
                portfolio_values.append(portfolio_value)
                actions_taken.append(actions)
            
            states = next_states
        
        print(f"\nEvaluation Results:")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(portfolio_values)
            plt.title('Portfolio Value During Evaluation')
            plt.xlabel('Trading Step')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.show()
        
        return total_reward, portfolio_values, actions_taken

    print("Loading data...")
    qqq, spy, voo = load_data()
    (train_qqq, test_qqq), (train_spy, test_spy), (train_voo, test_voo) = split_data(qqq, spy, voo)
    
    train_data = {'QQQ': train_qqq, 'SPY': train_spy, 'VOO': train_voo}
    test_data = {'QQQ': test_qqq, 'SPY': test_spy, 'VOO': test_voo}
    
    print("\nInitializing training environment...")
    env = StockTradingEnv(train_data)
    
    print("\nTraining agents...")
    agents, history = train_agent(train_data, episodes=200, early_stop_patience=50)
    
    plot_training_history(history)
    
    print("\nEvaluating agent on test data...")
    test_env = StockTradingEnv(test_data)
    eval_reward, eval_portfolio_values, eval_actions = evaluate_agent(agents, test_env)