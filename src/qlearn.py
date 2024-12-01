import numpy as np
import pandas as pd

class StockTradingEnv:
    def __init__(self, df):
        self.df = df
        
        # Simplified action space: Strong Sell, Sell, Hold, Buy, Strong Buy
        self.position_sizes = np.array([-1.0, -0.5, 0, 0.5, 1.0])  # 5 actions
        self.action_space = len(self.position_sizes)
        
        # Updated state space size: original features + ETF features + correlations
        self.state_space = 9 + 5 + (len(df) - 1)  # Original + ETF features + Correlations
        
        self.initial_balance = 10000
        self.transaction_fee = 0.001  # 0.1% transaction fee
        self.min_trade_pct = 0.2  # Reduced minimum trade size to 20% of portfolio
        
        # Simplified agent data
        self.agent_data = {
            key: {
                'balance': self.initial_balance / len(df),
                'shares_held': 0,
                'current_step': 0,
                'entry_price': None,
                'last_action_step': 0,
                'last_position': 0,
                'max_value': self.initial_balance / len(df)  # Track maximum portfolio value
            } for key in df.keys()
        }
        self.done = False
        self.reset()

    def reset(self):
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
        try:
            # Calculate RSI
            delta = data['Close'].diff()
            up_days = delta.copy()
            down_days = delta.copy()
            up_days[delta <= 0] = 0
            down_days[delta >= 0] = 0
            
            # Calculate exponential moving averages
            RS_up = up_days.ewm(com=13, min_periods=1, adjust=False).mean()
            RS_down = abs(down_days).ewm(com=13, min_periods=1, adjust=False).mean()
            
            # Calculate RSI
            RS = RS_up / RS_down
            RSI = 100.0 - (100.0 / (1.0 + RS))
            
            # Get current values with bounds checking
            if 0 <= current_step < len(RSI):
                current_rsi = RSI.iloc[current_step]
                if np.isnan(current_rsi):
                    current_rsi = 50.0
            else:
                current_rsi = 50.0

            # Calculate MACD
            exp1 = data['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
            macd_hist = macd - signal
            
            # Get current MACD values with bounds checking
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
            return 50.0, 0.0, 0.0  # Return neutral values on error

    def _calculate_etf_features(self, data, current_step):
        try:
            # Ensure current_step is within bounds
            if current_step >= len(data):
                return np.zeros(5)  # Return zeros for all features if out of bounds
            
            # Relative Volume Analysis
            volume = data['Volume'].iloc[current_step]
            start_idx = max(0, current_step-20)
            avg_volume_20d = data['Volume'].iloc[start_idx:current_step+1].mean()
            relative_volume = volume / avg_volume_20d if avg_volume_20d > 0 else 1.0

            # Price Momentum and Volatility
            returns_20d = 0.0
            volatility_20d = 0.0
            if current_step >= 20:
                returns_20d = (data['Close'].iloc[current_step] / data['Close'].iloc[current_step-20]) - 1
                volatility_20d = data['Close'].pct_change().iloc[current_step-20:current_step+1].std()
            
            # Money Flow Index (MFI)
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

            # Bollinger Band Position
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
                float(relative_volume - 1),  # Normalized relative volume
                float(returns_20d),          # 20-day returns
                float(volatility_20d),       # 20-day volatility
                float((current_mfi - 50) / 50),  # Normalized MFI
                float(bb_position),          # Bollinger Band position (-1 to 1)
            ])
            
        except Exception as e:
            print(f"Error calculating ETF features: {e}")
            return np.zeros(5)  # Return zeros for all features on error

    def _calculate_correlation_features(self, stock):
        current_step = self.agent_data[stock]['current_step']
        lookback = 20  # Correlation lookback period
        
        if current_step < lookback:
            return np.zeros(len(self.df) - 1)  # Return zeros for all correlations
        
        correlations = []
        current_returns = self.df[stock]['Close'].pct_change().iloc[current_step-lookback:current_step]
        
        for other_stock in self.df.keys():
            if other_stock != stock:
                other_returns = self.df[other_stock]['Close'].pct_change().iloc[current_step-lookback:current_step]
                corr = current_returns.corr(other_returns)
                correlations.append(float(corr) if not np.isnan(corr) else 0.0)
        
        return np.array(correlations)

    def _get_state(self, stock):
        data = self.df[stock]
        agent_data = self.agent_data[stock]
        current_step = agent_data['current_step']
        
        # Get existing features
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
        
        # Calculate technical indicators
        rsi, macd, macd_hist = self._calculate_technical_indicators(data, current_step)
        
        # Calculate ETF-specific features
        etf_features = self._calculate_etf_features(data, current_step)
        
        # Calculate correlation features
        correlation_features = self._calculate_correlation_features(stock)
        
        # Calculate position metrics
        position_size = 0.0
        position_pnl = 0.0
        if agent_data['shares_held'] != 0:
            current_price = data['Close'].iloc[current_step]
            position_value = agent_data['shares_held'] * current_price
            total_value = position_value + agent_data['balance']
            position_size = position_value / total_value
            
            if agent_data['entry_price'] is not None:
                position_pnl = (current_price - agent_data['entry_price']) / agent_data['entry_price']
        
        # Normalize RSI to -1 to 1 range
        normalized_rsi = (rsi - 50) / 50
        
        # Combine all features
        market_features = np.concatenate([
            [
                float(returns),              # Recent price change
                float(momentum_5),           # 5-day momentum
                float(momentum_10),          # 10-day momentum
                float(position_size),        # Current position size (-1 to 1)
                float(position_pnl),         # Current position P&L
                float(agent_data['balance'] / self.initial_balance),  # Normalized balance
                float(normalized_rsi),       # RSI indicator
                float(macd),                # MACD line
                float(macd_hist),           # MACD histogram
            ],
            etf_features,                   # ETF-specific features
            correlation_features            # Correlation with other ETFs
        ])
        
        # Ensure no NaN values in the state
        market_features = np.nan_to_num(market_features, nan=0.0)
        
        return market_features

    def step(self, stock, action):
        agent_data = self.agent_data[stock]
        data = self.df[stock]

        if agent_data['current_step'] >= len(data) - 1:
            self.done = True
            return None, 0, self.done

        previous_value = agent_data['balance'] + (agent_data['shares_held'] * data['Close'].iloc[agent_data['current_step']])
        agent_data['current_step'] += 1
        agent_data['current_price'] = data['Close'].iloc[agent_data['current_step']]

        # Calculate position change based on current stock holdings
        current_stock_value = agent_data['shares_held'] * agent_data['current_price']
        target_percentage = self.position_sizes[action]
        total_value = agent_data['balance'] + current_stock_value
        
        # Calculate target stock value based on total portfolio value
        target_stock_value = total_value * target_percentage if target_percentage > 0 else 0
        
        # Calculate shares to trade
        shares_to_trade = (target_stock_value - current_stock_value) / agent_data['current_price']
        
        # Track transaction costs
        transaction_cost = 0
        
        if shares_to_trade > 0:  # Buy
            cost = shares_to_trade * agent_data['current_price'] * (1 + self.transaction_fee)
            if cost <= agent_data['balance']:
                agent_data['shares_held'] += shares_to_trade
                agent_data['balance'] -= cost
                agent_data['entry_price'] = agent_data['current_price']
                transaction_cost = cost * self.transaction_fee
        elif shares_to_trade < 0:  # Sell
            shares_to_sell = min(abs(shares_to_trade), agent_data['shares_held'])
            sale_value = shares_to_sell * agent_data['current_price'] * (1 - self.transaction_fee)
            agent_data['balance'] += sale_value
            agent_data['shares_held'] -= shares_to_sell
            transaction_cost = sale_value * self.transaction_fee
            if agent_data['shares_held'] == 0:
                agent_data['entry_price'] = None

        current_value = agent_data['balance'] + (agent_data['shares_held'] * agent_data['current_price'])
        
        # Update maximum portfolio value
        agent_data['max_value'] = max(agent_data['max_value'], current_value)
        
        # Calculate reward components
        value_change = (current_value - previous_value) / previous_value
        drawdown = (agent_data['max_value'] - current_value) / agent_data['max_value']
        
        # Penalize excessive trading and drawdowns
        reward = value_change - (transaction_cost / previous_value) - (drawdown * 0.1)
        reward *= 100  # Scale reward for better learning
        
        self.done = agent_data['current_step'] >= len(data) - 1

        return self._get_state(stock), reward, self.done

class QLearningAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = {}
        self.learning_rate = 0.1  # Increased learning rate
        self.gamma = 0.99  # Increased future reward importance
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995  # Slower epsilon decay
        self.epsilon_min = 0.05  # Increased minimum exploration

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
