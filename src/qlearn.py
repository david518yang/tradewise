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
        self.min_trade_pct = 0.4  # Minimum trade size as percentage of portfolio
        
        # Simplified agent data
        self.agent_data = {
            key: {
                'balance': self.initial_balance / len(df),
                'shares_held': 0,
                'current_step': 0,
                'entry_price': None,
                'last_action_step': 0,
                'last_position': 0
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
        self.done = False
        return {key: self._get_state(key) for key in self.df.keys()}

    def _calculate_technical_indicators(self, data, current_step):
        # Calculate RSI
        delta = data['Close'].diff().dropna()
        up_days = delta.copy(), up_days[delta > 0] = delta[delta > 0]
        down_days = delta.copy(), down_days[delta < 0] = -delta[delta < 0]
        RS_up = up_days.ewm(com=13 - 1, min_periods=0, adjust=False).mean()
        RS_down = down_days.ewm(com=13 - 1, min_periods=0, adjust=False).mean().abs()
        RS = RS_up / RS_down
        RSI = 100.0 - (100.0 / (1.0 + RS))
        rsi = RSI.iloc[current_step] if not np.isnan(RSI.iloc[current_step]) else 50.0

        # Calculate MACD
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal
        macd = macd.iloc[current_step] if not np.isnan(macd.iloc[current_step]) else 0.0
        macd_hist = macd_hist.iloc[current_step] if not np.isnan(macd_hist.iloc[current_step]) else 0.0

        return rsi, macd, macd_hist

    def _calculate_etf_features(self, data, current_step):
        # Relative Volume Analysis
        volume = data['Volume'].iloc[current_step]
        avg_volume_20d = data['Volume'].iloc[max(0, current_step-20):current_step+1].mean()
        relative_volume = volume / avg_volume_20d if avg_volume_20d > 0 else 1.0

        # Price Momentum and Volatility
        returns_20d = data['Close'].pct_change(20).iloc[current_step]
        volatility_20d = data['Close'].pct_change().iloc[max(0, current_step-20):current_step+1].std()
        
        # Money Flow Index (MFI) - Volume-weighted RSI
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        delta_money_flow = money_flow.diff()
        positive_flow = delta_money_flow.where(delta_money_flow > 0, 0).rolling(window=14).sum()
        negative_flow = (-delta_money_flow.where(delta_money_flow < 0, 0)).rolling(window=14).sum()
        
        mfi_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + mfi_ratio))
        current_mfi = mfi.iloc[current_step] if not np.isnan(mfi.iloc[current_step]) else 50.0

        # Bollinger Band Position
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        current_price = data['Close'].iloc[current_step]
        
        if current_step >= 20:
            current_mean = rolling_mean.iloc[current_step]
            current_std = rolling_std.iloc[current_step]
            if current_std != 0:
                bb_position = (current_price - current_mean) / (2 * current_std)
            else:
                bb_position = 0
        else:
            bb_position = 0

        return np.array([
            float(relative_volume - 1),  # Normalized relative volume
            float(returns_20d),          # 20-day returns
            float(volatility_20d),       # 20-day volatility
            float((current_mfi - 50) / 50),  # Normalized MFI
            float(bb_position),          # Bollinger Band position (-1 to 1)
        ])

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
        
        # Calculate target stock value based on percentage change of current position
        if current_stock_value == 0 and target_percentage > 0:
            # Opening new position - use minimum position size as base
            target_stock_value = self.min_trade_pct * (agent_data['balance'] + current_stock_value)
        else:
            target_stock_value = current_stock_value * (1 + target_percentage)
        
        # Calculate shares to trade
        shares_to_trade = (target_stock_value - current_stock_value) / agent_data['current_price']
        
        if shares_to_trade > 0:  # Buy
            cost = shares_to_trade * agent_data['current_price'] * (1 + self.transaction_fee)
            # Check if we have enough balance and the position would be above minimum
            if cost <= agent_data['balance'] and target_stock_value >= self.min_trade_pct * (agent_data['balance'] + current_stock_value):
                agent_data['shares_held'] += shares_to_trade
                agent_data['balance'] -= cost
                agent_data['entry_price'] = agent_data['current_price']
        elif shares_to_trade < 0:  # Sell
            shares_to_sell = min(abs(shares_to_trade), agent_data['shares_held'])
            # If we're not selling everything, ensure remaining position is above minimum
            remaining_value = (agent_data['shares_held'] - shares_to_sell) * agent_data['current_price']
            if remaining_value == 0 or remaining_value >= self.min_trade_pct * (agent_data['balance'] + current_stock_value):
                sale_value = shares_to_sell * agent_data['current_price'] * (1 - self.transaction_fee)
                agent_data['balance'] += sale_value
                agent_data['shares_held'] -= shares_to_sell
                if agent_data['shares_held'] == 0:
                    agent_data['entry_price'] = None

        current_value = agent_data['balance'] + (agent_data['shares_held'] * agent_data['current_price'])
        reward = ((current_value - previous_value) / previous_value) * 100
        self.done = agent_data['current_step'] >= len(data) - 1

        return self._get_state(stock), reward, self.done

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
