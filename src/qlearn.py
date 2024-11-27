import numpy as np
import pandas as pd

class StockTradingEnv:
    def __init__(self, df):
        self.df = df
        self.action_space = 9  # [-100%, -75%, -50%, -25%, 0, +25%, +50%, +75%, +100%]
        self.actions = [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]  # Percentage mappings
        
        # State space: position + 5 market features
        self.state_space = 6
        
        self.initial_balance = 10000
        self.transaction_fee = 0.001
        # Minimum position size is 10% of initial balance per stock
        self.min_position = self.initial_balance * 0.1

        self.agent_data = {key: {'balance': self.initial_balance / len(df), 'shares_held': 0, 'current_step': 0} for key in df.keys()}
        self.done = False
        self.reset()

    def reset(self):
        for key, agent_data in self.agent_data.items():
            agent_data['balance'] = self.initial_balance / len(self.df)
            agent_data['shares_held'] = 0
            agent_data['current_step'] = 0
        self.done = False
        return {key: self._get_state(key) for key in self.df.keys()}

    def _get_state(self, stock):
        data = self.df[stock]
        agent_data = self.agent_data[stock]
        market_features = np.array([
            data['Open'].iloc[agent_data['current_step']],
            data['High'].iloc[agent_data['current_step']],
            data['Low'].iloc[agent_data['current_step']],
            data['Close'].iloc[agent_data['current_step']],
            data['Volume'].iloc[agent_data['current_step']]
        ])
        position = np.array([agent_data['shares_held']])
        return np.concatenate([position, market_features])

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
        target_percentage = self.actions[action]
        
        # Calculate target stock value based on percentage change of current position
        if current_stock_value == 0 and target_percentage > 0:
            # Opening new position - use minimum position size as base
            target_stock_value = self.min_position * (1 + target_percentage)
        else:
            target_stock_value = current_stock_value * (1 + target_percentage)
        
        # Calculate shares to trade
        shares_to_trade = (target_stock_value - current_stock_value) / agent_data['current_price']
        
        if shares_to_trade > 0:  # Buy
            cost = shares_to_trade * agent_data['current_price'] * (1 + self.transaction_fee)
            # Check if we have enough balance and the position would be above minimum
            if cost <= agent_data['balance'] and target_stock_value >= self.min_position:
                agent_data['shares_held'] += shares_to_trade
                agent_data['balance'] -= cost
        elif shares_to_trade < 0:  # Sell
            shares_to_sell = min(abs(shares_to_trade), agent_data['shares_held'])
            # If we're not selling everything, ensure remaining position is above minimum
            remaining_value = (agent_data['shares_held'] - shares_to_sell) * agent_data['current_price']
            if remaining_value == 0 or remaining_value >= self.min_position:
                sale_value = shares_to_sell * agent_data['current_price'] * (1 - self.transaction_fee)
                agent_data['balance'] += sale_value
                agent_data['shares_held'] -= shares_to_sell

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
