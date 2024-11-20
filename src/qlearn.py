import numpy as np
import pandas as pd

class StockTradingEnv:
    def __init__(self, df):
        self.df = df
        self.initial_balance = 10000
        self.transaction_fee = 0.001
        self.max_shares = 100  # Maximum shares to trade in single action
        
        # State space: [shares_held, cash_balance, open, high, low, close, volume]
        self.state_space = 7
        
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_price = self.df['Close'].iloc[0]
        self.done = False
        return self._get_state()

    def _get_state(self):
        state = np.array([
            self.shares_held,
            self.balance,
            self.df['Open'].iloc[self.current_step],
            self.df['High'].iloc[self.current_step],
            self.df['Low'].iloc[self.current_step],
            self.df['Close'].iloc[self.current_step],
            self.df['Volume'].iloc[self.current_step]
        ])
        return state

    def step(self, action):
        """
        action: integer representing number of shares to buy (positive) or sell (negative)
        """
        previous_value = self.balance + (self.shares_held * self.current_price)
        self.current_step += 1
        self.current_price = self.df['Close'].iloc[self.current_step]
        
        # Limit action to max_shares
        action = np.clip(action, -self.max_shares, self.max_shares)
        
        if action > 0:  # Buy
            max_shares_possible = self.balance // (self.current_price * (1 + self.transaction_fee))
            shares_to_buy = min(action, max_shares_possible)
            cost = shares_to_buy * self.current_price * (1 + self.transaction_fee)
            
            if cost <= self.balance:
                self.shares_held += shares_to_buy
                self.balance -= cost
                
        elif action < 0:  # Sell
            shares_to_sell = min(abs(action), self.shares_held)
            sale_value = shares_to_sell * self.current_price * (1 - self.transaction_fee)
            
            self.shares_held -= shares_to_sell
            self.balance += sale_value

        # Calculate reward as percentage change in portfolio value
        current_value = self.balance + (self.shares_held * self.current_price)
        reward = ((current_value - previous_value) / previous_value) * 100
        
        self.done = self.current_step >= len(self.df) - 1
        
        return self._get_state(), reward, self.done

class ContinuousQLearningAgent:
    def __init__(self, state_space, max_shares):
        # Initialize parameters first
        self.state_space = state_space
        self.max_shares = max_shares
        self.learning_rate = 0.001  # Move this up
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.batch_size = 32
        self.memory = []
        
        # Build model after parameters are initialized
        self.model = self._build_model()

    def _build_model(self):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            Dense(24, input_dim=self.state_space, activation='relu'),
            Dense(24, activation='relu'),
            Dense(1, activation='linear')  # Output is Q-value for given state and action
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action: buy/sell random number of shares or hold
            return np.random.randint(-self.max_shares, self.max_shares + 1)
        
        # Predict Q-values for different actions and choose the best one
        potential_actions = np.arange(-self.max_shares, self.max_shares + 1)
        q_values = []
        
        for action in potential_actions:
            state_action = np.append(state, action)
            q_values.append(self.model.predict(state_action.reshape(1, -1), verbose=0)[0])
            
        return potential_actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state))
        
        if len(self.memory) >= self.batch_size:
            batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
            for idx in batch:
                s, a, r, ns = self.memory[idx]
                
                # Get Q-value for current state-action
                target = r
                if not done:
                    # Predict future rewards for all possible actions
                    potential_actions = np.arange(-self.max_shares, self.max_shares + 1)
                    next_q_values = []
                    for next_action in potential_actions:
                        next_state_action = np.append(ns, next_action)
                        next_q_values.append(
                            self.model.predict(next_state_action.reshape(1, -1), verbose=0)[0]
                        )
                    target += self.gamma * np.max(next_q_values)
                
                state_action = np.append(s, a)
                self.model.fit(
                    state_action.reshape(1, -1),
                    np.array([target]),
                    epochs=1,
                    verbose=0
                )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(df, episodes=1000):
    env = StockTradingEnv(df)
    agent = ContinuousQLearningAgent(env.state_space + 1, env.max_shares)  # +1 for action
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while not env.done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
            
    return agent
