import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import entropy
from collections import defaultdict
import torch
import torch.nn as nn

class FeatureAnalyzer:
    def __init__(self, env):
        self.env = env
        self.feature_names = [
            'Returns', 'Momentum_5', 'Momentum_10', 'Position_Size', 'Position_PnL',
            'Normalized_Balance', 'RSI', 'MACD', 'MACD_Hist',
            'Relative_Volume', 'Returns_20d', 'Volatility_20d', 'MFI', 'BB_Position',
            'Corr_ETF1', 'Corr_ETF2'
        ]
        
    def collect_states_and_rewards(self, episodes=100):
        states_dict = defaultdict(list)
        rewards_dict = defaultdict(list)
        
        for episode in range(episodes):
            states = self.env.reset()
            for stock in self.env.df.keys():
                done = False
                while not done:
                    state = states[stock]
                    action = np.random.randint(self.env.action_space)
                    next_state, reward, done = self.env.step(stock, action)
                    if next_state is not None:
                        states_dict[stock].append(state)
                        rewards_dict[stock].append(reward)
                    states[stock] = next_state
        
        return states_dict, rewards_dict
    
    def calculate_feature_importance_rf(self, states_dict, rewards_dict):
        importance_dict = {}
        
        for stock in states_dict.keys():
            X = np.array(states_dict[stock])
            y = np.array(rewards_dict[stock])
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance_dict[stock] = dict(zip(self.feature_names, rf.feature_importances_))
        
        return importance_dict
    
    def calculate_feature_importance_entropy(self, states_dict):
        importance_dict = {}
        
        for stock in states_dict.keys():
            X = np.array(states_dict[stock])
            
            # Calculate entropy for each feature
            feature_entropies = []
            for i in range(X.shape[1]):
                # Bin the data for entropy calculation
                hist, _ = np.histogram(X[:, i], bins=20, density=True)
                feature_entropies.append(entropy(hist + 1e-10))  # Add small constant to avoid log(0)
            
            # Normalize entropies
            total_entropy = sum(feature_entropies)
            normalized_entropies = [e/total_entropy for e in feature_entropies]
            
            importance_dict[stock] = dict(zip(self.feature_names, normalized_entropies))
        
        return importance_dict
    
    def visualize_feature_importance(self, importance_dict, method='random_forest'):
        plt.figure(figsize=(15, 10))
        
        for i, (stock, importances) in enumerate(importance_dict.items()):
            plt.subplot(len(importance_dict), 1, i+1)
            
            features = list(importances.keys())
            values = list(importances.values())
            
            # Sort by importance
            sorted_idx = np.argsort(values)
            pos = np.arange(len(features))
            
            plt.barh(pos, np.array(values)[sorted_idx])
            plt.yticks(pos, np.array(features)[sorted_idx])
            plt.title(f'Feature Importance for {stock} using {method}')
            plt.xlabel('Importance Score')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_state_features(self, stock, window=100):
        states_dict, _ = self.collect_states_and_rewards(episodes=1)
        states = np.array(states_dict[stock])
        
        # Create a rolling window view of the features
        plt.figure(figsize=(15, 12))
        n_features = states.shape[1]
        n_cols = 2
        n_rows = (n_features + 1) // 2
        
        for i in range(n_features):
            plt.subplot(n_rows, n_cols, i+1)
            plt.plot(states[:, i], label=self.feature_names[i])
            plt.title(self.feature_names[i])
            plt.grid(True)
            
            # Add rolling mean
            if len(states) > window:
                rolling_mean = pd.Series(states[:, i]).rolling(window=window).mean()
                plt.plot(rolling_mean, 'r--', label=f'{window}-period MA')
            
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def visualize_feature_correlations(self, stock):
        states_dict, _ = self.collect_states_and_rewards(episodes=1)
        states = np.array(states_dict[stock])
        
        # Calculate correlation matrix
        corr_matrix = pd.DataFrame(states, columns=self.feature_names).corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title(f'Feature Correlations for {stock}')
        plt.tight_layout()
        plt.show()
    
    def analyze_features(self, episodes=100):
        print("Collecting state and reward data...")
        states_dict, rewards_dict = self.collect_states_and_rewards(episodes)
        
        print("\nCalculating feature importance using Random Forest...")
        rf_importance = self.calculate_feature_importance_rf(states_dict, rewards_dict)
        self.visualize_feature_importance(rf_importance, 'Random Forest')
        
        print("\nCalculating feature importance using Entropy...")
        entropy_importance = self.calculate_feature_importance_entropy(states_dict)
        self.visualize_feature_importance(entropy_importance, 'Entropy')
        
        print("\nVisualizing feature correlations...")
        for stock in states_dict.keys():
            self.visualize_feature_correlations(stock)
        
        print("\nVisualizing state features...")
        for stock in states_dict.keys():
            self.visualize_state_features(stock)
        
        return rf_importance, entropy_importance
