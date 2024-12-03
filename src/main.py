from load_data import load_data
from qlearn import StockTradingEnv, train_agents
import numpy as np

def main():
    # Load data
    qqq, spy, voo = load_data()
    
    # Create training data dictionary
    train_data = {
        'QQQ': qqq,
        'SPY': spy,
        'VOO': voo
    }
    
    # Initialize environment
    env = StockTradingEnv(train_data)
    
    print("\nTraining agents...")
    agents, training_history = train_agents(env, episodes=200)
    
    return agents, training_history

if __name__ == "__main__":
    agents, history = main()
    
    # Print final portfolio values
    for stock in history:
        print(f"\n{stock} Final Portfolio Value: ${history[stock]['portfolio_values'][-1]:.2f}")