# main.py
from load_data import load_data
from deepq import StockTradingEnv, DQNAgent, train_agents, evaluate_agents, plot_trade_histories
from analysis import FeatureAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
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
        'QQQ': train_qqq.copy(),
        'SPY': train_spy.copy(),
        'VOO': train_voo.copy()
    }
    test_data = {
        'QQQ': test_qqq.copy(),
        'SPY': test_spy.copy(),
        'VOO': test_voo.copy()
    }
    
    # Create environments
    env = StockTradingEnv(train_data)
    test_env = StockTradingEnv(test_data)
    
    # Analyze features before training
    print("\nAnalyzing features...")
    analyzer = FeatureAnalyzer(env)
    rf_importance, entropy_importance, states_dict = analyzer.analyze_features(episodes=100, show_plots=False)
    
    # Plot feature importance
    analyzer.visualize_feature_importance(rf_importance, 'Random Forest')
    analyzer.visualize_feature_importance(entropy_importance, 'Entropy')
    
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

if __name__ == "__main__":
    main()
