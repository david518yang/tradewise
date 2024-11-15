from load_data import load_data
from split import split_data
from qlearn import train_agent, StockTradingEnv
import matplotlib.pyplot as plt
import numpy as np

def evaluate_agent(agent, test_data):
    env = StockTradingEnv(test_data)
    state = env.reset()
    total_return = 0
    portfolio_values = [env.initial_balance]
    actions_taken = []
    
    while not env.done:
        action = agent.get_action(state)
        state, reward, done = env.step(action)
        total_return += reward
        
        # Track portfolio value and actions
        portfolio_value = env.balance + (env.shares_held * env.current_price)
        portfolio_values.append(portfolio_value)
        actions_taken.append(action)
    
    return portfolio_values, actions_taken

def plot_results(test_data, portfolio_values, actions_taken):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot portfolio value
    ax1.plot(portfolio_values, label='Portfolio Value')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Trading Steps')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot stock price and actions
    price_data = test_data['Close'].values
    ax2.plot(price_data, label='Stock Price', color='blue', alpha=0.6)
    ax2.set_title('Stock Price and Trading Actions')
    ax2.set_xlabel('Trading Steps')
    ax2.set_ylabel('Stock Price ($)')
    
    # Plot buy/sell points
    for i, action in enumerate(actions_taken):
        if action == 1:  # Buy
            ax2.scatter(i, price_data[i], color='green', marker='^', label='Buy' if i == 0 else "")
        elif action == 2:  # Sell
            ax2.scatter(i, price_data[i], color='red', marker='v', label='Sell' if i == 0 else "")
    
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show(block=True)

def calculate_metrics(portfolio_values, test_data):
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    
    # Total Return
    total_return = ((final_value - initial_value) / initial_value) * 100
    
    # Daily Returns
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    # Sharpe Ratio (assuming risk-free rate of 0.01)
    risk_free_rate = 0.01
    sharpe_ratio = (np.mean(daily_returns) - risk_free_rate) / np.std(daily_returns)
    
    # Maximum Drawdown
    cumulative_returns = np.array(portfolio_values) / initial_value
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (running_max - cumulative_returns) / running_max
    max_drawdown = np.max(drawdowns) * 100
    
    metrics = {
        "Total Return (%)": round(total_return, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Final Portfolio Value": round(final_value, 2)
    }
    
    return metrics

# Load and split data using your existing functions
data = load_data()
train, test = split_data(data)

# Train agent
print("Training agent...")
agent = train_agent(train, episodes=1000)

# Evaluate agent
print("\nEvaluating agent...")
portfolio_values, actions_taken = evaluate_agent(agent, test)

# Calculate and display metrics
metrics = calculate_metrics(portfolio_values, test)
print("\nTrading Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

# Plot results
plot_results(test, portfolio_values, actions_taken)