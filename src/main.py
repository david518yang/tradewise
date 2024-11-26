from load_data import load_data
from split import split_data
from qlearn import StockTradingEnv, QLearningAgent
import matplotlib.pyplot as plt
import numpy as np

def train_agents(env, episodes=1000):
    agents = {stock: QLearningAgent(env.state_space, env.action_space) for stock in env.df.keys()}
    for episode in range(episodes):
        states = env.reset()
        total_rewards = {stock: 0 for stock in env.df.keys()}

        while not env.done:
            for stock in env.df.keys():
                state = states[stock]
                
                # Skip if state is None
                if state is None:
                    continue
                
                action = agents[stock].get_action(state)
                next_state, reward, done = env.step(stock, action)
                
                # Skip updates if next_state is None
                if next_state is not None:
                    agents[stock].update(state, action, reward, next_state)
                states[stock] = next_state
                total_rewards[stock] += reward

        total_rewards_clean = {stock: float(reward) for stock, reward in total_rewards.items()}

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Rewards: {total_rewards_clean}")

    return agents


def evaluate_agents(agents, env):
    states = env.reset()
    portfolio_values = {stock: [env.initial_balance / len(env.df)] for stock in env.df.keys()}
    actions_taken = {stock: [] for stock in env.df.keys()}

    while not env.done:
        for stock in env.df.keys():
            state = states[stock]
            
            # Skip processing if state is None
            if state is None:
                continue

            action = agents[stock].get_action(state)
            next_state, reward, done = env.step(stock, action)

            # Skip updates if next_state is None
            if next_state is not None:
                states[stock] = next_state
                portfolio_value = env.agent_data[stock]['balance'] + (
                    env.agent_data[stock]['shares_held'] * env.df[stock]['Close'].iloc[env.agent_data[stock]['current_step']]
                )
                portfolio_values[stock].append(portfolio_value)
                actions_taken[stock].append(action)

    return portfolio_values, actions_taken

def calculate_metrics(portfolio_values, initial_balance):
    metrics = {}
    for stock, values in portfolio_values.items():
        initial_value = initial_balance / len(portfolio_values)
        final_value = values[-1]

        total_return = ((final_value - initial_value) / initial_value) * 100
        daily_returns = np.diff(values) / values[:-1]
        risk_free_rate = 0.01
        sharpe_ratio = (np.mean(daily_returns) - risk_free_rate) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
        cumulative_returns = np.array(values) / initial_value
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        max_drawdown = np.max(drawdowns) * 100

        metrics[stock] = {
            "Total Return (%)": round(total_return, 2),
            "Sharpe Ratio": round(sharpe_ratio, 2),
            "Max Drawdown (%)": round(max_drawdown, 2),
            "Final Portfolio Value": round(final_value, 2)
        }
    return metrics

def plot_results(portfolio_values, test_data, actions_taken):
    for stock, values in portfolio_values.items():
        plt.figure(figsize=(15, 10))

        # Plot portfolio value
        plt.subplot(2, 1, 1)
        plt.plot(values, label="Portfolio Value", color="blue")
        plt.title(f"{stock} Portfolio Value Over Time")
        plt.xlabel("Trading Steps")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True)

        # Plot stock price and actions
        plt.subplot(2, 1, 2)
        stock_prices = test_data[stock]['Close'].values
        plt.plot(stock_prices, label="Stock Price", color="orange", alpha=0.7)
        for i, action in enumerate(actions_taken[stock]):
            if action == 1:  # Buy
                plt.scatter(i, stock_prices[i], color="green", marker="^", label="Buy" if i == 0 else "")
            elif action == 2:  # Sell
                plt.scatter(i, stock_prices[i], color="red", marker="v", label="Sell" if i == 0 else "")
        plt.title(f"{stock} Stock Price and Trading Actions")
        plt.xlabel("Trading Steps")
        plt.ylabel("Stock Price ($)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show(block = True)

# Main workflow
aapl, googl, msft = load_data()
(train_aapl, test_aapl), (train_googl, test_googl), (train_msft, test_msft) = split_data(aapl, googl, msft)
train_data = {'AAPL': train_aapl, 'GOOGL': train_googl, 'MSFT': train_msft}
test_data = {'AAPL': test_aapl, 'GOOGL': test_googl, 'MSFT': test_msft}

env = StockTradingEnv(train_data)
print("Training agents...")
agents = train_agents(env, episodes=1000)

print("\nEvaluating agents...")
test_env = StockTradingEnv(test_data)
portfolio_values, actions_taken = evaluate_agents(agents, test_env)

print("\nCalculating performance metrics...")
metrics = calculate_metrics(portfolio_values, env.initial_balance)
print("\nPerformance Metrics:")
for stock, metric_data in metrics.items():
    print(f"{stock}:")
    for metric, value in metric_data.items():
        print(f"  {metric}: {value}")

print("\nPlotting results...")
plot_results(portfolio_values, test_data, actions_taken)