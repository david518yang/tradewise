import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from load_data import load_data
from split import split_data_individual
import matplotlib.pyplot as plt

class LinearRegressionAgent:
    """Linear Regression agent for stock trading."""
    
    def __init__(self):
        """Initialize the linear regression model."""
        self.model = LinearRegression()
    
    def train(self, X_train, y_train):
        """Train the Linear Regression model.
        
        Args:
            X_train (pd.DataFrame): Features for training
            y_train (pd.Series): Target labels for training
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model's performance.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): True labels for the test set
            
        Returns:
            float: R^2 score of the model
        """
        return self.model.score(X_test, y_test)

def prepare_features_labels(df):
    """Prepare features and labels from the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with stock data
    
    Returns:
        tuple: Features (X) and labels (y)
    """
    features = df[['MA5', 'MA20', 'RSI', 'MACD', 'Volume']]  # Adjust as needed
    labels = df['Returns']  # Target is typically the returns or next price
    return features, labels

def calculate_metrics(portfolio_values, initial_balance):
    """Calculate key financial metrics.
    
    Args:
        portfolio_values (list): List of portfolio values over time
        initial_balance (float): The initial amount of money in the portfolio
    
    Returns:
        dict: Dictionary of metrics (total return, sharpe ratio, max drawdown, final value)
    """
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_balance) / initial_balance * 100

    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) != 0 else 0

    max_drawdown = 0
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'final_portfolio_value': final_value
    }

def backtest_agent(test_df, predictions, initial_balance=3333.33):
    """Backtest trading strategy using predictions from the model.
    
    Args:
        test_df (pd.DataFrame): Test dataframe with true values
        predictions (np.ndarray): Predictions from the model
        initial_balance (float): Initial cash balance for the portfolio
    
    Returns:
        list: Portfolio values over time
    """
    balance = initial_balance
    shares = 0
    portfolio_values = []

    for i in range(len(test_df)):
        current_price = test_df['Close'].iloc[i]
        
        # If the model predicts positive returns, buy
        if predictions[i] > 0:
            shares_to_buy = balance / current_price
            balance = 0
            shares += shares_to_buy
        
        # If the model predicts negative returns, sell
        elif predictions[i] < 0 and shares > 0:
            balance += shares * current_price
            shares = 0

        portfolio_value = balance + (shares * current_price)
        portfolio_values.append(portfolio_value)
    
    return portfolio_values

def plot_portfolio(portfolio_values, test_df, stock_name):
    """Plot the portfolio values over time using the date as the x-axis.
    
    Args:
        portfolio_values (list): List of portfolio values over time
        test_df (pd.DataFrame): Test DataFrame containing the dates
        stock_name (str): Name of the stock being plotted
    """
    plt.figure(figsize=(10, 6))
    plt.plot(test_df.index, portfolio_values, label='Portfolio Value', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title(f'Portfolio Value Over Time for {stock_name}')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

def main():
    """Main function to run Linear Regression agent."""
    
    # Load data
    print("Loading data...")
    qqq, spy, voo = load_data()
    split_data = split_data_individual(qqq, spy, voo)
    
    results = {}

    # Prepare training and test data for each stock
    for stock_name in ['QQQ', 'SPY', 'VOO']:
        print(f"\nTraining on {stock_name}...")
        
        train_df = split_data[stock_name]['train']
        test_df = split_data[stock_name]['test']
        
        # Prepare features and labels
        X_train, y_train = prepare_features_labels(train_df)
        X_test, y_test = prepare_features_labels(test_df)
        
        # Train the agent
        agent = LinearRegressionAgent()
        print(f"Training Linear Regression Agent for {stock_name}...")
        agent.train(X_train, y_train)
        
        # Evaluate the agent
        print(f"Evaluating model for {stock_name}...")
        score = agent.evaluate(X_test, y_test)
        
        # Get predictions and backtest
        predictions = agent.predict(X_test)
        portfolio_values = backtest_agent(test_df, predictions, initial_balance=3333.33)
        
        # Plot portfolio values
        plot_portfolio(portfolio_values, test_df, stock_name)
        
        # Calculate financial metrics
        metrics = calculate_metrics(portfolio_values, initial_balance=3333.33)
        
        # Print metrics
        print(f"\n{stock_name} Final Metrics:")
        print(f"  Total Return: {metrics['total_return']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"  Final Portfolio Value: ${metrics['final_portfolio_value']:.2f}")
        
        # Store results
        results[stock_name] = metrics

if __name__ == "__main__":
    main()