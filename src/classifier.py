"""Random Forest Classifier for stock trading."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from load_data import load_data
from split import split_data
import matplotlib.pyplot as plt

def prepare_features(df):
    """Prepare features for classification.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Features dataframe
    """
    features = df[['Returns', 'MA5', 'MA20', 'RSI', 'MACD', 'Signal_Line', 
                  'Return_5D', 'Return_10D', 'Price_to_High', 'Price_to_Low', 
                  'Volume_Normalized']]
    return features

def prepare_labels(df):
    """Prepare labels for classification.
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.Series: Labels series
    """
    df['Label'] = 0
    df.loc[df['Returns'] > 0.01, 'Label'] = 1
    df.loc[df['Returns'] < -0.01, 'Label'] = -1
    return df['Label']

def train_classifier(X_train, y_train):
    """Train Random Forest classifier.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
    
    Returns:
        RandomForestClassifier: Trained classifier
    """
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_performance(clf, test_df, initial_balance=3333.33):
    """Evaluate classifier performance.
    
    Args:
        clf: Trained classifier
        test_df (pd.DataFrame): Test dataframe
        initial_balance (float): Initial portfolio balance
    
    Returns:
        tuple: Portfolio values and buy-and-hold values
    """
    features = prepare_features(test_df)
    predictions = clf.predict(features)
    
    balance = initial_balance
    shares = 0
    transaction_fee = 0.001
    
    portfolio_values = []
    buy_and_hold_values = []
    
    initial_price = test_df['Close'].iloc[0]
    buy_and_hold_shares = initial_balance / initial_price
    
    for i in range(len(test_df)):
        current_price = test_df['Close'].iloc[i]
        
        buy_and_hold_value = buy_and_hold_shares * current_price
        buy_and_hold_values.append(buy_and_hold_value)
        
        if predictions[i] == 1 and balance > 0:
            shares_to_buy = balance / (current_price * (1 + transaction_fee))
            shares += shares_to_buy
            balance -= shares_to_buy * current_price * (1 + transaction_fee)
            
        elif predictions[i] == -1 and shares > 0:
            balance += shares * current_price * (1 - transaction_fee)
            shares = 0
            
        portfolio_value = balance + (shares * current_price)
        portfolio_values.append(portfolio_value)
    
    final_portfolio_value = portfolio_values[-1]
    final_buy_hold_value = buy_and_hold_values[-1]
    
    total_return = (final_portfolio_value - initial_balance) / initial_balance * 100
    buy_hold_return = (final_buy_hold_value - initial_balance) / initial_balance * 100
    
    max_drawdown = 0
    peak = portfolio_values[0]
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    
    print("\nPerformance Metrics:")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"Outperformance vs Buy & Hold: {total_return - buy_hold_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    return portfolio_values, buy_and_hold_values

def plot_results(test_df, portfolio_values, predictions, title):
    """Plot trading results.
    
    Args:
        test_df (pd.DataFrame): Test dataframe
        portfolio_values (list): Portfolio values
        predictions (list): Trading predictions
        title (str): Plot title
    
    Returns:
        matplotlib.figure.Figure: Plot figure
    """
    steps = range(len(test_df))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(title)
    
    ax1.plot(steps, portfolio_values, label='Portfolio Value', color='blue')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Trading Steps')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    
    stock_prices = test_df['Close'].values
    ax2.plot(steps, stock_prices, label='Stock Price', color='black', alpha=0.7)
    
    buys = [i for i, pred in enumerate(predictions) if pred == 1]
    sells = [i for i, pred in enumerate(predictions) if pred == -1]
    
    if buys:
        ax2.scatter(buys, stock_prices[buys], color='green', marker='^', 
                   label='Buy', s=100, alpha=0.7)
    if sells:
        ax2.scatter(sells, stock_prices[sells], color='red', marker='v', 
                   label='Sell', s=100, alpha=0.7)
    
    ax2.set_title('Stock Price with Trading Signals')
    ax2.set_xlabel('Trading Steps')
    ax2.set_ylabel('Stock Price ($)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run the classifier."""
    qqq, spy, voo = load_data()
    (qqq_train, qqq_test), (spy_train, spy_test), (voo_train, voo_test) = split_data(qqq, spy, voo)

    X_train = pd.concat([prepare_features(qqq_train), prepare_features(spy_train), prepare_features(voo_train)])
    y_train = pd.concat([prepare_labels(qqq_train), prepare_labels(spy_train), prepare_labels(voo_train)])

    print("Training classifier...")
    clf = train_classifier(X_train, y_train)

    etfs = [('QQQ', qqq_test), ('SPY', spy_test), ('VOO', voo_test)]
    
    for name, test_df in etfs:
        print(f"\n{name} Performance:")
        features = prepare_features(test_df)
        predictions = clf.predict(features)
        portfolio_values, _ = evaluate_performance(clf, test_df)
        
        fig = plot_results(test_df, portfolio_values, predictions, f'{name} Trading Results')
        plt.savefig(f'{name}_trading_results.png')
        plt.close(fig)

if __name__ == "__main__":
    main()
