import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from load_data import load_data
from split import split_data

def prepare_features(df):
    features = df[['Returns', 'MA5', 'MA20', 'RSI', 'MACD', 'Signal_Line', 
                  'Return_5D', 'Return_10D', 'Price_to_High', 'Price_to_Low', 
                  'Volume_Normalized']]
    return features

def prepare_labels(df):
    df['Label'] = 0  # Default to hold
    df.loc[df['Returns'] > 0.01, 'Label'] = 1  # Buy if returns > 1%
    df.loc[df['Returns'] < -0.01, 'Label'] = -1  # Sell if returns < -1%
    return df['Label']

def train_classifier(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_performance(clf, test_df, initial_balance=10000):
    features = prepare_features(test_df)
    predictions = clf.predict(features)
    
    balance = initial_balance
    shares = 0
    transaction_fee = 0.001  # 0.1% transaction fee
    
    portfolio_values = []
    buy_and_hold_values = []
    
    initial_price = test_df['Close'].iloc[0]
    buy_and_hold_shares = initial_balance / initial_price
    
    for i in range(len(test_df)):
        current_price = test_df['Close'].iloc[i]
        
        # Track buy and hold strategy
        buy_and_hold_value = buy_and_hold_shares * current_price
        buy_and_hold_values.append(buy_and_hold_value)
        
        # Execute trading strategy
        if predictions[i] == 1 and balance > 0:  # Buy
            shares_to_buy = balance / (current_price * (1 + transaction_fee))
            shares += shares_to_buy
            balance -= shares_to_buy * current_price * (1 + transaction_fee)
            
        elif predictions[i] == -1 and shares > 0:  # Sell
            balance += shares * current_price * (1 - transaction_fee)
            shares = 0
            
        # Calculate current portfolio value
        portfolio_value = balance + (shares * current_price)
        portfolio_values.append(portfolio_value)
    
    # Calculate performance metrics
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
    
    print(f"\nPerformance Metrics:")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
    print(f"Outperformance vs Buy & Hold: {total_return - buy_hold_return:.2f}%")
    
    return portfolio_values, buy_and_hold_values

def main():
    # Load and split data
    qqq, spy, voo = load_data()
    (qqq_train, qqq_test), (spy_train, spy_test), (voo_train, voo_test) = split_data(qqq, spy, voo)

    # Prepare training data
    X_train = pd.concat([prepare_features(qqq_train), prepare_features(spy_train), prepare_features(voo_train)])
    y_train = pd.concat([prepare_labels(qqq_train), prepare_labels(spy_train), prepare_labels(voo_train)])

    # Train classifier
    print("Training classifier...")
    clf = train_classifier(X_train, y_train)

    # Evaluate on each test set
    print("\nQQQ Performance:")
    qqq_portfolio, qqq_buy_hold = evaluate_performance(clf, qqq_test)
    
    print("\nSPY Performance:")
    spy_portfolio, spy_buy_hold = evaluate_performance(clf, spy_test)
    
    print("\nVOO Performance:")
    voo_portfolio, voo_buy_hold = evaluate_performance(clf, voo_test)

if __name__ == "__main__":
    main()
