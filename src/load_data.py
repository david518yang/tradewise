import pandas as pd
import numpy as np

def load_data():
    # Reads and preprocesses the data
    qqq = pd.read_csv('../data/qqq.csv')
    spy = pd.read_csv('../data/spy.csv')
    voo = pd.read_csv('../data/voo.csv')
    
    for df in [qqq, spy, voo]:
        # Convert date and set index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change().fillna(0)
        
        # Calculate Moving Averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(window=14).mean()
        roll_down = down.rolling(window=14).mean()
        RS = roll_up / roll_down
        df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
        df['RSI'] = df['RSI'].fillna(50)  # Fill NaN values
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Returns over past 5 and 10 days
        df['Return_5D'] = df['Close'].pct_change(periods=5).fillna(0)
        df['Return_10D'] = df['Close'].pct_change(periods=10).fillna(0)
        
        # Price Ratios
        df['52_Week_High'] = df['Close'].rolling(window=252).max()
        df['52_Week_Low'] = df['Close'].rolling(window=252).min()
        df['Price_to_High'] = df['Close'] / df['52_Week_High']
        df['Price_to_Low'] = df['Close'] / df['52_Week_Low']
        df['Price_to_High'] = df['Price_to_High'].fillna(1)
        df['Price_to_Low'] = df['Price_to_Low'].fillna(1)
        
        # Trading Volume (normalized)
        df['Volume_Normalized'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['Volume_Normalized'] = df['Volume_Normalized'].fillna(1)
        
        # Drop NaN values resulting from rolling calculations
        df.dropna(inplace=True)
    
    return qqq, spy, voo
