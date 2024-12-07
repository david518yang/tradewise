# load_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA100'] = df['Close'].rolling(window=100).mean()
        df['MA20_Volume'] = df['Volume'].rolling(window=20).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        roll_up = up.rolling(window=14).mean()
        roll_down = down.rolling(window=14).mean()
        RS = roll_up / roll_down
        df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
        df['RSI'] = df['RSI'].fillna(50)
        
        # Calculate MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        # Initialize flow series with proper dtype
        positive_flow = pd.Series(0.0, index=df.index, dtype=np.float64)
        negative_flow = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Calculate positive and negative money flow with proper dtype handling
        positive_mask = typical_price > typical_price.shift(1)
        negative_mask = typical_price < typical_price.shift(1)
        
        positive_flow[positive_mask] = money_flow[positive_mask].astype(np.float64)
        negative_flow[negative_mask] = money_flow[negative_mask].astype(np.float64)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        
        # Calculate MFI with proper handling of Series operations
        mf_ratio = positive_mf / negative_mf
        df['MFI'] = 100 - (100 / (1 + mf_ratio))
        
        # Handle edge cases
        df.loc[negative_mf == 0, 'MFI'] = 100  # When negative flow is 0 and positive flow exists
        df.loc[(negative_mf == 0) & (positive_mf == 0), 'MFI'] = 50  # When both flows are 0
        df['MFI'] = df['MFI'].fillna(50)
        
        # Calculate Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        # Calculate 20-day returns and volatility
        df['Returns_20d'] = df['Close'].pct_change(20).fillna(0)
        df['Volatility_20d'] = df['Returns'].rolling(20).std().fillna(0)
        
        # Calculate correlations (20-day rolling)
        if 'QQQ' in df.index.name:
            df['Corr_SPY'] = 0
            df['Corr_QQQ'] = 1
        elif 'SPY' in df.index.name:
            df['Corr_SPY'] = 1
            df['Corr_QQQ'] = 0
        else:
            df['Corr_SPY'] = 0.5  # Default correlation for VOO
            df['Corr_QQQ'] = 0.5
        
        # Drop NaN values resulting from rolling calculations
        df.dropna(inplace=True)
    
    # Align data to have the same number of rows
    min_length = min(len(qqq), len(spy), len(voo))
    qqq = qqq.iloc[-min_length:]
    spy = spy.iloc[-min_length:]
    voo = voo.iloc[-min_length:]
    
    # Initialize scalers
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler(feature_range=(0, 1))
    
    # Fit scalers on training data
    combined_states = np.vstack([
        qqq[['Close', 'Volume', 'RSI', 'MACD', 'Signal_Line']].values,
        spy[['Close', 'Volume', 'RSI', 'MACD', 'Signal_Line']].values,
        voo[['Close', 'Volume', 'RSI', 'MACD', 'Signal_Line']].values
    ])
    
    # Fit MinMaxScaler on 'Close' to ensure positive scaling
    scaler_minmax.fit(combined_states[:, 0].reshape(-1, 1))  # 'Close' is the first feature
    
    # Fit StandardScaler on other features
    scaler_standard.fit(combined_states[:, 1:])  # ['Volume', 'RSI', 'MACD', 'Signal_Line']
    
    # Apply scaling
    for df in [qqq, spy, voo]:
        # Scale 'Close' with MinMaxScaler
        df['Close_scaled'] = scaler_minmax.transform(df[['Close']].values)
        
        # Scale other features with StandardScaler
        df[['Volume_scaled', 'RSI_scaled', 'MACD_scaled', 'Signal_Line_scaled']] = scaler_standard.transform(df[['Volume', 'RSI', 'MACD', 'Signal_Line']].values)
        
        # Select only scaled features for the environment
        df = df[['Close_scaled', 'Volume_scaled', 'RSI_scaled', 'MACD_scaled', 'Signal_Line_scaled']]
    
    return qqq, spy, voo
