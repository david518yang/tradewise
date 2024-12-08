from sklearn.model_selection import train_test_split
import pandas as pd

def calculate_features(df):
    """Calculate all required features for a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with at least Close and Volume columns
        
    Returns:
        pd.DataFrame: DataFrame with all features calculated
    """
    df = df.copy()
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA20_Volume'] = df['Volume'].rolling(window=20).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Handle NaN values
    df['Returns'] = df['Returns'].fillna(0)
    df['RSI'] = df['RSI'].fillna(50)
    df = df.ffill().bfill()
    
    # Add scaled versions of features
    df['Close_scaled'] = (df['Close'] - df['Close'].mean()) / df['Close'].std()
    df['Volume_scaled'] = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()
    df['RSI_scaled'] = df['RSI'] / 100.0 
    df['MACD_scaled'] = (df['MACD'] - df['MACD'].mean()) / df['MACD'].std()
    df['Signal_Line_scaled'] = (df['Signal_Line'] - df['Signal_Line'].mean()) / df['Signal_Line'].std()
    
    return df

def split_data_individual(qqq, spy, voo, test_size=0.2):
    """Split each dataset independently using its full duration.
    
    Args:
        qqq (pd.DataFrame): QQQ data from 1999
        spy (pd.DataFrame): SPY data
        voo (pd.DataFrame): VOO data from 2010
        test_size (float): Proportion of data to use for testing (default: 0.2)
    
    Returns:
        dict: Dictionary containing train/test splits and date ranges for each ETF
    """
    # Calculate features for each dataset
    qqq = calculate_features(qqq)
    spy = calculate_features(spy)
    voo = calculate_features(voo)
    
    def split_df(df, test_size):
        """Split a DataFrame while preserving all features."""
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        return train, test, test.index[0]  # Return test start date
    
    # Split each dataset independently using its full length
    qqq_train, qqq_test, qqq_test_start = split_df(qqq, test_size)
    spy_train, spy_test, spy_test_start = split_df(spy, test_size)
    voo_train, voo_test, voo_test_start = split_df(voo, test_size)
    
    # Create a unified test period dataframe with all dates
    all_test_dates = pd.DatetimeIndex(sorted(set(
        list(qqq_test.index) + 
        list(spy_test.index) + 
        list(voo_test.index)
    )))
    
    return {
        'QQQ': {
            'train': qqq_train,
            'test': qqq_test,
            'test_start': qqq_test_start
        },
        'SPY': {
            'train': spy_train,
            'test': spy_test,
            'test_start': spy_test_start
        },
        'VOO': {
            'train': voo_train,
            'test': voo_test,
            'test_start': voo_test_start
        },
        'all_test_dates': all_test_dates
    }

def create_unified_test_data(split_data):
    """Create unified test datasets where missing data is filled with NaN.
    
    Args:
        split_data (dict): Output from split_data_individual
        
    Returns:
        dict: Dictionary with unified test data for each ETF
    """
    all_dates = split_data['all_test_dates']
    unified_test = {}
    
    for etf in ['QQQ', 'SPY', 'VOO']:
        # Create a new dataframe with all dates
        test_data = split_data[etf]['test']
        unified = pd.DataFrame(index=all_dates)
        
        # Merge with actual test data
        unified = unified.join(test_data)
        
        # Fill missing values appropriately
        # Prices and other indicators will be NaN before the ETF's start date
        unified_test[etf] = unified
    
    return unified_test
