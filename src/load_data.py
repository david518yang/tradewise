import pandas as pd

def load_data():
    # Reads and preprocesses the data
    aapl = pd.read_csv('../data/aapl.csv')
    googl = pd.read_csv('../data/googl.csv')
    msft = pd.read_csv('../data/msft.csv')
    
    for df in [aapl, googl, msft]:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    return aapl, googl, msft