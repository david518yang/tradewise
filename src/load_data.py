import pandas as pd

def load_data():
    df = pd.read_csv('../data/aapl.csv')
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    return df