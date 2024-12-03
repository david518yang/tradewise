import pandas as pd

def load_data():
    # Reads and preprocesses the data
    qqq = pd.read_csv('../data/qqq.csv')
    spy = pd.read_csv('../data/spy.csv')
    voo = pd.read_csv('../data/voo.csv')
    
    for df in [qqq, spy, voo]:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    return qqq, spy, voo