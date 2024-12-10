# load_data.py
import pandas as pd
import numpy as np
import os
def load_data():
    """Load raw stock data.
    
    Returns:
        tuple: (qqq, spy, voo) DataFrames with raw price and volume data
    """
    try:
        # Read the raw data
        base_path = os.path.abspath(os.path.dirname(__file__))
        qqq = pd.read_csv(os.path.join(base_path, '../data/qqq.csv'))
        spy = pd.read_csv(os.path.join(base_path, '../data/spy.csv'))
        voo = pd.read_csv(os.path.join(base_path, '../data/voo.csv'))
        
        # Convert date and set index
        for df in [qqq, spy, voo]:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Keep the original code that preserves each dataset's full length
        qqq = qqq.resample('D').mean().interpolate(method='linear')
        spy = spy.resample('D').mean().interpolate(method='linear')
        voo = voo.resample('D').mean().interpolate(method='linear')
        
        return qqq, spy, voo
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not load stock data: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")
