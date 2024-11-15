import pandas as pd
from datetime import datetime
import pandas_market_calendars as mcal

def load_stock_data(filename='../data/aapl.csv', market='NYSE'):
    """
    Loads stock data from a CSV file and organizes it by date
    
    Args:
        filename (str): Path to the CSV file containing stock data
        market (str): Market to use for trading calendar (NYSE, NASDAQ, etc.)
        
    Returns:
        dict: Dictionary with dates as keys and daily stock data as values
    """
    df = pd.read_csv(filename)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    daily_data = {}
    
    for _, row in df.iterrows():
        date = row['Date']
        daily_data[date] = {
            'Open': row['Open'] if 'Open' in df.columns else None,
            'High': row['High'] if 'High' in df.columns else None,
            'Low': row['Low'] if 'Low' in df.columns else None,
            'Close': row['Close'] if 'Close' in df.columns else None,
            'Volume': row['Volume'] if 'Volume' in df.columns else None
        }
    
    return daily_data

def is_trading_day(date, market='NYSE'):
    """Check if given date is a trading day"""
    nyse = mcal.get_calendar(market)
    schedule = nyse.schedule(start_date=date, end_date=date)
    return not schedule.empty

stock_data = load_stock_data()
print(f"Loaded {len(stock_data)} days of stock data")
start_date = datetime(1984, 9, 18)
for i in range(5):
    current_date = start_date + pd.Timedelta(days=i)
    if current_date in stock_data and is_trading_day(current_date):
        print(f"\nStock data for {current_date.date()} (Trading Day):")
        print(stock_data[current_date])
    else:
        reason = "Weekend" if current_date.weekday() >= 5 else "Holiday"
        print(f"\nNo data available for {current_date.date()} ({reason})")