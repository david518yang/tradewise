from sklearn.model_selection import train_test_split

def split_data(aapl, googl, msft, test_size=0.2, random_state=42):
    def split_df(df, test_size):
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test
    
    # split the data
    aapl_train, aapl_test = split_df(aapl, test_size)
    googl_train, googl_test = split_df(googl, test_size)
    msft_train, msft_test = split_df(msft, test_size)

    return (aapl_train, aapl_test), (googl_train, googl_test), (msft_train, msft_test)
