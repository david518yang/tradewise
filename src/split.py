from sklearn.model_selection import train_test_split

def split_data(qqq, spy, voo, test_size=0.2, random_state=42):
    def split_df(df, test_size):
        split_idx = int(len(df) * (1 - test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test
    
    # split the data
    qqq_train, qqq_test = split_df(qqq, test_size)
    spy_train, spy_test = split_df(spy, test_size)
    voo_train, voo_test = split_df(voo, test_size)

    return (qqq_train, qqq_test), (spy_train, spy_test), (voo_train, voo_test)
