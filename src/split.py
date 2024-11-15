from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.2, random_state=42):
    split_idx = int(len(df) * (1 - test_size))
    
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    return train, test
