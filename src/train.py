from src.data.load_data import load_data
from sklearn.model_selection import train_test_split

data = load_data()

def split_data(data): 
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('target', axis=1), 
        data['target'],
        test_size=0.2,
        random_state=42
    )
    return X_train, X_test, y_train, y_test
