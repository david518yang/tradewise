import os
import torch

def save_agent(agent, stock, path="./models/"):
    """Save the model weights for a DQN agent.
    
    Args:
        agent (DQNAgent): The trained DQN agent.
        stock (str): The stock identifier.
        path (str): The directory path to save the model.
    """
    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, f'{stock}_dqn_agent.pth')
    torch.save(agent.state_dict(), model_path)
    print(f"Saved final model for {stock} at {model_path}")

def load_agent(agent, stock, path="./models/"):
    """Load model weights for a DQN agent if available.
    
    Args:
        agent (DQNAgent): The DQN agent to load weights into.
        stock (str): The stock identifier.
        path (str): The directory path to load the model from.
        
    Returns:
        bool: True if weights were loaded, False otherwise.
    """
    model_path = os.path.join(path, f'{stock}_dqn_agent.pth')
    if os.path.exists(model_path):
        print(f"Loading existing weights for {stock} from {model_path}")
        agent.load_state_dict(torch.load(model_path, weights_only=True))
        return True
    return False

def load_training_data():
    """Load training data for stock trading.
    
    Returns:
        dict: A dictionary with training data for each stock.
    """
    from load_data import load_data
    from split import split_data_individual
    
    qqq, spy, voo = load_data()
    split_data = split_data_individual(qqq, spy, voo, test_size=0.2)
    
    train_data = {
        'QQQ': split_data['QQQ']['train'],
        'SPY': split_data['SPY']['train'],
        'VOO': split_data['VOO']['train']
    }
    return train_data

def load_test_data():
    """Load test data for stock trading.
    
    Returns:
        dict: A dictionary with test data for each stock.
    """
    from load_data import load_data
    from split import split_data_individual
    
    qqq, spy, voo = load_data()
    split_data = split_data_individual(qqq, spy, voo, test_size=0.2)
    
    test_data = {
        'QQQ': split_data['QQQ']['test'],
        'SPY': split_data['SPY']['test'],
        'VOO': split_data['VOO']['test']
    }
    return test_data