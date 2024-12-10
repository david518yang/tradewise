# Introduction 
In this project, we explored the efficacy of a reinforcement learning approach to 
trading stocks and Exchange-Traded Funds (ETFs) using Q-learning. We developed individual 
agents that learned from historical market data for several stocks and ETFs including 
AAPL (Apple), GOOG (Google), MSFT (Microsoft), QQQ (Nasdaq), SPY (S&P 500), and 
VOO (Vanguard S&P 500), in order to learn what to do in certain market conditions, 
develop an optimized trading strategy, and maximize portfolio returns. 


## Setup 
To run this project, you must begin by installing the necessary libraries:
```
pip3 install -r requirements.txt
```

For ease of use and speed, we have included pretrained data files for each stock to test with.
If you would like to train the model yourself, delete the three files in models.
Warning: Running the deepq.py program typically takes 2 hours for completion if you do not
use the pretrained data.

Afterwards, navigate into the `src` directory:
```
cd src
```

Depending on which model you want to run, either enter:
```
python3 deepq.py
```

Or 
```
python3 qlearn.py
```

Let the agent train and enjoy the results!