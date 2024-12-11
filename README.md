# Tradewise

**Tradewise** is an exploratory project leveraging reinforcement learning to develop and optimize trading strategies for stocks and Exchange-Traded Funds (ETFs). Using Q-learning, we trained individual agents on historical market data to adapt to various market conditions and maximize portfolio returns. 

The project focuses on several major stocks and ETFs, including:
- **AAPL** (Apple)
- **GOOG** (Google)
- **MSFT** (Microsoft)
- **QQQ** (Nasdaq)
- **SPY** (S&P 500)
- **VOO** (Vanguard S&P 500)

By learning from past trends and behaviors, Tradewise provides a framework to evaluate the efficacy of reinforcement learning in financial markets.

---

## 🚀 Features
- **Q-learning** and **Deep Q-learning** implementations for  financial trading.
- **Random Forest Classifier** to analyze and predict market behavior using a supervised learning approach.
- **Linear Regression Model** for exploring and forecasting price trends based on historical data.
- Pretrained models for quick testing and evaluation (DQN).
- Capability to train custom models on historical market data.
- Modular codebase.

---

## 🛠️ Before You Begin

For your convenience, pretrained model data files are included in the `models` folder. These files enable immediate testing for the Deep Q Agent without the need to train the models from scratch.

If you prefer to train the models yourself:
1. Delete the existing files in the `models` folder.
2. Note that training can take approximately **2 hours** per session on a PC with a 3070.

---

## ⚙️ Setup Instructions

To get started with Tradewise, follow these steps:

### Step 1: Install Dependencies
Ensure you have Python 3 installed, then use the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Navigate to the Source Directory
Move into the project's source directory:

```bash
cd src
```

### Step 3: Run the Model
Choose which reinforcement learning model to execute:

For Random Forest Classification:

```bash
python classifier.py
```

For Linear Regression:

```bash
python classifier.py
```

For Deep Q-learning:

```bash
python deepq.py
```

For Q-learning:

```bash
python qlearn.py
```

### Step 4: Training and Evaluation

If running a fresh training session, the agent will train on the selected stock/ETF data.
Once training completes, the agents will be evaluated on a testing set, plot their results, and output their final returns and statistics in the terminal.
