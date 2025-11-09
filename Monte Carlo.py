import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Define the stocks and period
stocks = ['GOOGL', 'MSFT', 'TSLA', 'AMZN']
start_date = '2023-01-01'
end_date = '2025-11-08'

# Step 2: Download historical stock data
data = yf.download(stocks, start=start_date, end=end_date)
data = data['Close']  # Use 'Close' instead of 'Adj Close'


# Step 3: Calculate daily returns
returns = data.pct_change().dropna()

# Step 4: Set up Monte Carlo simulation parameters
num_simulations = 1000
num_days = 252  # One trading year

# Step 5: Run simulation
simulation_results = {}

for stock in stocks:
    last_price = data[stock][-1]
    daily_mean = returns[stock].mean()
    daily_std = returns[stock].std()

    simulations = np.zeros((num_days, num_simulations))

    for i in range(num_simulations):
        prices = [last_price]
        for _ in range(num_days):
            # Random walk formula
            price = prices[-1] * np.exp(np.random.normal(daily_mean, daily_std))
            prices.append(price)
        simulations[:, i] = prices[1:]

    simulation_results[stock] = simulations

# Step 6: Plot results
plt.figure(figsize=(15, 10))
for stock in stocks:
    plt.subplot(2, 2, stocks.index(stock) + 1)
    plt.plot(simulation_results[stock], color='grey', alpha=0.1)
    plt.title(f'Monte Carlo Simulation for {stock}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)

plt.tight_layout()
plt.show()