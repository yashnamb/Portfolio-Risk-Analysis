import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =============================================
# 1. Data Download and Preparation
# =============================================

def download_sp500_data():
    """Download S&P 500 tickers and company information."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    securities = df["Security"].tolist()
    sectors = df["GICS Sector"].tolist()
    return list(zip(tickers, securities, sectors))

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data for given tickers."""
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    data = yf.download(tickers, start=start_str, end=end_str, progress=False)
    if 'Close' not in data.columns:
        return None
    return data['Close']

# Example usage:
# ticker_security_pairs = download_sp500_data()
# selected_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
# start_date = datetime(2020, 1, 1)
# end_date = datetime(2023, 12, 31)
# historical_data = fetch_stock_data(selected_tickers, start_date, end_date)

# =============================================
# 2. Portfolio Performance Metrics
# =============================================

def calculate_portfolio_metrics(historical_data, investments):
    """Calculate key portfolio performance metrics."""
    returns = historical_data.pct_change().dropna()
    total_investment = sum(investments.values())
    weights = {ticker: amount/total_investment for ticker, amount in investments.items()}
    
    # Calculate portfolio returns
    portfolio_returns = pd.Series(index=returns.index)
    for ticker in investments.keys():
        portfolio_returns += returns[ticker] * weights[ticker]
    
    # Calculate metrics
    annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - 0.02) / annual_volatility  # Assuming 2% risk-free rate
    
    # Calculate drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    return {
        'portfolio_returns': portfolio_returns,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'weights': weights
    }

# Example usage:
# investments = {'AAPL': 2500, 'MSFT': 2500, 'GOOGL': 2500, 'AMZN': 2500}
# portfolio_metrics = calculate_portfolio_metrics(historical_data, investments)

# =============================================
# 3. Risk Analysis
# =============================================

def calculate_risk_metrics(portfolio_data, historical_data):
    """Calculate comprehensive risk metrics."""
    returns = historical_data.pct_change().dropna()
    weights = portfolio_data['weights']
    
    # Calculate Value at Risk
    var_95 = np.percentile(portfolio_data['portfolio_returns'], 5)
    var_99 = np.percentile(portfolio_data['portfolio_returns'], 1)
    
    # Calculate correlation matrix
    correlation_matrix = returns.corr()
    
    # Calculate portfolio volatility
    cov_matrix = returns.cov() * 252
    weights_array = np.array([weights[ticker] for ticker in returns.columns])
    portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
    
    return {
        'var_95': var_95,
        'var_99': var_99,
        'correlation_matrix': correlation_matrix,
        'portfolio_volatility': portfolio_volatility
    }

# Example usage:
# risk_metrics = calculate_risk_metrics(portfolio_metrics, historical_data)

# =============================================
# 4. Monte Carlo Simulation
# =============================================

def run_monte_carlo_simulation(portfolio_returns, initial_investment, n_simulations=1000, n_days=252):
    """Run Monte Carlo simulation for portfolio forecasting."""
    mean_return = portfolio_returns.mean()
    std_return = portfolio_returns.std()
    
    simulations = np.zeros((n_days, n_simulations))
    simulations[0, :] = initial_investment
    
    for simulation in range(n_simulations):
        daily_returns = np.random.normal(mean_return, std_return, n_days)
        for day in range(1, n_days):
            simulations[day, simulation] = simulations[day-1, simulation] * (1 + daily_returns[day])
    
    percentiles = {
        5: np.percentile(simulations[-1, :], 5),
        25: np.percentile(simulations[-1, :], 25),
        50: np.percentile(simulations[-1, :], 50),
        75: np.percentile(simulations[-1, :], 75),
        95: np.percentile(simulations[-1, :], 95)
    }
    
    return simulations, percentiles

# Example usage:
# simulations, percentiles = run_monte_carlo_simulation(
#     portfolio_metrics['portfolio_returns'],
#     initial_investment=10000,
#     n_simulations=1000
# )

# =============================================
# 5. Stock Clustering
# =============================================

def perform_stock_clustering(historical_data, n_clusters=5):
    """Perform K-means clustering on stocks based on returns and volatility."""
    returns = historical_data.pct_change().dropna()
    annual_returns = (1 + returns.mean()) ** 252 - 1
    annual_volatility = returns.std() * np.sqrt(252)
    
    features = pd.DataFrame({
        'Return': annual_returns * 100,
        'Volatility': annual_volatility * 100
    }).dropna()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    features['Cluster'] = clusters
    features['Cluster'] = features['Cluster'].astype(str)
    
    return features, kmeans

# Example usage:
# cluster_features, kmeans_model = perform_stock_clustering(historical_data, n_clusters=5)

# =============================================
# 6. Visualization Functions
# =============================================

def plot_portfolio_performance(portfolio_returns):
    """Create portfolio performance plot."""
    cum_returns = (1 + portfolio_returns).cumprod()
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=cum_returns,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_risk_return(portfolio_data, historical_data):
    """Create risk-return scatter plot."""
    returns = historical_data.pct_change().dropna()
    annual_returns = (1 + returns.mean()) ** 252 - 1
    annual_volatility = returns.std() * np.sqrt(252)
    
    df = pd.DataFrame({
        'Asset': returns.columns,
        'Return': annual_returns * 100,
        'Volatility': annual_volatility * 100
    })
    
    fig = px.scatter(
        df,
        x='Volatility',
        y='Return',
        color='Asset',
        title='Risk-Return Profile',
        labels={
            'Volatility': 'Annualized Volatility (%)',
            'Return': 'Annualized Return (%)'
        }
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_clusters(features, kmeans):
    """Create cluster visualization plot."""
    fig = px.scatter(
        features,
        x='Volatility',
        y='Return',
        color='Cluster',
        title='Stock Clustering by Return and Volatility',
        labels={
            'Volatility': 'Annualized Volatility (%)',
            'Return': 'Annualized Return (%)'
        }
    )
    
    centers = kmeans.cluster_centers_
    fig.add_trace(
        go.Scatter(
            x=centers[:, 1],
            y=centers[:, 0],
            mode='markers',
            marker=dict(
                color='black',
                size=12,
                symbol='x'
            ),
            name='Cluster Centers'
        )
    )
    
    fig.update_layout(
        height=600,
        template='plotly_white'
    )
    
    return fig

# Example usage:
# performance_plot = plot_portfolio_performance(portfolio_metrics['portfolio_returns'])
# risk_return_plot = plot_risk_return(portfolio_metrics, historical_data)
# cluster_plot = plot_clusters(cluster_features, kmeans_model)

# =============================================
# 7. Example Usage
# =============================================

if __name__ == "__main__":
    # Download S&P 500 data
    ticker_security_pairs = download_sp500_data()
    selected_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    # Set date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Fetch historical data
    historical_data = fetch_stock_data(selected_tickers, start_date, end_date)
    
    # Define investments
    investments = {'AAPL': 2500, 'MSFT': 2500, 'GOOGL': 2500, 'AMZN': 2500}
    
    # Calculate portfolio metrics
    portfolio_metrics = calculate_portfolio_metrics(historical_data, investments)
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(portfolio_metrics, historical_data)
    
    # Run Monte Carlo simulation
    simulations, percentiles = run_monte_carlo_simulation(
        portfolio_metrics['portfolio_returns'],
        initial_investment=10000
    )
    
    # Perform stock clustering
    cluster_features, kmeans_model = perform_stock_clustering(historical_data)
    
    # Print results
    print("\nPortfolio Metrics:")
    print(f"Annual Return: {portfolio_metrics['annual_return']*100:.2f}%")
    print(f"Annual Volatility: {portfolio_metrics['annual_volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {portfolio_metrics['max_drawdown']*100:.2f}%")
    
    print("\nRisk Metrics:")
    print(f"95% VaR: {risk_metrics['var_95']*100:.2f}%")
    print(f"99% VaR: {risk_metrics['var_99']*100:.2f}%")
    print(f"Portfolio Volatility: {risk_metrics['portfolio_volatility']*100:.2f}%")
    
    print("\nMonte Carlo Simulation Results:")
    print(f"5th Percentile: ${percentiles[5]:.2f}")
    print(f"50th Percentile: ${percentiles[50]:.2f}")
    print(f"95th Percentile: ${percentiles[95]:.2f}") 