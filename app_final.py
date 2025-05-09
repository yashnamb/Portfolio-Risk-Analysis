import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import datetime
from datetime import timedelta
import statsmodels.api as sm

# App configuration
st.set_page_config(
    page_title="Financial Portfolio Analyzer",
    page_icon=":)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    [data-theme="light"] {
        --tab-bg: #f0f0f0;
        --tab-text: #333333;
        --card-bg: #ffffff;
        --card-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        border-radius: 4px;
        padding: 8px 12px;
        background-color: var(--tab-bg);
        color: var(--tab-text);
        font-weight: normal;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    .card {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: var(--card-shadow);
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    h1 {
        color: #2E7D32;
        font-size: 2rem;
    }
    h2 {
        color: #388E3C;
        font-size: 1.5rem;
    }
    h3 {
        color: #43A047;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Utility Functions
def format_currency(value):
    """Format a numeric value as currency."""
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    elif abs_value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    elif abs_value >= 1_000:
        return f"{value/1_000:.2f}K"
    else:
        return f"{value:.2f}"

def format_percentage(value):
    """Format a numeric value as percentage."""
    return f"{value:.2f}"

@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    securities = df["Security"].tolist()
    sectors = df["GICS Sector"].tolist()
    return list(zip(tickers, securities, sectors))

# Data Functions
@st.cache_data(ttl=3600)
def fetch_stock_data(tickers, start_date, end_date, include_sp500=True):
    """
    Fetch historical stock data for the given tickers and date range.
    Optionally include S&P 500 for comparison.
    """
    if not tickers:
        st.error("No tickers provided")
        return None
        
    # Convert dates to string format
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Add S&P 500 for market comparison if requested
    if include_sp500:
        tickers = tickers + ['^GSPC']
    
    # Fetch data
    data = yf.download(tickers, start=start_str, end=end_str, progress=False)
    
    if data.empty:
        st.error("No data returned from Yahoo Finance. Please check your ticker symbols and try again.")
        return None
        
    # Handle single ticker case
    if isinstance(data, pd.Series) or (len(tickers) == 1 and isinstance(data, pd.DataFrame) and 'Close' not in data.columns):
        data = yf.download(tickers, start=start_str, end=end_str, progress=False)
        if data.empty:
            st.error(f"No data found for ticker {tickers[0]}. Please check the ticker symbol and try again.")
            return None
        prices_df = pd.DataFrame(data['Close'])
        prices_df.columns = tickers
    else:
        if 'Close' not in data.columns:
            st.error("Failed to fetch adjusted close prices. Please try again later.")
            return None
        prices_df = data['Close']
    
    # Check for missing tickers
    missing_tickers = set(tickers) - set(prices_df.columns)
    if missing_tickers:
        st.warning(f"Could not fetch data for the following tickers: {', '.join(missing_tickers)}")
    
    # Fill missing values
    prices_df = prices_df.ffill().bfill()
    
    if prices_df.empty:
        st.error("No valid data available after processing. Please check your date range and try again.")
        return None
        
    return prices_df

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_live_prices(tickers):
    """Get real-time prices for the given tickers."""
    prices = {}
    
    try:
        data = yf.download(tickers, period="1d", progress=False)
        
        if len(tickers) == 1:
            if 'Close' in data:
                ticker = tickers[0]
                prices[ticker] = data['Close'][-1]
            else:
                ticker = tickers[0]
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty and 'Close' in hist:
                    prices[ticker] = hist['Close'][-1]
                else:
                    prices[ticker] = 0
        else:
            for ticker in tickers:
                if ticker in data['Close']:
                    prices[ticker] = data['Close'][ticker][-1]
                else:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d")
                    if not hist.empty and 'Close' in hist:
                        prices[ticker] = hist['Close'][-1]
                    else:
                        prices[ticker] = 0
    
    except Exception as e:
        st.warning(f"Error fetching live prices: {e}")
        for ticker in tickers:
            prices[ticker] = 0
    
    return prices

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_stock_info(tickers):
    """Get detailed information about stocks."""
    stock_info = {}
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            stock_info[ticker] = {
                'name': info.get('shortName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'beta': info.get('beta', None),
                'year_high': info.get('fiftyTwoWeekHigh', 0),
                'year_low': info.get('fiftyTwoWeekLow', 0)
            }
        except Exception as e:
            stock_info[ticker] = {
                'name': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'pe_ratio': None,
                'dividend_yield': 0,
                'beta': None,
                'year_high': 0,
                'year_low': 0
            }
    
    return stock_info

def calculate_portfolio_performance(historical_data, investments):
    """Calculate portfolio performance metrics."""
    if historical_data is None or historical_data.empty:
        st.error("No historical data provided")
        return None
        
    if not investments:
        st.error("No investments provided")
        return None
        
    data = historical_data.copy()
    portfolio_tickers = list(investments.keys())
    market_ticker = '^GSPC'
    
    for ticker in portfolio_tickers:
        if ticker not in data.columns:
            st.warning(f"No historical data found for {ticker}. Excluding from analysis.")
            portfolio_tickers.remove(ticker)
    
    if not portfolio_tickers:
        st.error("No valid tickers found for analysis.")
        return None
    
    portfolio_data = data[portfolio_tickers]
    returns = portfolio_data.pct_change().dropna()
    returns = returns.clip(lower=-1, upper=1)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        st.error("No valid returns data available after processing")
        return None
    
    total_investment = sum(investments.values())
    weights = {ticker: investments[ticker] / total_investment for ticker in portfolio_tickers}
    
    weighted_returns = pd.DataFrame()
    for ticker in portfolio_tickers:
        if ticker in returns.columns:
            weighted_returns[ticker] = returns[ticker] * weights[ticker]
    
    if weighted_returns.empty:
        st.error("No valid weighted returns calculated")
        return None
    
    portfolio_returns = weighted_returns.sum(axis=1)
    portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
    
    if market_ticker in data.columns:
        market_returns = data[market_ticker].pct_change().dropna()
        market_returns = market_returns.reindex(portfolio_returns.index).fillna(0)
        market_cumulative_returns = (1 + market_returns).cumprod()
    else:
        st.warning("Market data (S&P 500) not available for comparison")
        market_returns = pd.Series(0, index=portfolio_returns.index)
        market_cumulative_returns = pd.Series(1, index=portfolio_returns.index)
    
    trading_days_per_year = 252
    portfolio_annual_return = (1 + portfolio_returns.mean()) ** trading_days_per_year - 1
    portfolio_annual_vol = portfolio_returns.std() * np.sqrt(trading_days_per_year)
    risk_free_rate = 0.02
    portfolio_sharpe_ratio = (portfolio_annual_return - risk_free_rate) / portfolio_annual_vol

    # --- Beta calculation using OLS regression ---
    if not market_returns.empty and not portfolio_returns.empty:
        aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        aligned.columns = ["Portfolio_Return", "Market_Return"]
        X = sm.add_constant(aligned["Market_Return"])
        y = aligned["Portfolio_Return"]
        model = sm.OLS(y, X).fit()
        beta = model.params["Market_Return"]
    else:
        beta = np.nan
    # --------------------------------------------

    result = {
        'portfolio_returns': portfolio_returns,
        'portfolio_cumulative_returns': portfolio_cumulative_returns,
        'market_returns': market_returns,
        'market_cumulative_returns': market_cumulative_returns,
        'portfolio_annual_return': portfolio_annual_return,
        'portfolio_annual_vol': portfolio_annual_vol,
        'portfolio_sharpe_ratio': portfolio_sharpe_ratio,
        'portfolio_beta': beta,
        'weights': weights
    }
    
    return result 

# Technical Analysis Functions
def calculate_moving_averages(prices_df, window_short=20, window_long=50):
    """Calculate short and long-term moving averages."""
    short_ma = prices_df.rolling(window=window_short).mean()
    long_ma = prices_df.rolling(window=window_long).mean()
    return short_ma, long_ma

def calculate_bollinger_bands(prices_df, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    middle_band = prices_df.rolling(window=window).mean()
    std_dev = prices_df.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    return upper_band, middle_band, lower_band

def calculate_rsi(prices_df, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = prices_df.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi 

# Analysis Functions
def perform_kmeans_clustering(historical_data, tickers, n_clusters=3):
    """Perform K-means clustering on stocks based on their returns."""
    valid_tickers = [ticker for ticker in tickers if ticker in historical_data.columns]
    
    if len(valid_tickers) < 2:
        return {
            'cluster_plot': None,
            'cluster_characteristics': pd.DataFrame(),
            'recommendations': "Not enough valid stocks for clustering analysis."
        }
    
    returns = historical_data[valid_tickers].pct_change().dropna()
    features = pd.DataFrame()
    
    for ticker in valid_tickers:
        stock_returns = returns[ticker]
        features.loc[ticker, 'mean_return'] = stock_returns.mean()
        features.loc[ticker, 'volatility'] = stock_returns.std()
        features.loc[ticker, 'skewness'] = stock_returns.skew()
        features.loc[ticker, 'kurtosis'] = stock_returns.kurtosis()
        features.loc[ticker, 'max_drawdown'] = (
            (stock_returns.cumsum() - stock_returns.cumsum().cummax()).min()
        )
        features.loc[ticker, 'autocorrelation'] = stock_returns.autocorr()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    cluster_df = pd.DataFrame({
        'ticker': valid_tickers,
        'pca1': pca_result[:, 0],
        'pca2': pca_result[:, 1],
        'cluster': clusters
    })
    
    cluster_plot = px.scatter(
        cluster_df,
        x="pca1",
        y="pca2",
        color="cluster",
        hover_name="ticker",
        title="Stock Clustering based on Return Patterns",
        labels={"pca1": "Principal Component 1", "pca2": "Principal Component 2"},
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    cluster_plot.update_traces(
        marker=dict(size=12),
        textposition='top center'
    )
    
    for i, row in cluster_df.iterrows():
        cluster_plot.add_annotation(
            x=row['pca1'],
            y=row['pca2'],
            text=row['ticker'],
            showarrow=False,
            font=dict(size=10)
        )
    
    cluster_plot.update_layout(
        height=600,
        template='plotly_white',
        legend_title_text='Cluster'
    )
    
    features['cluster'] = clusters
    cluster_characteristics = features.groupby('cluster').mean().reset_index()
    
    display_characteristics = pd.DataFrame()
    
    for i, row in cluster_characteristics.iterrows():
        cluster_id = int(row['cluster'])
        display_characteristics.loc[f'Cluster {cluster_id}', 'Average Return'] = f"{row['mean_return']*100:.2f}%"
        display_characteristics.loc[f'Cluster {cluster_id}', 'Volatility'] = f"{row['volatility']*100:.2f}%"
        display_characteristics.loc[f'Cluster {cluster_id}', 'Max Drawdown'] = f"{row['max_drawdown']*100:.2f}%"
        
        cluster_tickers = cluster_df[cluster_df['cluster'] == cluster_id]['ticker'].tolist()
        display_characteristics.loc[f'Cluster {cluster_id}', 'Stocks'] = ", ".join(cluster_tickers)
        
        if row['mean_return'] > 0 and row['volatility'] < features['volatility'].median():
            cluster_type = "Low-Risk Growth"
        elif row['mean_return'] > 0 and row['volatility'] >= features['volatility'].median():
            cluster_type = "High-Risk Growth"
        elif row['mean_return'] <= 0 and row['volatility'] < features['volatility'].median():
            cluster_type = "Low-Risk Decline"
        else:
            cluster_type = "High-Risk Decline"
        
        display_characteristics.loc[f'Cluster {cluster_id}', 'Cluster Type'] = cluster_type
    
    recommendations = []
    cluster_counts = cluster_df['cluster'].value_counts()
    max_cluster = cluster_counts.idxmax()
    
    if (cluster_counts[max_cluster] / len(valid_tickers)) > 0.6:
        cluster_type = display_characteristics.loc[f'Cluster {max_cluster}', 'Cluster Type']
        recommendations.append(f"Your portfolio is heavily concentrated in '{cluster_type}' stocks (Cluster {max_cluster}).")
        
        if "High-Risk" in cluster_type:
            recommendations.append("Consider adding more low-volatility stocks to balance your portfolio risk.")
        if "Decline" in cluster_type:
            recommendations.append("Consider including more positive-return assets for better growth potential.")
    
    if len(recommendations) > 0:
        recommendations_text = "### Diversification Insights\n\n" + "\n\n".join(recommendations)
    else:
        recommendations_text = "### Diversification Insights\n\nYour portfolio appears to be well-diversified across different types of stocks."
    
    result = {
        'cluster_plot': cluster_plot,
        'cluster_characteristics': display_characteristics,
        'recommendations': recommendations_text
    }
    
    return result

def run_monte_carlo_simulation(portfolio_returns, initial_investment, n_simulations=1000, n_days=252):
    """Run Monte Carlo simulation to forecast portfolio values."""
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
    
    percentile_returns = {
        percentile: ((value / initial_investment) - 1) * 100
        for percentile, value in percentiles.items()
    }
    
    expected_value = np.mean(simulations[-1, :])
    expected_return = ((expected_value / initial_investment) - 1) * 100
    
    fig = go.Figure()
    
    sample_size = min(100, n_simulations)
    for i in np.random.choice(n_simulations, sample_size, replace=False):
        fig.add_trace(
            go.Scatter(
                y=simulations[:, i],
                mode='lines',
                line=dict(width=0.5, color='rgba(30, 58, 138, 0.1)'),
                showlegend=False
            )
        )
    
    percentile_colors = {
        5: 'rgba(239, 68, 68, 0.8)',
        50: 'rgba(16, 185, 129, 0.8)',
        95: 'rgba(59, 130, 246, 0.8)'
    }
    
    for p, color in percentile_colors.items():
        p_values = np.percentile(simulations, p, axis=1)
        fig.add_trace(
            go.Scatter(
                y=p_values,
                mode='lines',
                line=dict(width=2, color=color),
                name=f'{p}th Percentile'
            )
        )
    
    fig.add_trace(
        go.Scatter(
            y=[initial_investment] * n_days,
            mode='lines',
            line=dict(width=1, color='black', dash='dash'),
            name='Initial Investment'
        )
    )
    
    start_date = datetime.datetime.now()
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    date_strings = [date.strftime('%Y-%m-%d') for date in dates]
    
    fig.update_layout(
        title='Monte Carlo Simulation: Projected Portfolio Value',
        xaxis_title='Trading Days',
        yaxis_title='Portfolio Value ($)',
        height=600,
        template='plotly_white',
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(
        tickvals=list(range(0, n_days, n_days//5)),
        ticktext=[date_strings[i] for i in range(0, n_days, n_days//5)]
    )
    
    result = {
        'simulations': simulations,
        'percentiles': percentiles,
        'percentile_returns': percentile_returns,
        'expected_value': expected_value,
        'expected_return': expected_return,
        'plot': fig
    }
    
    return result

def calculate_risk_metrics(portfolio_data, historical_data, investments):
    """Calculate comprehensive risk metrics for the portfolio."""
    weights = portfolio_data['weights']
    portfolio_returns = portfolio_data['portfolio_returns']
    portfolio_tickers = list(weights.keys())
    
    returns = historical_data[portfolio_tickers].pct_change().dropna()
    cov_matrix = returns.cov() * 252
    
    weights_array = np.array([weights[ticker] for ticker in portfolio_tickers])
    portfolio_volatility_value = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
    
    stock_volatilities = returns.std() * np.sqrt(252)
    
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)
    
    # total_investment = sum(weights.values())
    total_investment = sum(investments.values())
    var_95_dollar = abs(var_95 * total_investment)
    var_99_dollar = abs(var_99 * total_investment)

    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min() * 100
    
    correlation_matrix = returns.corr()
    
    metrics_data = []
    for ticker in portfolio_tickers:
        metrics_data.append({
            'Asset': ticker,
            'Weight (%)': weights[ticker] * 100 / total_investment,
            'Volatility (%)': stock_volatilities[ticker] * 100,
            'Contribution to Risk (%)': weights[ticker] * stock_volatilities[ticker] * 100 / total_investment
        })
    
    volatility_metrics = pd.DataFrame(metrics_data)
    volatility_metrics = volatility_metrics.sort_values('Contribution to Risk (%)', ascending=False)
    
    def portfolio_volatility(weights, mean_returns, cov_matrix):
        weights_array = np.array(weights)
        return np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
    
    def portfolio_return(weights, mean_returns):
        weights_array = np.array(weights)
        return np.sum(mean_returns * weights_array) * 252
    
    def minimize_volatility(weights, mean_returns, cov_matrix):
        return portfolio_volatility(weights, mean_returns, cov_matrix)
    
    mean_returns = returns.mean()
    target_returns = np.linspace(mean_returns.min() * 252, mean_returns.max() * 252, 50)
    efficient_frontier = []
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(portfolio_tickers)))
    
    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns) - target},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        )
        
        result = minimize(
            minimize_volatility,
            np.array([1/len(portfolio_tickers)] * len(portfolio_tickers)),
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result['success']:
            efficient_frontier.append({
                'return': target * 100,
                'volatility': portfolio_volatility(result['x'], mean_returns, cov_matrix) * 100
            })
    
    current_return = portfolio_return(weights_array, mean_returns) * 100
    current_volatility = portfolio_volatility(weights_array, mean_returns, cov_matrix) * 100
    
    corr_plot = px.imshow(
        correlation_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title="Asset Correlation Matrix"
    )
    
    corr_plot.update_layout(
        height=400,
        template='plotly_white'
    )
    
    var_plot = px.histogram(
        portfolio_returns, 
        nbins=50,
        title="Portfolio Returns Distribution with VaR",
        labels={'value': 'Daily Return', 'count': 'Frequency'},
        color_discrete_sequence=['rgba(30, 58, 138, 0.7)']
    )
    
    var_plot.add_vline(
        x=var_95, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"95% VaR: {var_95:.2%}",
        annotation_position="top right"
    )
    
    var_plot.add_vline(
        x=var_99, 
        line_dash="dash", 
        line_color="darkred",
        annotation_text=f"99% VaR: {var_99:.2%}",
        annotation_position="top right"
    )
    
    var_plot.update_layout(
        height=400,
        template='plotly_white',
        showlegend=False
    )
    ef_df = pd.DataFrame(efficient_frontier)

    ef_plot = px.scatter(
        ef_df, 
        x='volatility', 
        y='return',
        title="Efficient Frontier",
        labels={'volatility': 'Annualized Volatility (%)', 'return': 'Annualized Return (%)'}
    )
    
    ef_plot.add_trace(
        go.Scatter(
            x=[current_volatility],
            y=[current_return],
            mode='markers',
            marker=dict(
                color='red',
                size=12,
                symbol='star'
            ),
            name='Current Portfolio'
        )
    )
    
    ef_plot.update_layout(
        height=400,
        template='plotly_white'
    )
    
    result = {
        'volatility_metrics': volatility_metrics,
        'correlation_matrix': correlation_matrix,
        'correlation_plot': corr_plot,
        'var_95': var_95 * 100,
        'var_99': var_99 * 100,
        'var_95_dollar': abs(var_95_dollar),
        'var_99_dollar': abs(var_99_dollar),
        'max_drawdown': abs(max_drawdown),
        'portfolio_volatility': portfolio_volatility_value * 100,
        'efficient_frontier': ef_df,
        'efficient_frontier_plot': ef_plot,
        'var_plot': var_plot
    }
    
    return result 

# Visualization Functions
def plot_portfolio_performance(portfolio_data):
    """Create a plot showing portfolio performance over time."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_data['portfolio_cumulative_returns'].index,
            y=portfolio_data['portfolio_cumulative_returns'],
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        )
    )
    
    if 'market_cumulative_returns' in portfolio_data:
        fig.add_trace(
            go.Scatter(
                x=portfolio_data['market_cumulative_returns'].index,
                y=portfolio_data['market_cumulative_returns'],
                mode='lines',
                name='S&P 500',
                line=dict(color='gray', width=1, dash='dash')
            )
        )
    
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        height=400,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def plot_asset_allocation(weights):
    """Create a pie chart showing asset allocation."""
    df = pd.DataFrame({
        'Asset': list(weights.keys()),
        'Weight': [w * 100 for w in weights.values()]
    })
    
    fig = px.pie(
        df,
        values='Weight',
        names='Asset',
        title='Asset Allocation',
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white'
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label}<br>Weight: %{value:.1f}%<extra></extra>'
    )
    
    return fig
def plot_risk_return(portfolio_data, historical_data):
    """Create a risk-return scatter plot for all assets."""
    returns = historical_data.pct_change().dropna()
    annual_returns = (1 + returns.mean()) ** 252 - 1
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Calculate weights and ensure they're positive
    weights = portfolio_data['weights']
    weight_values = [weights.get(ticker, 0) * 100 for ticker in returns.columns]
    # Scale weights to a reasonable range for visualization (e.g., 5-30)
    min_size = 5
    max_size = 30
    if len(weight_values) > 0:
        scaled_weights = np.interp(weight_values, 
                                 (min(weight_values), max(weight_values)), 
                                 (min_size, max_size))
    else:
        scaled_weights = [min_size] * len(returns.columns)
    
    df = pd.DataFrame({
        'Asset': returns.columns,
        'Return': annual_returns * 100,
        'Volatility': annual_volatility * 100,
        'Weight': scaled_weights
    })

    fig = px.scatter(
        df,
        x='Volatility',
        y='Return',
        size='Weight',
        color='Asset',
        title='Risk-Return Profile',
        labels={
            'Volatility': 'Annualized Volatility (%)',
            'Return': 'Annualized Return (%)',
            'Weight': 'Portfolio Weight (%)'
        },
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    fig.add_trace(
        go.Scatter(
            x=[portfolio_data['portfolio_annual_vol'] * 100],
            y=[portfolio_data['portfolio_annual_return'] * 100],
            mode='markers',
            marker=dict(
                color='red',
                size=15,
                symbol='star'
            ),
            name='Portfolio'
        )
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig

def plot_drawdown(portfolio_returns):
    """Create a plot showing portfolio drawdown over time."""
    # Safety cleaning
    portfolio_returns1 = portfolio_returns.clip(lower=-1, upper=1)
    portfolio_returns1 = portfolio_returns1.replace([np.inf, -np.inf], np.nan).dropna()
    
    cum_returns = (1 + portfolio_returns1).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_returns1.index,
            y=drawdown * 100,
            fill='tozeroy',
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.2)'
        )
    )
    
    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        height=400,
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(
            tickformat='.1%',
            range=[-1, 0]
    ))
    
    return fig

def plot_rolling_metrics(portfolio_returns, window=30):
    """Create plots showing rolling metrics (returns, volatility, Sharpe ratio)."""
    rolling_returns = portfolio_returns.rolling(window).mean() * 252 * 100
    rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252) * 100
    risk_free_rate = 0.02
    rolling_sharpe = (rolling_returns / 100 - risk_free_rate) / (rolling_vol / 100)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=rolling_returns.index,
            y=rolling_returns,
            mode='lines',
            name='Rolling Returns',
            line=dict(color='blue', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode='lines',
            name='Rolling Volatility',
            line=dict(color='red', width=1)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe,
            mode='lines',
            name='Rolling Sharpe Ratio',
            line=dict(color='green', width=1)
        )
    )
    
    fig.update_layout(
        title=f'Rolling Metrics ({window}-day window)',
        xaxis_title='Date',
        yaxis_title='Value',
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def plot_correlation_heatmap(historical_data):
    """Create a correlation heatmap for all assets."""
    corr_matrix = historical_data.pct_change().corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Asset Correlation Matrix'
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_technical_indicators(prices_df, ticker):
    """Create a plot showing technical indicators for a specific asset."""
    fig = go.Figure()
    
    # Add price trace
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=prices_df[ticker],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        )
    )
    
    # Calculate and add moving averages
    short_ma, long_ma = calculate_moving_averages(prices_df[[ticker]])
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=short_ma[ticker],
            mode='lines',
            name='20-day MA',
            line=dict(color='blue', width=1)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=long_ma[ticker],
            mode='lines',
            name='50-day MA',
            line=dict(color='red', width=1)
        )
    )
    
    # Calculate and add Bollinger Bands
    upper, middle, lower = calculate_bollinger_bands(prices_df[[ticker]])
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=upper[ticker],
            mode='lines',
            name='Upper Band',
            line=dict(color='gray', width=1, dash='dash')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prices_df.index,
            y=lower[ticker],
            mode='lines',
            name='Lower Band',
            line=dict(color='gray', width=1, dash='dash')
        )
    )
    
    fig.update_layout(
        title=f'Technical Indicators: {ticker}',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis2=dict(
            title='RSI',
            overlaying='y',
            side='right',
            range=[0, 100]
        ),
        height=500,
        template='plotly_white',
        hovermode='x unified',
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d',
            tickangle=45
        )
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", yref="y2")
    fig.add_hline(y=30, line_dash="dash", line_color="green", yref="y2")
    
    return fig

def plot_sector_allocation(selected_tickers, ticker_security_pairs, investments):
    """Create a bar chart showing sector allocation of the portfolio."""
    # Create a mapping of ticker to sector
    ticker_sector_map = {ticker: sector for ticker, _, sector in ticker_security_pairs}
    
    # Calculate sector weights
    sector_weights = {}
    total_investment = sum(investments.values())
    
    for ticker, amount in investments.items():
        sector = ticker_sector_map.get(ticker, "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0) + (amount / total_investment * 100)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Sector': list(sector_weights.keys()),
        'Weight (%)': list(sector_weights.values())
    }).sort_values('Weight (%)', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        df,
        x='Sector',
        y='Weight (%)',
        title='Portfolio Sector Allocation',
        color='Sector',
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        xaxis_title='GICS Sector',
        yaxis_title='Portfolio Weight (%)',
        showlegend=False,
        xaxis={'categoryorder': 'total descending'}
    )
    
    fig.update_traces(
        texttemplate='%{y:.1f}%',
        textposition='outside'
    )
    
    return fig

def perform_sp500_clustering(start_date, end_date, n_clusters=5):
    """Perform K-means clustering on all S&P 500 stocks based on returns and volatility."""
    # Get all S&P 500 tickers
    ticker_security_pairs = load_sp500_tickers()
    all_tickers = [ticker for ticker, _, _ in ticker_security_pairs]
    
    # Fetch data for all S&P 500 stocks
    historical_data1 = fetch_stock_data(all_tickers, start_date, end_date, include_sp500=False)
    
    if historical_data is None or historical_data.empty:
        return None, None
    
    # Calculate returns and volatility for all stocks
    returns = historical_data1.pct_change().dropna()
    annual_returns = (1 + returns.mean()) ** 252 - 1
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Create features DataFrame
    features = pd.DataFrame({
        'Return': annual_returns * 100,
        'Volatility': annual_volatility * 100
    }).dropna()
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to features
    features['Cluster'] = clusters
    features['Cluster'] = features['Cluster'].astype(str)
    
    # Create scatter plot
    fig = px.scatter(
        features,
        x='Volatility',
        y='Return',
        color='Cluster',
        title='S&P 500 Stock Clustering by Return and Volatility',
        labels={
            'Volatility': 'Annualized Volatility (%)',
            'Return': 'Annualized Return (%)'
        },
        hover_name=features.index,
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    # Add cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
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
        template='plotly_white',
        showlegend=True,
        legend_title_text='Cluster'
    )
    
    # Calculate cluster statistics
    cluster_stats = features.groupby('Cluster').agg({
        'Return': ['mean', 'std', 'count'],
        'Volatility': ['mean', 'std']
    }).round(2)
    
    # Rename columns for better readability
    cluster_stats.columns = [
        'Average Return (%)',
        'Return Std Dev (%)',
        'Number of Stocks',
        'Average Volatility (%)',
        'Volatility Std Dev (%)'
    ]
    
    return fig, cluster_stats

# Main Application
def main():
    # App title and description
    st.title("Financial Portfolio Analyzer")
    st.markdown("Analyze your investment portfolio with advanced metrics and visualizations")
    st.markdown("Created by: Yashna Meher, Hemangi Suthar, Mahika Jain for BA870 Financial Analytics")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Portfolio Configuration")
        
        # Date range selection
        st.subheader("Date Range")
        min_date = datetime.date(2015, 1, 2)
        max_date = datetime.date(2024, 12, 30)
        today = datetime.date.today()
        
        start_date = st.date_input(
            "Start Date",
            value=datetime.date(2020, 1, 1),
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.date_input(
            "End Date",
            value=min(today, max_date),
            min_value=start_date,
            max_value=max_date
        )
        
        # Load S&P 500 tickers
        ticker_security_pairs = load_sp500_tickers()
        ticker_options = [f"{ticker} - {security}" for ticker, security, _ in ticker_security_pairs]
        ticker_dict = {f"{ticker} - {security}": ticker for ticker, security, _ in ticker_security_pairs}
        
        # Find the exact default options from our list
        default_options = []
        default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        for ticker, security, _ in ticker_security_pairs:
            if ticker in default_tickers:
                default_options.append(f"{ticker} - {security}")
        
        # Stock selection
        st.subheader("Stock Selection")
        selected_ticker_names = st.multiselect(
            "Select stocks (max 50)",
            options=ticker_options,
            default=default_options,
            max_selections=50
        )
        
        # Convert selected ticker names back to just tickers
        selected_tickers = [ticker_dict[name] for name in selected_ticker_names]
        
        # Investment amounts
        st.subheader("Investment Amounts")
        investments = {}
        
        if not selected_tickers:
            st.warning("Please select at least one stock")
        else:
            st.markdown("### Enter Investment per Stock")
            investments = {}
            total_input = 0
            for ticker in selected_tickers:
                val = st.number_input(
                    f"Enter amount invested in {ticker}",
                    min_value=0,
                    value=1000,
                    step=100,
                    key=f"investment_{ticker}"
                    )
                investments[ticker] = val
                total_input += val

 #Show total and warnings
            st.markdown(f"**Total Allocated: ${format_currency(total_input)}**")
            if total_input == 0:
                st.error("⚠️ You must allocate some investment amount.")
            elif total_input < 1000:
                st.warning("⚠️ Total investment seems too low for meaningful analysis.")
            else:
                st.success("✓ Allocation looks good.")

            
            # Add investment status message
            current_total = sum(investments.values())
            if current_total < total_input:
                st.error(f"⚠️ Total allocated: ${format_currency(current_total)} (${format_currency(total_investment - current_total)} remaining)")
            elif current_total > total_input:
                st.error(f"⚠️ Total allocated: ${format_currency(current_total)} (${format_currency(current_total - total_investment)} over allocated)")
            else:
                st.success(f"✓ Total allocated: ${format_currency(current_total)} (Fully allocated)")
        
        # Analysis options
        st.subheader("Analysis Options")
        monte_carlo_sims = st.slider("Monte Carlo Simulations", min_value=100, max_value=10000, value=1000, step=100)
        
        analyze_button = st.button("Analyze Portfolio", type="primary", use_container_width=True)
    
    # Main content
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
    
    if analyze_button:
        st.session_state.analyzed = True
    
    if st.session_state.analyzed and selected_tickers:
        # Create tabs for different analyses
        tabs = st.tabs([
            "Portfolio Overview", 
            "Monte Carlo Simulation", 
            "Risk Metrics", 
            "Technical Analysis", 
            "Stock Clustering"
        ])
        
        # Fetch historical data
        with st.spinner("Fetching historical data..."):
            historical_data = fetch_stock_data(selected_tickers, start_date, end_date)
        
        if historical_data is None or historical_data.empty:
            st.error("Failed to fetch historical data. Please check your internet connection and try again.")
            st.stop()
        
        # Calculate portfolio performance
        portfolio_data = calculate_portfolio_performance(historical_data, investments)
        
        if portfolio_data is None:
            st.error("Failed to calculate portfolio performance. Please check your investments and try again.")
            st.stop()
        
        # Tab 1: Portfolio Overview
        with tabs[0]:
            st.header("Portfolio Overview")
            
            # Portfolio metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_value = sum(investments.values())
            
            with col1:
                st.metric(
                    "Total Investment", 
                    f"${format_currency(total_value)}"
                )
            
            with col2:
                if 'portfolio_cumulative_returns' in portfolio_data and len(portfolio_data['portfolio_cumulative_returns']) > 0:
                    portfolio_return = portfolio_data['portfolio_cumulative_returns'].iloc[-1] * 100 - 100
                    st.metric(
                        "Total Return", 
                        f"{format_percentage(portfolio_return)}%",
                        delta=f"{format_percentage(portfolio_return)}"
                    )
                else:
                    st.metric(
                        "Total Return", 
                        "N/A",
                        delta="No data available"
                    )
            
            with col3:
                if 'portfolio_annual_return' in portfolio_data:
                    annual_return = portfolio_data['portfolio_annual_return'] * 100
                    st.metric(
                        "Annual Return", 
                        f"{format_percentage(annual_return)}%",
                        delta=f"{format_percentage(annual_return)}"
                    )
                else:
                    st.metric(
                        "Annual Return", 
                        "N/A",
                        delta="No data available"
                    )
            
            with col4:
                sharpe = portfolio_data['portfolio_sharpe_ratio']
                st.metric(
                    "Sharpe Ratio", 
                    f"{sharpe:.2f}"
                )
            
            with col5:
                beta = portfolio_data.get('portfolio_beta', None)
                if beta is not None and not np.isnan(beta):
                    st.metric(
                        "Beta (vs S&P 500)",
                        f"{beta:.2f}"
                    )
                else:
                    st.metric(
                        "Beta (vs S&P 500)",
                        "N/A"
                    )
            
            # Portfolio performance chart
            st.markdown("### Portfolio Performance")
            st.plotly_chart(plot_portfolio_performance(portfolio_data), use_container_width=True)
            
            # Sector allocation chart
            st.markdown("### Sector Allocation")
            st.plotly_chart(plot_sector_allocation(selected_tickers, ticker_security_pairs, investments), use_container_width=True)
            
            # Asset allocation
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Asset Allocation")
                st.plotly_chart(plot_asset_allocation(portfolio_data['weights']), use_container_width=True)
            
            with col2:
                st.markdown("### Risk-Return Profile")
                st.plotly_chart(plot_risk_return(portfolio_data, historical_data), use_container_width=True)
            
            # Drawdown analysis
            st.markdown("### Drawdown Analysis")
            st.plotly_chart(plot_drawdown(portfolio_data['portfolio_returns']), use_container_width=True)
            
            # Rolling metrics
            st.markdown("### Rolling Metrics")
            st.plotly_chart(plot_rolling_metrics(portfolio_data['portfolio_returns']), use_container_width=True)
        
        # Tab 2: Monte Carlo Simulation
        with tabs[1]:
            st.header("Monte Carlo Simulation")
            
            simulation_results = run_monte_carlo_simulation(
                portfolio_data['portfolio_returns'],
                total_value,
                n_simulations=monte_carlo_sims
            )
            
            # Display simulation plot
            st.plotly_chart(simulation_results['plot'], use_container_width=True)
            
            # Display simulation metrics
            st.markdown("### Simulation Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Expected Value", 
                    f"${format_currency(simulation_results['expected_value'])}",
                    delta=f"{format_percentage(simulation_results['expected_return'])}%"
                )
            
            with col2:
                st.metric(
                    "95th Percentile", 
                    f"${format_currency(simulation_results['percentiles'][95])}",
                    delta=f"{format_percentage(simulation_results['percentile_returns'][95])}%"
                )
            
            with col3:
                st.metric(
                    "5th Percentile", 
                    f"${format_currency(simulation_results['percentiles'][5])}",
                    delta=f"{format_percentage(simulation_results['percentile_returns'][5])}%"
                )
        
        # Tab 3: Risk Metrics
        with tabs[2]:
            st.header("Risk Analysis")
            
            risk_metrics = calculate_risk_metrics(portfolio_data, historical_data, investments)


            # Display risk metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Portfolio Volatility",
                    f"{format_percentage(risk_metrics['portfolio_volatility'])}%"
                )
            
            with col2:
                st.metric(
                    "Max Drawdown",
                    f"{format_percentage(risk_metrics['max_drawdown'])}%"
                )
            
            with col3:
                st.metric(
                    "95% VaR",
                    f"${format_currency(risk_metrics['var_95_dollar'])}"
                )
            
            with col4:
                st.metric(
                    "99% VaR",
                    f"${format_currency(risk_metrics['var_99_dollar'])}"
                )
            
            # Display risk plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Value at Risk Distribution")
                st.plotly_chart(risk_metrics['var_plot'], use_container_width=True)
            
            with col2:
                st.markdown("### Efficient Frontier")
                st.plotly_chart(risk_metrics['efficient_frontier_plot'], use_container_width=True)
            
            st.markdown("### Asset Correlation Matrix")
            st.plotly_chart(risk_metrics['correlation_plot'], use_container_width=True)
            
            st.markdown("### Risk Contribution by Asset")
            st.dataframe(risk_metrics['volatility_metrics'], use_container_width=True)
        
        # Tab 4: Technical Analysis
        with tabs[3]:
            st.header("Technical Analysis")
            
            # Stock selector for technical analysis
            selected_stock = st.selectbox(
                "Select Stock for Technical Analysis",
                options=selected_tickers
            )
            
            if selected_stock:
                st.plotly_chart(
                    plot_technical_indicators(historical_data, selected_stock),
                    use_container_width=True
                )
        
        # Tab 5: Stock Clustering
        with tabs[4]:
            st.header("Stock Clustering Analysis")
            
            # Portfolio clustering
            clustering_results = perform_kmeans_clustering(historical_data, selected_tickers)
            
            if clustering_results['cluster_plot'] is not None:
                st.plotly_chart(clustering_results['cluster_plot'], use_container_width=True)
                
                st.markdown("### Cluster Characteristics")
                st.dataframe(clustering_results['cluster_characteristics'], use_container_width=True)
                
                st.markdown(clustering_results['recommendations'])
            else:
                st.warning(clustering_results['recommendations'])
            
            # S&P 500 clustering
            st.markdown("### S&P 500 Stock Clustering")
            st.markdown("This analysis clusters all S&P 500 stocks based on their historical returns and volatility.")
            
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=3,
                max_value=8,
                value=3,
                step=1
            )
            
            with st.spinner("Fetching and analyzing S&P 500 data..."):
                sp500_cluster_fig, sp500_cluster_stats = perform_sp500_clustering(start_date, end_date, n_clusters)
                
                if sp500_cluster_fig is not None:
                    st.plotly_chart(sp500_cluster_fig, use_container_width=True)
                    st.markdown("### Cluster Statistics")
                    st.dataframe(sp500_cluster_stats, use_container_width=True)
                    
                    st.markdown("""
                    #### Interpretation:
                    - Each point represents a stock in the S&P 500
                    - Stocks are grouped into clusters based on similar return and volatility characteristics
                    - The 'X' markers show the center of each cluster
                    - Hover over points to see the stock ticker
                    """)
                else:
                    st.error("Failed to fetch S&P 500 data. Please try again later.")
    else:
        # Initial state - show instructions
        st.markdown("""
        ## Welcome to the Financial Portfolio Analyzer!
        
        This application allows you to analyze and visualize your investment portfolio with advanced metrics and simulations.
        
        ### How to get started:
        
        1. Select your preferred stocks from the S&P 500 list in the sidebar
        2. Allocate investment amounts to each stock
        3. Set your analysis timeframe
        4. Click "Analyze Portfolio" to generate insights
        
        ### Available Analyses:
        
        - **Portfolio Overview**: See your portfolio's performance compared to the market
        - **Monte Carlo Simulation**: Visualize potential future outcomes
        - **Risk Metrics**: Understand your portfolio's risk profile
        - **Sector Breakdown**: Analyze sector allocation and performance
        - **Real-Time Prices**: Check current stock prices and metrics
        - **Stock Clustering**: Identify stocks with similar behavior patterns
        
        Select stocks in the sidebar to begin!
        """)
        
        # Sample visualizations
        st.image("https://images.pexels.com/photos/6801648/pexels-photo-6801648.jpeg", caption="Financial data visualization")

if __name__ == "__main__":
    main() 
