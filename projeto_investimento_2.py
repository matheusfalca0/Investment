import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta


# ---------------------------
# Funções para Otimização e Fronteira
# ---------------------------

def download_data(tickers, start, end):
    """Baixa dados para cada ticker, usando 'Adj Close' ou 'Close'."""
    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start, end=end)
            if 'Adj Close' in data.columns:
                adj_close_df[ticker] = data['Adj Close']
            elif 'Close' in data.columns:
                adj_close_df[ticker] = data['Close']
            else:
                st.warning(f"Neither 'Adj Close' nor 'Close' found for {ticker}")
        except Exception as e:
            st.error(f"Error downloading data for {ticker}: {e}")
    adj_close_df.dropna(inplace=True)
    return adj_close_df


def calculate_statistics(adj_close_df):
    """Calcula retornos logarítmicos, retorno esperado anual e matriz de covariância anualizada."""
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    retornos_esperados = log_returns.mean() * 252
    matriz_covariancia = log_returns.cov() * 252
    return log_returns, retornos_esperados, matriz_covariancia


def std_deviation(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)


def expected_return(weights, retornos_esperados):
    return np.sum(retornos_esperados * weights)


def sharpe_ratio(weights, retornos_esperados, cov_matrix, risk_free_rate):
    return (expected_return(weights, retornos_esperados) - risk_free_rate) / std_deviation(weights, cov_matrix)


def neg_sharpe_ratio(weights, retornos_esperados, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, retornos_esperados, cov_matrix, risk_free_rate)


# ---------------------------
# Funções para Screening
# ---------------------------
def get_stock_info(ticker):
    """Retorna informações básicas e histórico do ticker usando yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        return info, hist
    except Exception as e:
        st.error(f"Erro ao obter dados para {ticker}: {e}")
        return None, None


# ---------------------------
# Função para Backtest
# ---------------------------
def run_backtest(tickers, start, end):
    """Realiza um backtest simples usando pesos iguais."""
    data = download_data(tickers, start, end)
    if data.empty:
        st.error("Não foi possível baixar dados para os tickers informados.")
        return None
    daily_returns = data.pct_change().dropna()
    # Utiliza pesos iguais
    pesos = np.array([1 / len(tickers)] * len(tickers))
    portfolio_returns = daily_returns.dot(pesos)
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    return portfolio_cumulative


# ---------------------------
# Interface com Streamlit: Abas
# ---------------------------
st.title('Projeto de Investimentos')

tabs = st.tabs(["Otimização de Carteira", "Screening de Ações", "Backtest"])

# -------------
# Aba 1: Otimização de Carteira e Fronteira Eficiente
# -------------
with tabs[0]:
    st.header("Otimização de Carteira e Fronteira Eficiente")

    st.markdown("Configure os ativos e o período para análise:")
    tickers_input = st.text_input('Ativos (separados por vírgula)', 'IVV, SPY, VGT, QQQ, VOO, OEF')
    tickers = [t.strip().upper() for t in tickers_input.split(',')]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Data de Início', datetime.today() - timedelta(days=5 * 365))
    with col2:
        end_date = st.date_input('Data de Fim', datetime.today())

    if st.button('Calcular Otimização'):
        with st.spinner('Baixando dados e otimizando...'):
            data = download_data(tickers, start_date, end_date)
            if data.empty:
                st.error("Não foi possível baixar os dados para os ativos informados.")
            else:
                log_returns, retornos_esperados, matriz_covariancia = calculate_statistics(data)
                risk_free_rate = 0.05  # Taxa livre de risco

                # Otimização com restrição: cada ativo <=20%
                constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
                bounds = [(0, 0.2) for _ in range(len(tickers))]
                initial_weights = np.array([1 / len(tickers)] * len(tickers))

                optimized_results = minimize(
                    neg_sharpe_ratio,
                    initial_weights,
                    args=(retornos_esperados, matriz_covariancia, risk_free_rate),
                    method='SLSQP',
                    constraints=constraints,
                    bounds=bounds
                )

                if optimized_results.success:
                    optimal_weights = optimized_results.x
                    optimal_weights_percent = optimal_weights * 100
                    portfolio_df = pd.DataFrame({
                        'Ativo': tickers,
                        'Peso (%)': optimal_weights_percent
                    }).sort_values(by='Peso (%)', ascending=False)

                    st.subheader("Pesos Otimizados")
                    st.dataframe(portfolio_df)

                    # Fronteira Eficiente
                    num_carteiras = 10000
                    pesos_carteiras = np.random.dirichlet(np.ones(len(tickers)), num_carteiras)
                    retornos_carteiras = np.dot(pesos_carteiras, retornos_esperados)
                    volatilidades_carteiras = np.sqrt(
                        np.sum(pesos_carteiras @ matriz_covariancia * pesos_carteiras, axis=1))
                    sharpe_ratios = (retornos_carteiras - risk_free_rate) / volatilidades_carteiras

                    idx_max_sharpe = np.argmax(sharpe_ratios)
                    sharpe_otimo = sharpe_ratio(optimal_weights, retornos_esperados, matriz_covariancia, risk_free_rate)

                    st.write(f"Índice de Sharpe da Carteira Ótima: {sharpe_otimo:.4f}")

                    fig = px.scatter(
                        x=volatilidades_carteiras,
                        y=retornos_carteiras,
                        color=sharpe_ratios,
                        title='Fronteira Eficiente',
                        labels={'x': 'Volatilidade (Risco)', 'y': 'Retorno Esperado'},
                        color_continuous_scale='viridis'
                    )

                    fig.add_trace(go.Scatter(
                        x=[volatilidades_carteiras[idx_max_sharpe]],
                        y=[retornos_carteiras[idx_max_sharpe]],
                        mode='markers+text',
                        marker=dict(color='red', size=12, symbol='star'),
                        text=['Máximo Sharpe'],
                        textposition='top left',
                        name='Máximo Sharpe'
                    ))

                    st.plotly_chart(fig)
                else:
                    st.error("Falha na otimização. Tente ajustar os parâmetros ou os ativos.")

# -------------
# Aba 2: Screening de Ações
# -------------
with tabs[1]:
    st.header("Screening de Ações")
    st.markdown("Pesquise um ticker para visualizar informações básicas e o histórico do ativo.")

    ticker_screen = st.text_input('Ticker para Screening', 'AAPL')

    if st.button('Buscar Informações'):
        with st.spinner('Obtendo informações...'):
            info, hist = get_stock_info(ticker_screen)
            if info is not None:
                st.subheader("Informações Básicas")
                # Exibe algumas informações selecionadas
                info_to_show = {
                    'Nome': info.get('longName'),
                    'Setor': info.get('sector'),
                    'Indústria': info.get('industry'),
                    'País': info.get('country'),
                    'Valor de Mercado': info.get('marketCap'),
                    'Preço Atual': info.get('currentPrice'),
                    'P/L': info.get('trailingPE')
                }
                st.write(info_to_show)

                st.subheader("Histórico do Preço (1 ano)")
                st.line_chart(hist['Close'] if 'Close' in hist.columns else hist['Adj Close'])
            else:
                st.error("Não foi possível obter informações para o ticker informado.")

# -------------
# Aba 3: Backtest
# -------------
with tabs[2]:
    st.header("Backtest da Carteira")
    st.markdown("Selecione os ativos e o período para realizar um backtest simples da carteira (pesos iguais).")

    tickers_back = st.text_input('Ativos para Backtest (separados por vírgula)', 'AAPL, MSFT, GOOGL')
    tickers_back = [t.strip().upper() for t in tickers_back.split(',')]

    col3, col4 = st.columns(2)
    with col3:
        backtest_start = st.date_input('Data de Início do Backtest', datetime.today() - timedelta(days=5 * 365),
                                       key='bt_start')
    with col4:
        backtest_end = st.date_input('Data de Fim do Backtest', datetime.today(), key='bt_end')

    if st.button('Executar Backtest'):
        with st.spinner('Realizando backtest...'):
            portfolio_cumulative = run_backtest(tickers_back, backtest_start, backtest_end)
            if portfolio_cumulative is not None:
                st.subheader("Retorno Acumulado da Carteira")
                st.line_chart(portfolio_cumulative)
                st.write(f"Valor final da carteira (supondo capital inicial 1): {portfolio_cumulative.iloc[-1]:.4f}")