import plotly.graph_objects as go  # Import Plotly for interactive charts
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime  
from scipy.stats import norm

# Set Streamlit configuration to avoid deprecation warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Title and Introduction
st.title("Simulation d'Options Européennes")

# Create a sidebar to select the asset
st.sidebar.title("Sélectionnez un Actif Sous-jacent")
asset_option = st.sidebar.radio("Choisissez une option :", ["Sélectionnez parmi les actions prédéfinies", "Entrer un autre symbole"])

if asset_option == "Sélectionnez parmi les actions prédéfinies":
    selected_action = st.sidebar.selectbox("Sélectionnez une action pour la simulation :", ["AAPL", "MSFT", "GOOGL", "TSLA"])
else:
    custom_symbol = st.sidebar.text_input("Entrez le symbole de l'action :", "AAPL")
    if custom_symbol:
        selected_action = custom_symbol.upper()  # Convert to uppercase
    else:
        st.sidebar.warning("Veuillez entrer un symbole d'action.")

# Setup tabs for different simulations
tab1, tab2, tab3, tab4 = st.tabs(["Mouvement Brownien", "Mouvement Brownien Géométrique", "Simulation de Monte Carlo", "Prix des Options"])

# Real-time data retrieval from Yahoo Finance
try:
    start_date = st.date_input("Date de début", datetime(2023, 1, 1)) 
    end_date = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(selected_action, start=start_date, end=end_date, progress=False)
    df = df[["Adj Close"]]

    # Calculate Historical Volatility
    daily_returns = df['Adj Close'].pct_change()
    daily_std = daily_returns.std()
    annualized_volatility = daily_std * np.sqrt(252)  # Assuming 252 trading days in a year
except Exception as e:
    st.error(f"Erreur lors de la récupération des données pour le symbole {selected_action}. Assurez-vous d'entrer un symbole valide.")
    st.stop()


with tab1:  # For Standard Brownian Motion
    with st.expander("Paramètres de la simulation de Mouvement Brownien"):
        # Inputs for Brownian Motion simulation
        num_simulations = st.number_input("Le nombre de simulations", value=1, min_value=1, key='num_simulations_bm')
        T = st.number_input("La période", value=200, min_value=1, key='T_bm')

    # Define the function to simulate Brownian Motion inside the tab to ensure proper scope
    def simulate_brownian_motion(num_simulations, T):
        n = T
        dt = 1.0 / n
        t = np.linspace(0., T, n)
        d = 1
        B = np.zeros((n, d))
        simulations = []
        for _ in range(num_simulations):
            dB = np.sqrt(dt) * np.random.normal(size=(n-1, d))
            B[1:, :] = np.cumsum(dB, axis=0)
            simulations.append((t, B))
        return simulations 

    # Button to perform Brownian Motion simulation
    if st.button('Simuler', key='simulate_bm'):
        st.header("Résultats de la simulation du Mouvement Brownien Standard")
        simulations = simulate_brownian_motion(num_simulations, T)
        
        # Display the graph for the first simulation
        st.subheader("Graphique du Mouvement Brownien Standard")
        for t, B in simulations:
            plt.plot(t, B)
        plt.xlabel("Temps")
        plt.ylabel("Valeur")
        st.pyplot()

        # Display the data table for the first simulation
        st.subheader("Aperçu des valeurs générées:")
        data = pd.DataFrame({
            'Temps': t.flatten(),
            'Valeur': B[:, 0].flatten()
        })
        st.dataframe(data)

with tab2:  # For Geometric Brownian Motion
    with st.expander("Paramètres de la simulation du Mouvement Brownien Géométrique"):
        # Calculate daily log returns for mu calculation
        log_returns = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
        # Calculate mu (annualized average log return)
        mu_geo = log_returns.mean() * 252  # Assuming 252 trading days in a year
        # Use calculated historical volatility for sigma
        sigma_geo = annualized_volatility  # Already calculated outside this tab

        S0_geo = st.number_input("Prix initial de l'action", value=float(df["Adj Close"][-1]), min_value=0.01, step=0.01)
        T_geo = st.number_input("Période (T) du Mouvement Brownien Géométrique", value=1.0, min_value=0.01, max_value=10.0, step=0.01)
        n_geo = st.number_input("Nombre de pas (n) du Mouvement Brownien Géométrique", value=252, min_value=10)

        # Display the calculated mu and sigma to the user
        st.write(f"Drift calculé (μ): {mu_geo:.4f}")
        st.write(f"Volatilité historique calculée (σ): {sigma_geo:.4f}")

    # Button to perform Geometric Brownian Motion simulation
    if st.button('Simuler le Mouvement Brownien Géométrique', key='simulate_gbm'):
        st.header("La simulation du Mouvement Brownien Géométrique")
        
        # Geometric Brownian Motion simulation
        dt_geo = T_geo / n_geo
        t_geo = np.linspace(0, T_geo, n_geo)
        St_geo = np.zeros(n_geo)
        St_geo[0] = S0_geo
        for t in range(1, n_geo):
            St_geo[t] = St_geo[t-1] * np.exp((mu_geo - 0.5 * sigma_geo**2) * dt_geo + sigma_geo * np.sqrt(dt_geo) * np.random.normal())
        
        # Plotting the Geometric Brownian Motion
        st.subheader("Graphique du Mouvement Brownien Géométrique")
        plt.figure(figsize=(10, 6))
        plt.plot(t_geo, St_geo)
        plt.xlabel("Temps")
        plt.ylabel("Prix de l'action")
        st.pyplot()


        # Display the data table for the Geometric Brownian Motion
        st.subheader("Aperçu des valeurs générées pour le Mouvement Brownien Géométrique:")
        data_geo = pd.DataFrame({
            'Temps': t_geo,
            'Prix de l\'action': St_geo
        })
        st.dataframe(data_geo)
with tab3:  # For Monte Carlo Simulation
    with st.expander("Simulation de Monte Carlo pour le pricing d'options"):
        S0_mc = st.number_input("Prix initial de l'action S0", value=float(df["Adj Close"][-1]), min_value=0.01, step=0.01)
        K_mc = st.number_input("Prix d'exercice K", value=100.0, min_value=0.01, step=0.01)
        T_mc = st.number_input("Maturité de l'option T (en années)", value=1.0, min_value=0.01, max_value=10.0, step=0.01)
        r_mc = st.number_input("Taux d'intérêt sans risque r", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
        # Use the calculated historical volatility for sigma
        sigma_mc = annualized_volatility  # Use the calculated historical volatility here
        simulations_mc = st.number_input("Nombre de simulations", value=10000, min_value=100, max_value=1000000, step=100)
        st.write(f"Volatilité historique calculée (σ): {sigma_mc:.4f}")

        def monte_carlo_pricing(S0, K, T, r, sigma, simulations):
            """European Call Option via Monte Carlo simulation."""
            dt = T / simulations
            price_paths = np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal(size=(simulations,)))
            S = S0 * np.cumprod(price_paths)
            payoff = np.maximum(S - K, 0)
            prices = np.exp(-r * T) * payoff
            return prices  # Return prices over time

        if st.button('Calculer le prix via Monte Carlo', key='calculate_mc'):
            option_prices = monte_carlo_pricing(S0_mc, K_mc, T_mc, r_mc, sigma_mc, simulations_mc)

            st.header("Résultat du Pricing d'Option")
            st.write(f"Le prix moyen estimé est: {np.mean(option_prices):.2f} €")
            st.write(f"Le prix médian estimé est: {np.median(option_prices):.2f} €")

            # Create an interactive Plotly line chart for option prices over time
            time_intervals = np.linspace(0, T_mc, simulations_mc)
            fig_prices_over_time = go.Figure()
            fig_prices_over_time.add_trace(go.Scatter(x=time_intervals, y=option_prices, mode='lines', name='Prix de l\'option'))
            fig_prices_over_time.update_layout(
                title="Prix de l'Option en Fonction du Temps",
                xaxis_title="Temps (années)",
                yaxis_title="Prix de l'Option",
                template="plotly_dark"  # You can choose different templates
            )
            st.plotly_chart(fig_prices_over_time)

            

with tab4:  # For Pricing Options using Black-Scholes Model
    with st.expander("Prix des Options Européennes avec le modèle Black-Scholes"):
        S0_bs = st.number_input("Prix initial de l'action S0 (Black-Scholes)", value=float(df["Adj Close"][-1]), min_value=0.01, step=0.01)
        K_bs = st.number_input("Prix d'exercice K (Black-Scholes)", value=100.0, min_value=0.01, step=0.01)
        
        # Dynamically calculate the time to maturity based on the option's expiration date
        expiration_date = st.date_input("Date d'expiration de l'option", value=datetime.now().date())
        current_date = datetime.now().date()
        T_bs = (expiration_date - current_date).days / 365.25  # Convert days to years
        
        r_bs = st.number_input("Taux d'intérêt sans risque r (Black-Scholes)", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
        sigma_bs = annualized_volatility  # Use the calculated historical volatility

        def black_scholes(S0, K, T, r, sigma, option_type="call"):
            """Black-Scholes formula."""
            d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            if option_type == "call":
                price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            return price

        option_type_bs = st.selectbox("Type d'option", ["call", "put"], key='option_type_bs')
        if st.button('Calculer le prix Black-Scholes', key='calculate_bs'):
            option_price_bs = black_scholes(S0_bs, K_bs, T_bs, r_bs, sigma_bs, option_type=option_type_bs)
            st.header("Résultat du Pricing selon Black-Scholes")
            st.write(f"Le prix estimé de l'option {option_type_bs} européenne est: {option_price_bs:.2f} €")