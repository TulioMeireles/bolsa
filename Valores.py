import yfinance as yf
import streamlit as st
from PIL import Image
from prophet import Prophet
import plotly.express as px
from plotly import graph_objects as go

# Configuração da página
st.set_page_config(page_title='Análise Financeira', layout='wide')
st.markdown(f""" <h1 style='text-align: center;'>Análise de ações financeiras</h1> """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # Logo da Loja
    logo = Image.open('bolsa.png')
    st.sidebar.image(logo, use_column_width=True, caption='Ações financeira')
    st.header('', divider='rainbow')

    st.info('Ticker da empresa: ', icon='🔢')
    ticker = st.text_input(label='Insira o Ticker:')
    st.header('', divider='rainbow')

    st.info('Previsão: ', icon='⏱')
    n_dias = st.slider('Dias de previsão', 30, 365)  # Slider para previsão de dias

if ticker:
    try:
        ticker_sa = f"{ticker}.SA"
        # Verifica se o ticker sem .SA possui dados, se não possui tenta buscar com .SA
        if yf.Ticker(ticker).history(period='1Y').empty:
            # Verifica se o ticker com .SA possui dados
            if yf.Ticker(ticker_sa).history(period='1Y').empty:
                raise ValueError('Dados não encontrados para o ticker fornecido')
            else:
                ticker = ticker_sa

        # Obtendo os dados financeiros da empresa selecionada no período de 1 ano
        df = yf.Ticker(ticker).history(period='1Y').reset_index()

        # Filtro para os últimos 'n_dias' dias
        df_valores = df.tail(n_dias)

        # Preparando os dados para o Prophet
        df_treino = df_valores[['Date', 'Close']]
        df_treino = df_treino.rename(columns={'Date': 'ds', 'Close': 'y'})

        df_treino['ds'] = df_treino['ds'].apply(lambda x: x.replace(tzinfo=None) if x.tzinfo else x)

        # Criando o modelo e fazendo a previsão
        modelo = Prophet()
        modelo.fit(df_treino)
        futuro = modelo.make_future_dataframe(periods=n_dias, freq='B')
        futuro['ds'] = futuro['ds'].apply(lambda x: x.replace(tzinfo=None) if x.tzinfo else x)
        previsao = modelo.predict(futuro)

        # --- GRÁFICOS ---

        # Criando as colunas dos gráficos!!
        col1, col2, col3 = st.columns(3)
        col4, col5 = st.columns(2)
        col6, col7 = st.columns(2)
        col8, col9 = st.columns(2)

        # Card fechamento máximo
        with col1:
            maximo = df_valores['Close'].max()
            st.info('Máximo fechamento R$', icon='💲')
            st.metric('Total R$', '', f'{maximo:,.2f}')

        # Card fechamento mínimo
        with col2:
            minimo = df_valores['Close'].min()
            st.info('Mínimo fechamento R$', icon='💲')
            st.metric('Total R$', '', f'{minimo:,.2f}')

        # Card fechamento média
        with col3:
            media = df_valores['Close'].mean()
            st.info('Média fechamento R$', icon='🔢')
            st.metric('Total R$', '', f'{media:,.2f}')

        # Tabela de valores das ações
        with col4:
            st.info('Tabela de valores das ações - ' + ticker, icon='📅')
            st.dataframe(df_valores[['Date', 'Open', 'Close']].sort_values('Date', ascending=False))

        # Gráfico dos valores das ações
        with col5:
            st.info('Gráfico de valores da ações - ' + ticker, icon='📉')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_valores['Date'],
                                     y=df_valores['Close'],
                                     name='Preço Fechamento',
                                     line_color='Yellow',
                                     hovertemplate='%{x} <br> Valor: %{y:.2f}'))  # Exibe o dia (x) e o valor com 2 casas decimais (y)
            fig.update_layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))

            fig.add_trace(go.Scatter(x=df_valores['Date'],
                                     y=df_valores['Open'],
                                     name='Preço Abertura',
                                     line_color='green',
                                     hovertemplate='%{x} <br> Valor: %{y:.2f}'))  # Exibe o dia (x) e o valor com 2 casas decimais (y)
            fig.update_layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))

            col5.plotly_chart(fig, use_container_width=True)


        # Tabela de Previsões
        with col6:
            st.info('Tabela de Previsões - ' + ticker, icon='🗓')
            st.write(previsao[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_dias))

        # Gráfico da previsão
        with col7:
            st.info('Gráfico de previsões - ' + ticker, icon='📉')
            fig_prophet = go.Figure()
            fig_prophet.add_trace(go.Scatter(x=previsao['ds'],
                                             y=previsao['yhat'],
                                             name='Previsto',
                                             line_color='blue',
                                             hovertemplate='%{x} <br> Valor: %{y:.2f}'))  # Exibe o dia (x) e o valor com 2 casas decimais (y)

            fig_prophet.add_trace(go.Scatter(x=previsao['ds'],
                                             y=previsao['yhat_lower'],
                                             name='Limite Inferior',
                                             line=dict(dash='dot'),
                                             line_color='red',
                                             hovertemplate='%{x} <br> Valor: %{y:.2f}'))  # Exibe o dia (x) e o valor com 2 casas decimais (y)

            fig_prophet.add_trace(go.Scatter(x=previsao['ds'],
                                             y=previsao['yhat_upper'],
                                             name='Limite Superior',
                                             line=dict(dash='dot'),
                                             line_color='green',
                                             hovertemplate='%{x} <br> Valor: %{y:.2f}'))  # Exibe o dia (x) e o valor com 2 casas decimais (y)

            col7.plotly_chart(fig_prophet, use_container_width=True)

        with col8:

            st.info('Previsão com gráfico de barras e Linha', icon='📉')
            # Criando o gráfico de barras
            # Estilizando o gráfico de barras

            fig_barras = px.bar(previsao,
                                x=df_valores['Date'],
                                y=df_valores['Close'],
                                labels={'Close': 'Close'})
            fig_barras.update_traces(marker_color='green', marker_line_color='green', marker_line_width=1.5,
                                     opacity=0.8)
            fig_barras.update_layout(xaxis_title='Data',
                                     yaxis_title='Valor',
                                     legend_title='Legendas',
                                     font=dict(family='Arial', size=12),
                                     xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                                     yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
                                     )

            # Criando a linha no gráfico
            fig_barras.add_trace(go.Scatter(
                x=previsao['ds'],
                y=previsao['yhat'],
                mode='lines',
                name='Previsto',
                line=dict(color='red', width=2),
                hovertemplate='%{x} <br> Valor: %{y:.2f}'
            ))

            col8.plotly_chart(fig_barras, use_container_width=True)



    except Exception as e:
        st.error(f"Ocorreu um erro ao buscar os dados: {e}")
