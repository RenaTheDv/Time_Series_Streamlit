import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet
import lightgbm as lgb

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor



# структура:
# Основная страница - загрузка st.file_uploader, внутри происходят выявление временных признаков и таргета
# таргет и временной ряд выбираем через st.selectbox - если нет временного ряда - ошибка
# пусть выдает получившийся датасет (первые 5 колонок), потом дает возможность выбрать модель обучения для r2 и mae
# через st.sidebar пусть будут показываться графики для конкретной модели (предсказания ее и реальные значения)
if 'page' not in st.session_state:
    st.session_state.page = 'page_1'

def page_1():
    st.title('Приложение обработки временных рядов')
    st.write('Пользователю необходимо загрузить свой DataFrame с временным признаком, выбрать target.')
    uploaded_file = st.file_uploader('Загрузите свой DataFrame...', type=['csv', 'json', 'xlsx', 'xls', 'parquet'])
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()    #берем расширение загруженного файла

        try:
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext == 'json':
                df = pd.read_json(uploaded_file)
            elif file_ext == 'xlsx':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_ext == 'xls':
                df = pd.read_excel(uploaded_file, engine='xlrd')
            elif file_ext == 'parquet':
                df = pd.read_parquet(io.BytesIO(uploaded_file.read()))
            else:
                st.error('Неподдерживаемый формат файла: загрузите CSV, Excel, JSON или Parquet.')
            
            if df is not None:
                st.write('Предварительный просмотр ваших данных:')
                st.dataframe(df.head(5))

                # поищем колонки с временными признаками
                datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

                # если не нашли по dtypes, попробуем сделать из колонок datetime - вдруг колонки не в том формате
                if not datetime_cols:
                    possible_date_col = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) 
                    and pd.to_datetime(df[col], errors='coerce').notna().sum() > 0]
                    datetime_cols = possible_date_col
                
                # если и так нет - выдаем ошибку
                if not datetime_cols:
                    st.error('Временные признаки не найдены! Выберите другой файл.')
                else:
                    select_time_col = st.selectbox('Выберите доступную колонку с временем', datetime_cols)
                    df[select_time_col] = pd.to_datetime(df[select_time_col])
                    target_col = [col for col in df.columns if col != select_time_col]
                    select_target = st.selectbox('Выберите таргет.', [col for col in target_col])
                    if select_time_col and target_col:
                        st.success(f'Вы выбрали временную колонку {select_time_col} и таргет {select_target}.')
                    else:
                        st.write('Выберите временный признак и таргет.')
                
                time_df = df[[select_time_col, select_target]].copy()
                time_df.set_index(select_time_col, inplace=True)
                st.session_state.time_df = time_df
                st.markdown('<h3 style="text-align: center;">Новый DataFrame, с которым будем работать:</h3>', unsafe_allow_html=True)
                st.markdown("""
                <style>
                .dataframe {
                margin-left: auto;
                margin-right: auto;
                width: 80%;
                border: 1px solid black;
                }
                </style>
                """, unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center;'>{time_df.head(5).to_html(classes='table table-striped table bordered', index=True)}</div>", unsafe_allow_html=True)

                if st.button('Перейти к графикам!'):
                    st.session_state.page = 'page_2'


        except Exception as e:
            st.error(f'Ошибка при загрузке файла: {e}')


    else:
        st.stop()

def page_2():
    st.markdown('<h2 style="text-align: center;">Вторая страница аналитики по вашему DF: строим графики, смотрим r2 и mae, обучаем модели и предсказываем таргет.</h2>', unsafe_allow_html=True)
    time_df = st.session_state.time_df
    time_df = time_df.dropna()

    st.markdown("""
    <style>
    .stButton>button {
    width: 170px !important;
    height: 70px !important;
    font-size: 12px !important;
    font-weight: bold !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.write('Какой период времени рассмотрим?')
    col1, col2, col3, col4 = st.columns(4)
    selected_plot = 'nothing'

    with col1:
        if st.button('Неделя'):
            selected_plot = 'week'
    with col2:
        if st.button('Месяц'):
            selected_plot = 'month'
    with col3:
        if st.button('Квартал'):
            selected_plot = 'quar'
    with col4:
        if st.button('Год'):
            selected_plot = 'year'

    if selected_plot == 'week':
        figure = plt.figure(figsize=(18,8))
        plot_sub_sum = time_df.resample('W-Mon').sum()
        plot_sub_mean = time_df.resample('W-Mon').mean()

        #среднее по неделям
        plt.subplot(1, 2, 1)
        plt.plot(plot_sub_mean.index, plot_sub_mean[plot_sub_mean.columns[0]], label='Среднее продаж', color='orange')
        plt.title('Средние продажи понедельно')
        plt.xlabel('Даты')
        plt.ylabel('Продажи')
        plt.legend()

        # суммарное по неделям
        plt.subplot(1, 2, 2)
        plt.plot(plot_sub_sum.index, plot_sub_sum[plot_sub_sum.columns[0]], label='Сумма продаж', color='blue')
        plt.title('Суммарные продажи понедельно')
        plt.xlabel('Даты')
        plt.ylabel('Продажи')
        plt.legend()

        st.pyplot(figure)
    elif selected_plot == 'month':
        figure = plt.figure(figsize=(16,8))
        plot_sub_sum = time_df.resample('M').sum()
        plot_sub_mean = time_df.resample('M').mean()

        #среднее по месяцам
        plt.subplot(1, 2, 1)
        plt.plot(plot_sub_mean.index, plot_sub_mean[plot_sub_mean.columns[0]], label='Среднее продаж', color='orange')
        plt.title('Средние продажи понедельно')
        plt.xlabel('Даты')
        plt.ylabel('Продажи')
        plt.legend()

        # суммарное по месяцам
        plt.subplot(1, 2, 2)
        plt.plot(plot_sub_sum.index, plot_sub_sum[plot_sub_sum.columns[0]], label='Сумма продаж', color='blue')
        plt.title('Суммарные продажи понедельно')
        plt.xlabel('Даты')
        plt.ylabel('Продажи')
        plt.legend()

        st.pyplot(figure)
    elif selected_plot == 'quar':
        figure = plt.figure(figsize=(16,8))
        plot_sub_sum = time_df.resample('Q').sum()
        plot_sub_mean = time_df.resample('Q').mean()

        #среднее по кварталам
        plt.subplot(1, 2, 1)
        plt.plot(plot_sub_mean.index, plot_sub_mean[plot_sub_mean.columns[0]], label='Среднее продаж', color='orange')
        plt.title('Средние продажи понедельно')
        plt.xlabel('Даты')
        plt.ylabel('Продажи')
        plt.legend()

        # суммарное по кварталам
        plt.subplot(1, 2, 2)
        plt.plot(plot_sub_sum.index, plot_sub_sum[plot_sub_sum.columns[0]], label='Сумма продаж', color='blue')
        plt.title('Суммарные продажи понедельно')
        plt.xlabel('Даты')
        plt.ylabel('Продажи')
        plt.legend()

        st.pyplot(figure)
    elif selected_plot == 'year':
        figure = plt.figure(figsize=(16,8))
        plot_sub_sum = time_df.resample('Y').sum()
        plot_sub_mean = time_df.resample('Y').mean()

        #среднее по годам
        plt.subplot(1, 2, 1)
        plt.plot(plot_sub_mean.index, plot_sub_mean[plot_sub_mean.columns[0]], label='Среднее продаж', color='orange')
        plt.title('Средние продажи понедельно')
        plt.xlabel('Даты')
        plt.ylabel('Продажи')
        plt.legend()

        # суммарное по годам
        plt.subplot(1, 2, 2)
        plt.plot(plot_sub_sum.index, plot_sub_sum[plot_sub_sum.columns[0]], label='Сумма продаж', color='blue')
        plt.title('Суммарные продажи понедельно')
        plt.xlabel('Даты')
        plt.ylabel('Продажи')
        plt.legend()

        st.pyplot(figure)
    elif selected_plot == 'nothing':
        st.write('Ожидаю выбора графика...')
    else:
        st.stop()

    st.markdown('<h4 style="text-align: center;">Выберите период, с которым будем работать:</h4>', unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    mark = 'nothing'
    with col5:
        if st.button('Отчет понедельно'):
            mark = 'week'
    with col6:
        if st.button('Отчет по месяцам'):
            mark = 'month'
    with col7:
        if st.button('Отчет по кварталам'):
            mark = 'quar'
    with col8:
        if st.button('Отчет по годам'):
            mark = 'year'
    
    if mark == 'nothing':
        st.write('Выберите период для отчетности.')
    elif mark == 'week':
        plot_sub_sum = time_df.resample('W-Mon').sum()
        target = plot_sub_sum.columns[0]

        figure1 = plt.figure(figsize=(18,5))
        
        plt.plot(plot_sub_sum.index, plot_sub_sum[target], label='Сумма продаж', color='blue', marker='o')
        plt.title('Суммарные продажи понедельно')
        plt.xlabel('Даты')
        plt.ylabel('Продажи')
        plt.legend()

        st.pyplot(figure1)

        st.write('Проверим стационарность временного ряда...')
        result = adfuller(plot_sub_sum[target])
        if result[1] < 0.05:
            st.markdown(f'<h5 style="text-align: center;">Метод Дики-Фуллера показывает... {result[1]:.8f}, что доказывает статичность временного ряда!</h5>', unsafe_allow_html=True)
            adfuller_mark = 'good'
        else:
            st.markdown(f'<h5 style="text-align: center;">Метод Дики-Фуллера показывает... {result[1]:.8f}, что доказывает отсутствие статичности временного ряда...</h5>', unsafe_allow_html=True)
            adfuller_mark = 'bad'
        
        st.markdown(f'<h4 style="text-align: center;">Теперь посмотрим на скользящее среднее...</h4>', unsafe_allow_html=True)

        window_size_moving = 1
        rolling_mean = plot_sub_sum[target].rolling(window=window_size_moving, closed='left').mean()
        moving_avg_mae = np.round(mean_absolute_error(plot_sub_sum[window_size_moving:], rolling_mean[window_size_moving:]), 4)
        moving_avg_r2 = np.round(r2_score(plot_sub_sum[window_size_moving:], rolling_mean[window_size_moving:]), 4)

        st.markdown(f'<h6 style="text-align: center;">Метрики для вашего временного ряда при скользящем среднем: R2: {moving_avg_r2}, MAE: {moving_avg_mae}</h6>', unsafe_allow_html=True)

        figure2 = plt.figure(figsize=(18, 5))
        plt.plot(plot_sub_sum, label='real')
        plt.plot(rolling_mean, label='pred', marker='v')
        plt.legend()
        plt.title(f'Скользящее среднее при window = {window_size_moving}')
        st.pyplot(figure2)

        st.markdown(f'<h4 style="text-align: center;">Теперь посмотрим на взвешенное скользящее среднее...</h4>', unsafe_allow_html=True)

        def weighted_moving_average(x, n, weights):
            weights = np.array(weights) # веса
            wma = x.rolling(n).apply(lambda x: np.dot(x, weights[:len(x)]) / weights[:len(x)].sum(), raw=True).to_list() 
            result = pd.Series(wma, index=x.index).shift(1)   
            return result
        
        def calculate_weights(target_column, n):
            weights = np.abs(target_column) / np.mean(np.abs(target_column))
            weights[weights < 0] = 0.0001
            return weights[:n]

        weights = calculate_weights(plot_sub_sum[target], 20)
        weighted_rolling_mean = weighted_moving_average(plot_sub_sum[target], 20, weights)

        weighted_moving_avg_mae = np.round(mean_absolute_error(plot_sub_sum[20:], weighted_rolling_mean[20:]), 4)
        weighted_moving_avg_r2 = np.round(r2_score(plot_sub_sum[20:], weighted_rolling_mean[20:]), 4)

        st.markdown(f'<h6 style="text-align: center;">Метрики для вашего временного ряда при взвешенном скользящем среднем: R2: {weighted_moving_avg_r2}, MAE: {weighted_moving_avg_mae}</h6>', unsafe_allow_html=True)

        figure3 = plt.figure(figsize=(18, 5))
        plt.plot(plot_sub_sum, label='real')
        plt.plot(weighted_rolling_mean, label='pred', marker='v')
        plt.legend()
        plt.title(f'Взвешенное скользящее среднее при window = 20')
        st.pyplot(figure3)

        st.markdown(f'<h4 style="text-align: center;">Декомпозируем данные для установки тренда, сезонности, шумов.</h4>', unsafe_allow_html=True)

        fig, (ax1,ax2,ax3, ax4) = plt.subplots(4,1, figsize=(15,8))
        res = seasonal_decompose(plot_sub_sum[target])
        res.observed.plot(ax=ax1)
        res.trend.plot(ax=ax2)
        res.seasonal.plot(ax=ax3)
        res.resid.plot(ax=ax4)
        ax1.set_ylabel('Наблюдения')
        ax2.set_ylabel('Тренд')
        ax3.set_ylabel('Сезонность')
        ax4.set_ylabel('Шумы (остатки)')
        st.pyplot(fig)

        st.markdown(f'<h4 style="text-align: center;">Построим графики автокорреляции и частичной автокорреляции</h4>', unsafe_allow_html=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        n_lags = 52   
        acf = plot_acf(plot_sub_sum, ax=ax1, lags=n_lags)
        pacf = plot_pacf(plot_sub_sum, ax=ax2, lags=n_lags)
        st.pyplot(fig)

        st.markdown(f'<h4 style="text-align: center;">Перейдем к обучению моделей и предсказанию таргета</h4>', unsafe_allow_html=True)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(plot_sub_sum.values.reshape(-1, 1))
        data_scaled_df = pd.DataFrame(data_scaled, columns=['scaled_target'], index=plot_sub_sum.index)

        # SARIMAX
        model = SARIMAX(data_scaled_df['scaled_target'], order=(1, 1, 1), seasonal_order=(1,1,1,52))
        model_fit = model.fit()

        forecast = model_fit.get_forecast(steps=104)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()

        mse_sarimax = mean_squared_error(data_scaled_df['scaled_target'][-52:], forecast_mean[:52])
        r2_sarimax = r2_score(data_scaled_df['scaled_target'][-52:], forecast_mean[:52])

        st.markdown(f'<h6 style="text-align: center;">Метрики в модели SARIMAX: R2: {r2_sarimax}, MSE: {mse_sarimax}</h6>', unsafe_allow_html=True)

        figure4 = plt.figure(figsize=(12, 6))
        plt.plot(data_scaled_df['scaled_target'], label='Исходные данные')
        plt.plot(forecast_mean, label='Прогноз', color='red')
        plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.5)
        plt.legend()
        st.pyplot(figure4)

        # Prophet
        data_prophet = data_scaled_df['scaled_target'].reset_index().rename(columns={data_scaled_df.index.name: 'ds', 'scaled_target': 'y'})
        train_size = 0.8 #сколько хотим сделать train
        split_date = data_prophet.index[int(len(data_prophet) * train_size)]

        train_df = data_prophet[data_prophet.index <= split_date]
        test_df = data_prophet[data_prophet.index > split_date]
        
        pr_model = Prophet()
        pr_model.fit(train_df)

        season_period = 52
        n_of_future_pred = 5 * season_period
        future = pr_model.make_future_dataframe(periods=n_of_future_pred, freq='W-Mon')
        forecast = pr_model.predict(future)

        forecast_train = forecast[:-n_of_future_pred]   # период нашего train
        forecast_test = forecast[-n_of_future_pred: -n_of_future_pred + len(test_df)]   # период нашего test
        forecast_future = forecast[-n_of_future_pred + len(test_df):]

        prophet_mae_test = np.round(mean_absolute_error(test_df['y'], forecast_test['yhat']), 4)
        prophet_r2_test = np.round(r2_score(test_df['y'], forecast_test['yhat']), 4)

        st.markdown(f'<h6 style="text-align: center;">Метрики в модели Prophet: R2: {prophet_r2_test}, MSE: {prophet_mae_test}</h6>', unsafe_allow_html=True)

        figure5 = plt.figure(figsize=(20, 8))
        plt.plot(plot_sub_sum[target], label='true_data', marker='o')
        plt.plot(forecast_train['ds'], forecast_train['yhat'], marker='v', linestyle=':', label='forecast train')
        plt.plot(forecast_test['ds'], forecast_test['yhat'], marker='v', linestyle=':', label='forecast test')
        plt.plot(forecast_future['ds'], forecast_future['yhat'], marker='v', linestyle=':', label='forecast future', color='blue')
        plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)
        st.pyplot(figure5)


        

    if st.button('Вернуться на первую страницу!'):
        st.session_state.page = 'page_1'

if st.session_state.page == 'page_1':
    page_1()
else:
    page_2()