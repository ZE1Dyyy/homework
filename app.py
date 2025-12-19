"""
Главное Streamlit приложение для системы прогнозирования спроса
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader import DataLoader
from src.time_series_analysis import TimeSeriesAnalyzer
from src.segmentation import DemandSegmentation
from src.forecasting import ForecastingEngine
from src.hierarchical import HierarchicalReconciliation
from src.metrics import MetricsCalculator
from src.visualization import Visualizer

@st.cache_data(ttl=3600)
def load_data_cached(data_dir, state, categories, limit_rows):
    """Кэшированная загрузка данных"""
    loader = DataLoader(data_dir)
    sales_df, calendar_df, prices_df = loader.load_data(
        state=state,
        categories=categories if categories else None,
        limit_rows=limit_rows
    )
    sales_long = loader.preprocess_sales_data()
    return sales_long, calendar_df, prices_df, loader

st.set_page_config(
    page_title="Система прогнозирования спроса",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'forecasts' not in st.session_state:
    st.session_state.forecasts = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

st.title("Система прогнозирования спроса")
st.markdown("---")

with st.sidebar:
    st.header("Настройки")
    
    st.subheader("Загрузка данных")
    data_source = st.radio(
        "Источник данных",
        ["M5 Dataset (Kaggle)", "Тестовые данные"],
        help="Выберите источник данных для работы"
    )
    
    if data_source == "M5 Dataset (Kaggle)":
        data_dir = st.text_input("Путь к папке с данными", value="data")
        state = st.selectbox("Штат", ["CA", "TX", "WI"], index=0)
        categories = st.multiselect(
            "Категории",
            ["HOBBIES", "FOODS", "HOUSEHOLD"],
            default=["HOBBIES", "FOODS"]
        )
        limit_rows = st.number_input("Ограничение рядов", min_value=100, max_value=10000, value=1000, step=100)
        if limit_rows > 5000:
            st.warning(f"⚠️ Загрузка {limit_rows} рядов может занять некоторое время. Рекомендуется использовать до 5000 рядов для быстрой работы.")
    else:
        state = "CA"
        categories = ["HOBBIES", "FOODS"]
        limit_rows = 500
    
    if st.button("Загрузить данные", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Загрузка данных...")
            progress_bar.progress(20)
            
            data_dir_path = data_dir if data_source == "M5 Dataset (Kaggle)" else "data"
            
            if data_source == "Тестовые данные":
                sales_long, calendar_df, prices_df, loader = load_data_cached(
                    data_dir_path, state, categories, limit_rows
                )
            else:
                loader = DataLoader(data_dir_path)
                progress_bar.progress(40)
                sales_df, calendar_df, prices_df = loader.load_data(
                    state=state,
                    categories=categories if categories else None,
                    limit_rows=limit_rows
                )
                progress_bar.progress(70)
                sales_long = loader.preprocess_sales_data()
                
            progress_bar.progress(90)
                
            st.session_state.sales_data = sales_long
            st.session_state.calendar_data = calendar_df
            st.session_state.prices_data = prices_df
            st.session_state.data_loader = loader
            st.session_state.data_loaded = True
                
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            num_series = len(sales_long['unique_id'].unique())
            num_records = len(sales_long)
            st.success(f"Загружено {num_series} временных рядов ({num_records:,} записей)")
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Ошибка загрузки данных: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    st.markdown("---")
    
    st.subheader("Параметры прогнозирования")
    
    try:
        from statsforecast import StatsForecast
        statsforecast_available = True
    except ImportError:
        statsforecast_available = False
    
    try:
        from mlforecast import MLForecast
        mlforecast_available = True
    except ImportError:
        mlforecast_available = False
    
    if not statsforecast_available and not mlforecast_available:
        st.warning("Библиотеки StatsForecast и MLForecast не установлены. Используется упрощенная модель с учетом тренда и сезонности. Для лучших результатов установите: pip install statsforecast mlforecast lightgbm catboost")
    elif not statsforecast_available:
        st.info("StatsForecast не установлен. Статистические модели недоступны.")
    elif not mlforecast_available:
        st.info("MLForecast не установлен. ML модели недоступны.")
    
    horizon = st.number_input("Горизонт прогнозирования (дней)", min_value=7, max_value=365, value=28, step=7)
    test_size = st.number_input("Размер тестовой выборки", min_value=0, max_value=365, value=28, step=7)
    
    forecasting_method = st.selectbox(
        "Метод прогнозирования",
        ["Статистические модели", "ML модели", "Оба метода"]
    )
    
    reconciliation_method = st.selectbox(
        "Метод согласования",
        ["Bottom-up", "Top-down", "Без согласования"]
    )

if not st.session_state.data_loaded:
    st.info("Пожалуйста, загрузите данные используя боковую панель")
    st.markdown("""
    ### Описание системы
    
    Эта система предназначена для:
    - Автоматического построения прогнозов на разных уровнях иерархии
    - Оценки точности прогнозов
    - Выявления проблемных зон, требующих ручного вмешательства
    
    ### Функциональность
    
    1. **Подготовка данных и сегментация**
       - Загрузка и предобработка данных
       - Анализ временных рядов (тренд, сезонность)
       - XYZ/ABC-анализ для сегментации товаров
    
    2. **Движок прогнозирования**
       - Иерархическое прогнозирование (Bottom-up, Top-down)
       - Статистические модели (ARIMA, ETS)
       - ML-модели (LightGBM)
    
    3. **Аналитика и интерфейс**
       - Интерактивный дашборд
       - Метрики качества (WAPE, MAE, RMSE)
       - Exception Management (Alerts)
        """)
else:
    sales_data = st.session_state.sales_data
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Обзор данных",
        "Анализ временных рядов",
        "Прогнозирование",
        "Метрики качества",
        "Exception Management"
    ])
    
    with tab1:
        st.header("Обзор загруженных данных")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Количество рядов", len(sales_data['unique_id'].unique()))
        with col2:
            st.metric("Период данных", 
                     f"{sales_data['date'].min().strftime('%Y-%m-%d')} - {sales_data['date'].max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("Всего наблюдений", len(sales_data))
        with col4:
            st.metric("Средние продажи", f"{sales_data['sales'].mean():.2f}")
        
        st.subheader("Пример данных")
        st.dataframe(sales_data.head(100), use_container_width=True)
        
        if 'cat_id' in sales_data.columns:
            st.subheader("Распределение по категориям")
            cat_dist = sales_data.groupby('cat_id')['unique_id'].nunique().reset_index()
            cat_dist.columns = ['Категория', 'Количество рядов']
            st.bar_chart(cat_dist.set_index('Категория'))
    
    with tab2:
        st.header("Анализ временных рядов и сегментация")
        
        if st.button("Выполнить анализ", type="primary"):
            with st.spinner("Анализ временных рядов..."):
                analyzer = TimeSeriesAnalyzer()
                analysis_results = analyzer.analyze_all_series(sales_data)
                
                st.session_state.analysis_results = analysis_results
                
                st.success("Анализ завершен")
        
        if 'analysis_results' in st.session_state:
            analysis_results = st.session_state.analysis_results
            
            st.subheader("Результаты анализа")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Рядов с трендом", 
                         analysis_results['has_trend'].sum())
                st.metric("Рядов с сезонностью", 
                         analysis_results['has_seasonality'].sum())
            
            with col2:
                st.metric("Стационарных рядов", 
                         analysis_results['is_stationary'].sum())
                st.metric("Средний CV", 
                         f"{analysis_results['cv'].mean():.2f}")
            
            st.subheader("XYZ/ABC-анализ и классификация спроса")
            
            with st.spinner("Выполнение сегментации..."):
                segmentation = DemandSegmentation()
                xyz_abc = segmentation.calculate_xyz_abc(sales_data)
                demand_patterns = segmentation.classify_demand_pattern(sales_data)
                
                segmentation_df = xyz_abc.merge(demand_patterns, on='unique_id')
                segmentation_df = segmentation_df.merge(
                    analysis_results[['unique_id', 'has_trend', 'has_seasonality']],
                    on='unique_id'
                )
                
                st.session_state.segmentation = segmentation_df
                
                segmentation_df = segmentation.get_forecasting_strategy(segmentation_df)
                st.session_state.segmentation = segmentation_df
            
            st.dataframe(segmentation_df.head(100), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Распределение по XYZ-сегментам")
                xyz_counts = segmentation_df['xyz_segment'].value_counts()
                st.bar_chart(xyz_counts)
            
            with col2:
                st.subheader("Распределение по паттернам спроса")
                pattern_counts = segmentation_df['demand_pattern'].value_counts()
                st.bar_chart(pattern_counts)
            
            st.subheader("Детальный анализ ряда")
            selected_id = st.selectbox(
                "Выберите ряд",
                options=sales_data['unique_id'].unique()[:100]
            )
            
            if selected_id:
                series_data = sales_data[sales_data['unique_id'] == selected_id].sort_values('date')
                visualizer = Visualizer()
                
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=series_data['date'],
                    y=series_data['sales'],
                    mode='lines+markers',
                    name='Продажи'
                ))
                fig.update_layout(
                    title=f'Временной ряд: {selected_id}',
                    xaxis_title='Дата',
                    yaxis_title='Продажи'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if selected_id in analysis_results['unique_id'].values:
                    row_analysis = analysis_results[analysis_results['unique_id'] == selected_id].iloc[0]
                    row_segmentation = segmentation_df[segmentation_df['unique_id'] == selected_id].iloc[0]
                    
                    st.json({
                        'Тренд': 'Есть' if row_analysis['has_trend'] else 'Нет',
                        'Направление тренда': row_analysis['trend_direction'],
                        'Сезонность': 'Есть' if row_analysis['has_seasonality'] else 'Нет',
                        'XYZ-сегмент': row_segmentation['xyz_segment'],
                        'ABC-сегмент': row_segmentation['abc_segment'],
                        'Паттерн спроса': row_segmentation['demand_pattern'],
                        'Стратегия прогнозирования': row_segmentation.get('forecasting_strategy', 'direct')
                    })
    
    with tab3:
        st.header("Прогнозирование")
        
        if st.button("Построить прогнозы", type="primary"):
            with st.spinner("Построение прогнозов..."):
                try:
                    engine = ForecastingEngine(horizon=horizon)
                    prepared_data = engine.prepare_data(sales_data)
                    
                    if test_size > 0:
                        train_data = prepared_data.groupby('unique_id').apply(
                            lambda x: x.iloc[:-test_size]
                        ).reset_index(drop=True)
                        test_data = prepared_data.groupby('unique_id').apply(
                            lambda x: x.iloc[-test_size:]
                        ).reset_index(drop=True)
                    else:
                        train_data = prepared_data
                        test_data = None
                    
                    forecasts_dict = {}
                    
                    if forecasting_method in ["Статистические модели", "Оба метода"]:
                        try:
                            stat_models = engine.fit_statistical_models(train_data, test_size=0)
                            stat_forecasts = engine.predict_statistical(stat_models, horizon=horizon)
                            if not stat_forecasts.empty:
                                forecast_cols = [c for c in stat_forecasts.columns 
                                                if c not in ['unique_id', 'ds']]
                                if forecast_cols:
                                    forecasts_dict['AutoARIMA'] = stat_forecasts
                        except Exception as e:
                            st.warning(f"Ошибка при обучении статистических моделей: {str(e)}")
                    
                    if forecasting_method in ["ML модели", "Оба метода"]:
                        try:
                            ml_models = engine.fit_ml_models(train_data, test_size=0)
                            if ml_models and 'model' in ml_models:
                                ml_forecasts = engine.predict_ml(ml_models, horizon=horizon)
                                if not ml_forecasts.empty:
                                    forecasts_dict['LightGBM'] = ml_forecasts
                        except Exception as e:
                            st.warning(f"Ошибка при обучении ML моделей: {str(e)}")
                    
                    if not forecasts_dict:
                        st.warning("Не удалось построить прогнозы. Проверьте установленные библиотеки.")
                    else:
                        best_forecast = list(forecasts_dict.values())[0]
                        
                        if reconciliation_method != "Без согласования":
                            try:
                                reconciler = HierarchicalReconciliation()
                                method_map = {
                                    "Bottom-up": "bottom_up",
                                    "Top-down": "top_down"
                                }
                                best_forecast = reconciler.reconcile(
                                    best_forecast,
                                    sales_data,
                                    method=method_map[reconciliation_method]
                                )
                            except Exception as e:
                                st.warning(f"Ошибка при согласовании прогнозов: {str(e)}")
                        
                        st.session_state.forecasts = best_forecast
                        st.session_state.test_data = test_data
                        st.session_state.forecasts_dict = forecasts_dict
                        
                        st.success("Прогнозы построены")
                
                except Exception as e:
                    st.error(f"Ошибка построения прогнозов: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        if st.session_state.forecasts is not None:
            forecasts = st.session_state.forecasts
            
            st.subheader("Результаты прогнозирования")
            
            forecast_ids = forecasts['unique_id'].unique()[:100]
            selected_forecast_id = st.selectbox(
                "Выберите ряд для визуализации",
                options=forecast_ids
            )
            
            if selected_forecast_id:
                visualizer = Visualizer()
                
                actual_data = sales_data[sales_data['unique_id'] == selected_forecast_id].copy()
                actual_data['ds'] = actual_data['date']
                actual_data['y'] = actual_data['sales']
                
                forecast_col = [c for c in forecasts.columns if c not in ['unique_id', 'ds']][0]
                fig = visualizer.plot_forecast_vs_actual(
                    actual_data,
                    forecasts,
                    selected_forecast_id,
                    forecast_col=forecast_col
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Таблица прогнозов")
                forecast_table = forecasts[forecasts['unique_id'] == selected_forecast_id]
                st.dataframe(forecast_table, use_container_width=True)
    
    with tab4:
        st.header("Метрики качества прогнозирования")
        
        if st.session_state.forecasts is not None and 'test_data' in st.session_state:
            test_data = st.session_state.test_data
            
            if test_data is not None and len(test_data) > 0:
                with st.spinner("Расчет метрик..."):
                    calculator = MetricsCalculator()
                    
                    forecasts = st.session_state.forecasts
                    forecast_col = [c for c in forecasts.columns if c not in ['unique_id', 'ds']][0]
                    
                    all_metrics = calculator.calculate_all_metrics(
                        test_data,
                        forecasts,
                        forecast_col=forecast_col
                    )
                    
                    st.subheader("Общие метрики")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MAE", f"{all_metrics['MAE']:.2f}")
                    with col2:
                        st.metric("RMSE", f"{all_metrics['RMSE']:.2f}")
                    with col3:
                        st.metric("WAPE", f"{all_metrics['WAPE']:.2f}%")
                    with col4:
                        st.metric("MAPE", f"{all_metrics['MAPE']:.2f}%")
                    
                    metrics_by_series = calculator.calculate_metrics_by_series(
                        test_data,
                        forecasts,
                        forecast_col=forecast_col
                    )
                    
                    st.session_state.metrics = metrics_by_series
                    
                    st.subheader("Метрики по временным рядам")
                    st.dataframe(metrics_by_series.sort_values('WAPE', ascending=False), use_container_width=True)
                    
                    visualizer = Visualizer()
                    fig = visualizer.plot_metrics_comparison(metrics_by_series, metric_col='WAPE', top_n=20)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Для расчета метрик необходимо разделить данные на train/test. Установите test_size > 0 в настройках.")
        else:
            st.info("Сначала постройте прогнозы на вкладке 'Прогнозирование'")
    
    with tab5:
        st.header("Exception Management - Управление исключениями")
        
        if st.session_state.metrics is not None:
            metrics_df = st.session_state.metrics
            
            st.subheader("Alerts - Товары с проблемами прогнозирования")
            
            col1, col2 = st.columns(2)
            with col1:
                threshold_wape = st.slider("Порог WAPE (%)", min_value=0, max_value=200, value=50, step=5)
            with col2:
                threshold_mae = st.slider("Порог MAE", min_value=0, max_value=1000, value=10, step=1)
            
            alerts = metrics_df[
                (metrics_df['WAPE'] > threshold_wape) | 
                (metrics_df['MAE'] > threshold_mae)
            ].sort_values('WAPE', ascending=False)
            
            st.metric("Количество проблемных рядов", len(alerts))
            
            if len(alerts) > 0:
                st.dataframe(alerts, use_container_width=True)
                
                if len(alerts) > 0:
                    st.subheader("Детальный анализ проблемного ряда")
                    problem_id = st.selectbox(
                        "Выберите проблемный ряд",
                        options=alerts['unique_id'].tolist()
                    )
                    
                    if problem_id:
                        visualizer = Visualizer()
                        
                        actual_data = sales_data[sales_data['unique_id'] == problem_id].copy()
                        actual_data['ds'] = actual_data['date']
                        actual_data['y'] = actual_data['sales']
                        
                        forecasts = st.session_state.forecasts
                        forecast_col = [c for c in forecasts.columns if c not in ['unique_id', 'ds']][0]
                        
                        fig = visualizer.plot_forecast_vs_actual(
                            actual_data,
                            forecasts,
                            problem_id,
                            forecast_col=forecast_col,
                            title=f"Проблемный ряд: {problem_id}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        row_metrics = alerts[alerts['unique_id'] == problem_id].iloc[0]
                        st.json({
                            'MAE': float(row_metrics['MAE']),
                            'RMSE': float(row_metrics['RMSE']),
                            'WAPE': float(row_metrics['WAPE']),
                            'MAPE': float(row_metrics['MAPE'])
                        })
            else:
                st.success("Нет рядов, превышающих установленные пороги")
        else:
            st.info("Сначала постройте прогнозы и рассчитайте метрики на соответствующих вкладках")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Система прогнозирования спроса | Разработано в рамках курса "Информационные системы прогнозирования и планирования цепи поставок"
</div>
""", unsafe_allow_html=True)
