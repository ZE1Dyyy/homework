"""
Модуль движка прогнозирования с использованием Nixtla и ML-моделей
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, ETS, Naive, SeasonalNaive
    STATSFORECAST_AVAILABLE = True
except ImportError:
    STATSFORECAST_AVAILABLE = False
    print("StatsForecast не установлен. Используются упрощенные модели.")

try:
    from mlforecast import MLForecast
    from mlforecast.target_transforms import Differences
    import lightgbm as lgb
    MLFORECAST_AVAILABLE = True
except ImportError:
    MLFORECAST_AVAILABLE = False
    print("MLForecast не установлен. ML-модели недоступны.")


class ForecastingEngine:
    """Движок прогнозирования временных рядов"""
    
    def __init__(self, 
                 models: Optional[List[str]] = None,
                 horizon: int = 28):
        """
        Parameters:
        -----------
        models : list, optional
            Список моделей для использования
        horizon : int
            Горизонт прогнозирования (по умолчанию 28 дней)
        """
        self.horizon = horizon
        self.models_config = models or ['AutoARIMA', 'ETS', 'Naive']
        self.fitted_models = {}
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    id_col: str = 'unique_id',
                    date_col: str = 'date',
                    value_col: str = 'sales') -> pd.DataFrame:
        """
        Подготавливает данные для прогнозирования
        
        Parameters:
        -----------
        df : pd.DataFrame
            Датафрейм с временными рядами
        id_col : str
            Колонка с идентификатором ряда
        date_col : str
            Колонка с датами
        value_col : str
            Колонка со значениями
            
        Returns:
        --------
        pd.DataFrame
            Подготовленные данные в формате для StatsForecast
        """
        prepared = df[[id_col, date_col, value_col]].copy()
        prepared.columns = ['unique_id', 'ds', 'y']
        prepared['ds'] = pd.to_datetime(prepared['ds'])
        prepared = prepared.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        
        # Удаляем отрицательные значения
        prepared['y'] = prepared['y'].clip(lower=0)
        
        return prepared
    
    def fit_statistical_models(self, 
                               df: pd.DataFrame,
                               test_size: Optional[int] = None) -> Dict:
        """
        Обучает статистические модели (ARIMA, ETS)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Подготовленные данные
        test_size : int, optional
            Размер тестовой выборки для валидации
            
        Returns:
        --------
        Dict
            Словарь с обученными моделями
        """
        if not STATSFORECAST_AVAILABLE:
            return self._fit_simple_models(df, test_size)
        
        # Разделение на train/test
        if test_size:
            df_train = df.groupby('unique_id').apply(
                lambda x: x.iloc[:-test_size]
            ).reset_index(drop=True)
            df_test = df.groupby('unique_id').apply(
                lambda x: x.iloc[-test_size:]
            ).reset_index(drop=True)
        else:
            df_train = df
            df_test = None
        
        # Выбор моделей
        models = []
        if 'AutoARIMA' in self.models_config:
            models.append(AutoARIMA(season_length=7))
        if 'ETS' in self.models_config:
            models.append(ETS(season_length=7))
        if 'Naive' in self.models_config:
            models.append(Naive())
        if 'SeasonalNaive' in self.models_config:
            models.append(SeasonalNaive(season_length=7))
        
        if not models:
            models = [Naive()]  # Минимальная модель по умолчанию
        
        # Обучение
        sf = StatsForecast(
            df=df_train,
            models=models,
            freq='D',
            n_jobs=-1
        )
        
        self.fitted_models['statistical'] = sf
        
        return {
            'model': sf,
            'train_data': df_train,
            'test_data': df_test
        }
    
    def predict_statistical(self, 
                          model_dict: Dict,
                          horizon: Optional[int] = None) -> pd.DataFrame:
        """
        Делает прогноз с помощью статистических моделей
        
        Parameters:
        -----------
        model_dict : Dict
            Словарь с обученной моделью
        horizon : int, optional
            Горизонт прогнозирования
            
        Returns:
        --------
        pd.DataFrame
            Прогнозы
        """
        if horizon is None:
            horizon = self.horizon
        
        if not STATSFORECAST_AVAILABLE:
            return self._predict_simple(model_dict, horizon)
        
        sf = model_dict['model']
        forecasts = sf.forecast(h=horizon, level=[80, 95])
        
        return forecasts.reset_index()
    
    def fit_ml_models(self,
                     df: pd.DataFrame,
                     test_size: Optional[int] = None,
                     lags: List[int] = None) -> Dict:
        """
        Обучает ML-модели (LightGBM, CatBoost)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Подготовленные данные
        test_size : int, optional
            Размер тестовой выборки
        lags : list, optional
            Лаги для генерации признаков
            
        Returns:
        --------
        Dict
            Словарь с обученными моделями
        """
        if not MLFORECAST_AVAILABLE:
            return {}
        
        if lags is None:
            lags = [7, 14, 28]
        
        # Разделение на train/test
        if test_size:
            df_train = df.groupby('unique_id').apply(
                lambda x: x.iloc[:-test_size]
            ).reset_index(drop=True)
            df_test = df.groupby('unique_id').apply(
                lambda x: x.iloc[-test_size:]
            ).reset_index(drop=True)
        else:
            df_train = df
            df_test = None
        
        # Подготовка календарных признаков
        df_train['day_of_week'] = df_train['ds'].dt.dayofweek
        df_train['day_of_month'] = df_train['ds'].dt.day
        df_train['month'] = df_train['ds'].dt.month
        
        # Обучение LightGBM
        models = []
        if 'LightGBM' in self.models_config:
            models.append(lgb.LGBMRegressor(
                random_state=42,
                verbose=-1,
                n_estimators=100
            ))
        
        if models:
            mlf = MLForecast(
                models=models,
                freq='D',
                lags=lags,
                target_transforms=[Differences([1])],
                num_threads=4
            )
            
            mlf.fit(df_train)
            
            self.fitted_models['ml'] = mlf
            
            return {
                'model': mlf,
                'train_data': df_train,
                'test_data': df_test
            }
        
        return {}
    
    def predict_ml(self,
                  model_dict: Dict,
                  horizon: Optional[int] = None) -> pd.DataFrame:
        """
        Делает прогноз с помощью ML-моделей
        
        Parameters:
        -----------
        model_dict : Dict
            Словарь с обученной моделью
        horizon : int, optional
            Горизонт прогнозирования
            
        Returns:
        --------
        pd.DataFrame
            Прогнозы
        """
        if horizon is None:
            horizon = self.horizon
        
        if not MLFORECAST_AVAILABLE or 'model' not in model_dict:
            return pd.DataFrame()
        
        mlf = model_dict['model']
        forecasts = mlf.predict(horizon)
        
        return forecasts.reset_index()
    
    def _fit_simple_models(self, df: pd.DataFrame, test_size: Optional[int]) -> Dict:
        """Упрощенные модели если StatsForecast недоступен"""
        return {
            'model': 'simple',
            'train_data': df,
            'test_data': None
        }
    
    def _predict_simple(self, model_dict: Dict, horizon: int) -> pd.DataFrame:
        """Упрощенный прогноз если StatsForecast недоступен"""
        from scipy import stats
        
        df_train = model_dict['train_data']
        forecasts = []
        
        for unique_id in df_train['unique_id'].unique():
            series_data = df_train[df_train['unique_id'] == unique_id].sort_values('ds')
            series = series_data['y'].values
            
            if len(series) == 0:
                continue
            
            # Базовое значение (среднее последних значений)
            last_window = min(14, len(series))  # Используем последние 14 дней или все, если меньше
            last_values = series[-last_window:]
            base_value = np.mean(last_values)
            
            # Определяем тренд (линейная регрессия)
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            has_trend = p_value < 0.05 and abs(slope) > std_err
            
            # Определяем сезонность (недельная)
            seasonality = None
            if len(series) >= 14:
                # Средние значения по дням недели
                dates = pd.to_datetime(series_data['ds'].values)
                day_of_week = dates.dayofweek
                
                weekly_pattern = {}
                for day in range(7):
                    day_values = series[day_of_week == day]
                    if len(day_values) > 0:
                        weekly_pattern[day] = np.mean(day_values)
                
                if len(weekly_pattern) == 7:
                    # Нормализуем паттерн относительно среднего
                    avg_weekly = np.mean(list(weekly_pattern.values()))
                    seasonality = {day: (weekly_pattern[day] / avg_weekly - 1.0) 
                                 for day in weekly_pattern}
            
            # Создаем прогнозы
            last_date = series_data['ds'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            for i, date in enumerate(future_dates):
                forecast_value = base_value
                
                # Добавляем тренд
                if has_trend:
                    trend_component = slope * (len(series) + i)
                    forecast_value += trend_component
                
                # Добавляем сезонность
                if seasonality is not None:
                    day_of_week = date.dayofweek
                    if day_of_week in seasonality:
                        seasonal_factor = seasonality[day_of_week]
                        forecast_value = forecast_value * (1 + seasonal_factor)
                
                # Не допускаем отрицательных значений
                forecast_value = max(0, forecast_value)
                
                forecasts.append({
                    'unique_id': unique_id,
                    'ds': date,
                    'AutoARIMA': forecast_value,
                    'ETS': forecast_value,
                    'Naive': base_value  # Naive остается константой
                })
        
        return pd.DataFrame(forecasts)
    
    def select_best_model(self,
                         forecasts_dict: Dict[str, pd.DataFrame],
                         actual: pd.DataFrame,
                         metric: str = 'mae') -> str:
        """
        Выбирает лучшую модель на основе метрик
        
        Parameters:
        -----------
        forecasts_dict : Dict
            Словарь с прогнозами разных моделей
        actual : pd.DataFrame
            Фактические значения
        metric : str
            Метрика для сравнения ('mae', 'rmse', 'wape')
            
        Returns:
        --------
        str
            Название лучшей модели
        """
        try:
            from src.metrics import MetricsCalculator
        except ImportError:
            from .metrics import MetricsCalculator
        
        calculator = MetricsCalculator()
        best_model = None
        best_score = np.inf
        
        for model_name, forecast_df in forecasts_dict.items():
            if forecast_df.empty:
                continue
            
            try:
                score = calculator.calculate_metric(
                    actual=actual,
                    forecast=forecast_df,
                    metric=metric
                )
                
                if not np.isnan(score) and score < best_score:
                    best_score = score
                    best_model = model_name
            except Exception:
                continue
        
        return best_model if best_model else list(forecasts_dict.keys())[0]
