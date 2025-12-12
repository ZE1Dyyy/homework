"""
Модуль анализа временных рядов: определение тренда и сезонности
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels не установлен. Некоторые функции анализа будут недоступны.")


class TimeSeriesAnalyzer:
    """Класс для анализа свойств временных рядов"""
    
    def __init__(self, min_periods: int = 28):
        """
        Parameters:
        -----------
        min_periods : int
            Минимальное количество наблюдений для анализа
        """
        self.min_periods = min_periods
        
    def analyze_series(self, 
                      series: pd.Series,
                      freq: str = 'D') -> Dict[str, any]:
        """
        Анализирует временной ряд и определяет его свойства
        
        Parameters:
        -----------
        series : pd.Series
            Временной ряд с датами в индексе
        freq : str
            Частота данных ('D' для дневных)
            
        Returns:
        --------
        Dict
            Словарь с характеристиками ряда:
            - has_trend: наличие тренда
            - trend_direction: направление тренда ('up', 'down', 'none')
            - has_seasonality: наличие сезонности
            - seasonality_period: период сезонности
            - is_stationary: стационарность
            - cv: коэффициент вариации
            - mean: среднее значение
            - std: стандартное отклонение
        """
        if len(series) < self.min_periods:
            return {
                'has_trend': False,
                'trend_direction': 'none',
                'has_seasonality': False,
                'seasonality_period': None,
                'is_stationary': False,
                'cv': np.nan,
                'mean': series.mean(),
                'std': series.std()
            }
        
        # Удаляем пропуски
        series_clean = series.dropna()
        if len(series_clean) < self.min_periods:
            return self._default_result(series)
        
        # Проверка стационарности
        is_stationary = self._check_stationarity(series_clean)
        
        # Определение тренда
        trend_info = self._detect_trend(series_clean)
        
        # Определение сезонности
        seasonality_info = self._detect_seasonality(series_clean, freq)
        
        # Коэффициент вариации
        cv = series_clean.std() / series_clean.mean() if series_clean.mean() > 0 else np.inf
        
        return {
            'has_trend': trend_info['has_trend'],
            'trend_direction': trend_info['direction'],
            'trend_strength': trend_info['strength'],
            'has_seasonality': seasonality_info['has_seasonality'],
            'seasonality_period': seasonality_info['period'],
            'seasonality_strength': seasonality_info['strength'],
            'is_stationary': is_stationary,
            'cv': cv,
            'mean': series_clean.mean(),
            'std': series_clean.std(),
            'min': series_clean.min(),
            'max': series_clean.max()
        }
    
    def _check_stationarity(self, series: pd.Series) -> bool:
        """Проверяет стационарность ряда с помощью теста ADF"""
        if not STATSMODELS_AVAILABLE:
            # Упрощенная проверка: если коэффициент вариации мал, считаем стационарным
            cv = series.std() / series.mean() if series.mean() > 0 else np.inf
            return cv < 0.5
        try:
            result = adfuller(series.values)
            return result[1] < 0.05  # p-value < 0.05 означает стационарность
        except:
            return False
    
    def _detect_trend(self, series: pd.Series) -> Dict[str, any]:
        """Определяет наличие и направление тренда"""
        x = np.arange(len(series))
        y = series.values
        
        # Линейная регрессия для определения тренда
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        has_trend = p_value < 0.05 and abs(slope) > std_err
        
        if not has_trend:
            direction = 'none'
            strength = 0.0
        elif slope > 0:
            direction = 'up'
            strength = min(abs(r_value), 1.0)
        else:
            direction = 'down'
            strength = min(abs(r_value), 1.0)
        
        return {
            'has_trend': has_trend,
            'direction': direction,
            'strength': strength,
            'slope': slope
        }
    
    def _detect_seasonality(self, series: pd.Series, freq: str) -> Dict[str, any]:
        """Определяет наличие сезонности"""
        if len(series) < 14:  # Минимум 2 недели для определения недельной сезонности
            return {'has_seasonality': False, 'period': None, 'strength': 0.0}
        
        # Пробуем разные периоды сезонности
        periods_to_test = [7, 14, 30, 90, 365] if freq == 'D' else [12]
        
        best_period = None
        best_strength = 0.0
        
        for period in periods_to_test:
            if len(series) < period * 2:
                continue
                
            try:
                # Автокорреляция с лагом периода
                autocorr = series.autocorr(lag=period)
                strength = abs(autocorr) if not np.isnan(autocorr) else 0.0
                
                if strength > best_strength:
                    best_strength = strength
                    best_period = period
            except:
                continue
        
        # Также пробуем декомпозицию для недельной сезонности
        if STATSMODELS_AVAILABLE and len(series) >= 28:
            try:
                period = 7 if freq == 'D' else 12
                decomposition = seasonal_decompose(
                    series.values, 
                    model='additive', 
                    period=period,
                    extrapolate_trend='freq'
                )
                seasonal_strength = np.std(decomposition.seasonal) / np.std(series.values)
                
                if seasonal_strength > best_strength:
                    best_strength = seasonal_strength
                    best_period = period
            except:
                pass
        
        has_seasonality = best_strength > 0.3 and best_period is not None
        
        return {
            'has_seasonality': has_seasonality,
            'period': best_period if has_seasonality else None,
            'strength': best_strength
        }
    
    def _default_result(self, series: pd.Series) -> Dict[str, any]:
        """Возвращает результат по умолчанию для коротких рядов"""
        return {
            'has_trend': False,
            'trend_direction': 'none',
            'trend_strength': 0.0,
            'has_seasonality': False,
            'seasonality_period': None,
            'seasonality_strength': 0.0,
            'is_stationary': False,
            'cv': np.nan,
            'mean': series.mean() if len(series) > 0 else 0,
            'std': series.std() if len(series) > 0 else 0,
            'min': series.min() if len(series) > 0 else 0,
            'max': series.max() if len(series) > 0 else 0
        }
    
    def analyze_all_series(self, 
                          df: pd.DataFrame,
                          id_col: str = 'unique_id',
                          date_col: str = 'date',
                          value_col: str = 'sales') -> pd.DataFrame:
        """
        Анализирует все временные ряды в датафрейме
        
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
            Датафрейм с характеристиками каждого ряда
        """
        results = []
        
        for unique_id in df[id_col].unique():
            series_data = df[df[id_col] == unique_id].sort_values(date_col)
            series = pd.Series(
                series_data[value_col].values,
                index=series_data[date_col]
            )
            
            analysis = self.analyze_series(series)
            analysis[id_col] = unique_id
            
            results.append(analysis)
        
        return pd.DataFrame(results)



