"""
Модуль расчета метрик качества прогнозирования
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict


class MetricsCalculator:
    """Класс для расчета метрик качества прогнозов"""
    
    def __init__(self):
        pass
    
    def calculate_mae(self,
                     actual: pd.Series,
                     forecast: pd.Series) -> float:
        """
        Вычисляет Mean Absolute Error (MAE)
        
        MAE = mean(|actual - forecast|)
        
        Parameters:
        -----------
        actual : pd.Series
            Фактические значения
        forecast : pd.Series
            Прогнозируемые значения
            
        Returns:
        --------
        float
            Значение MAE
        """
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        if mask.sum() == 0:
            return np.nan
        
        return np.mean(np.abs(actual[mask] - forecast[mask]))
    
    def calculate_rmse(self,
                      actual: pd.Series,
                      forecast: pd.Series) -> float:
        """
        Вычисляет Root Mean Squared Error (RMSE)
        
        RMSE = sqrt(mean((actual - forecast)^2))
        
        Parameters:
        -----------
        actual : pd.Series
            Фактические значения
        forecast : pd.Series
            Прогнозируемые значения
            
        Returns:
        --------
        float
            Значение RMSE
        """
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        if mask.sum() == 0:
            return np.nan
        
        return np.sqrt(np.mean((actual[mask] - forecast[mask]) ** 2))
    
    def calculate_wape(self,
                      actual: pd.Series,
                      forecast: pd.Series) -> float:
        """
        Вычисляет Weighted Absolute Percentage Error (WAPE)
        
        WAPE = sum(|actual - forecast|) / sum(actual) * 100
        
        Parameters:
        -----------
        actual : pd.Series
            Фактические значения
        forecast : pd.Series
            Прогнозируемые значения
            
        Returns:
        --------
        float
            Значение WAPE в процентах
        """
        mask = ~(np.isnan(actual) | np.isnan(forecast))
        if mask.sum() == 0:
            return np.nan
        
        actual_filtered = actual[mask]
        forecast_filtered = forecast[mask]
        
        if actual_filtered.sum() == 0:
            return np.nan
        
        numerator = np.abs(actual_filtered - forecast_filtered).sum()
        denominator = actual_filtered.sum()
        
        return (numerator / denominator) * 100
    
    def calculate_mape(self,
                       actual: pd.Series,
                       forecast: pd.Series) -> float:
        """
        Вычисляет Mean Absolute Percentage Error (MAPE)
        
        MAPE = mean(|actual - forecast| / actual) * 100
        
        Parameters:
        -----------
        actual : pd.Series
            Фактические значения
        forecast : pd.Series
            Прогнозируемые значения
            
        Returns:
        --------
        float
            Значение MAPE в процентах
        """
        mask = ~(np.isnan(actual) | np.isnan(forecast)) & (actual != 0)
        if mask.sum() == 0:
            return np.nan
        
        return np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
    
    def calculate_metric(self,
                        actual: pd.DataFrame,
                        forecast: pd.DataFrame,
                        metric: str = 'mae',
                        id_col: str = 'unique_id',
                        date_col: str = 'ds',
                        actual_col: str = 'y',
                        forecast_col: Optional[str] = None) -> float:
        """
        Вычисляет метрику для датафреймов с прогнозами
        
        Parameters:
        -----------
        actual : pd.DataFrame
            Датафрейм с фактическими значениями
        forecast : pd.DataFrame
            Датафрейм с прогнозами
        metric : str
            Название метрики ('mae', 'rmse', 'wape', 'mape')
        id_col : str
            Колонка с идентификатором ряда
        date_col : str
            Колонка с датами
        actual_col : str
            Колонка с фактическими значениями
        forecast_col : str, optional
            Колонка с прогнозами (если не указана, берется первая числовая колонка)
            
        Returns:
        --------
        float
            Значение метрики
        """
        # Объединяем факт и прогноз
        merged = actual[[id_col, date_col, actual_col]].merge(
            forecast[[id_col, date_col] + [c for c in forecast.columns 
                                          if c not in [id_col, date_col]]],
            on=[id_col, date_col],
            how='inner'
        )
        
        if len(merged) == 0:
            return np.nan
        
        # Определяем колонку с прогнозом
        if forecast_col is None:
            forecast_cols = [c for c in forecast.columns 
                           if c not in [id_col, date_col]]
            if len(forecast_cols) == 0:
                return np.nan
            forecast_col = forecast_cols[0]
        
        if forecast_col not in merged.columns:
            return np.nan
        
        actual_series = merged[actual_col]
        forecast_series = merged[forecast_col]
        
        # Вычисляем метрику
        if metric.lower() == 'mae':
            return self.calculate_mae(actual_series, forecast_series)
        elif metric.lower() == 'rmse':
            return self.calculate_rmse(actual_series, forecast_series)
        elif metric.lower() == 'wape':
            return self.calculate_wape(actual_series, forecast_series)
        elif metric.lower() == 'mape':
            return self.calculate_mape(actual_series, forecast_series)
        else:
            raise ValueError(f"Неизвестная метрика: {metric}")
    
    def calculate_all_metrics(self,
                             actual: pd.DataFrame,
                             forecast: pd.DataFrame,
                             id_col: str = 'unique_id',
                             date_col: str = 'ds',
                             actual_col: str = 'y',
                             forecast_col: Optional[str] = None) -> Dict[str, float]:
        """
        Вычисляет все метрики
        
        Parameters:
        -----------
        actual : pd.DataFrame
            Датафрейм с фактическими значениями
        forecast : pd.DataFrame
            Датафрейм с прогнозами
        id_col : str
            Колонка с идентификатором ряда
        date_col : str
            Колонка с датами
        actual_col : str
            Колонка с фактическими значениями
        forecast_col : str, optional
            Колонка с прогнозами
            
        Returns:
        --------
        Dict
            Словарь с метриками
        """
        return {
            'MAE': self.calculate_metric(actual, forecast, 'mae', 
                                        id_col, date_col, actual_col, forecast_col),
            'RMSE': self.calculate_metric(actual, forecast, 'rmse',
                                         id_col, date_col, actual_col, forecast_col),
            'WAPE': self.calculate_metric(actual, forecast, 'wape',
                                         id_col, date_col, actual_col, forecast_col),
            'MAPE': self.calculate_metric(actual, forecast, 'mape',
                                         id_col, date_col, actual_col, forecast_col)
        }
    
    def calculate_metrics_by_series(self,
                                   actual: pd.DataFrame,
                                   forecast: pd.DataFrame,
                                   id_col: str = 'unique_id',
                                   date_col: str = 'ds',
                                   actual_col: str = 'y',
                                   forecast_col: Optional[str] = None) -> pd.DataFrame:
        """
        Вычисляет метрики для каждого временного ряда отдельно
        
        Parameters:
        -----------
        actual : pd.DataFrame
            Датафрейм с фактическими значениями
        forecast : pd.DataFrame
            Датафрейм с прогнозами
        id_col : str
            Колонка с идентификатором ряда
        date_col : str
            Колонка с датами
        actual_col : str
            Колонка с фактическими значениями
        forecast_col : str, optional
            Колонка с прогнозами
            
        Returns:
        --------
        pd.DataFrame
            Датафрейм с метриками для каждого ряда
        """
        results = []
        
        for unique_id in actual[id_col].unique():
            actual_series = actual[actual[id_col] == unique_id]
            forecast_series = forecast[forecast[id_col] == unique_id]
            
            if len(forecast_series) == 0:
                continue
            
            # Объединяем по датам
            merged = actual_series[[date_col, actual_col]].merge(
                forecast_series[[date_col] + [c for c in forecast.columns 
                                            if c not in [id_col, date_col]]],
                on=date_col,
                how='inner'
            )
            
            if len(merged) == 0:
                continue
            
            # Определяем колонку с прогнозом
            if forecast_col is None:
                forecast_cols = [c for c in forecast.columns 
                               if c not in [id_col, date_col]]
                if len(forecast_cols) == 0:
                    continue
                fc_col = forecast_cols[0]
            else:
                fc_col = forecast_col
            
            if fc_col not in merged.columns:
                continue
            
            metrics = {
                id_col: unique_id,
                'MAE': self.calculate_mae(merged[actual_col], merged[fc_col]),
                'RMSE': self.calculate_rmse(merged[actual_col], merged[fc_col]),
                'WAPE': self.calculate_wape(merged[actual_col], merged[fc_col]),
                'MAPE': self.calculate_mape(merged[actual_col], merged[fc_col])
            }
            
            results.append(metrics)
        
        return pd.DataFrame(results)
