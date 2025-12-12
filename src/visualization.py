"""
Модуль визуализации данных и прогнозов
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly не установлен. Визуализация будет недоступна.")
    # Создаем заглушки для совместимости
    class go:
        class Figure:
            pass


class Visualizer:
    """Класс для создания визуализаций"""
    
    def __init__(self):
        pass
    
    def plot_forecast_vs_actual(self,
                               actual: pd.DataFrame,
                               forecast: pd.DataFrame,
                               unique_id: str,
                               id_col: str = 'unique_id',
                               date_col: str = 'ds',
                               actual_col: str = 'y',
                               forecast_col: Optional[str] = None,
                               title: Optional[str] = None) -> go.Figure:
        """
        Создает график "Факт vs Прогноз"
        
        Parameters:
        -----------
        actual : pd.DataFrame
            Фактические значения
        forecast : pd.DataFrame
            Прогнозы
        unique_id : str
            Идентификатор временного ряда
        id_col : str
            Колонка с идентификатором
        date_col : str
            Колонка с датами
        actual_col : str
            Колонка с фактическими значениями
        forecast_col : str, optional
            Колонка с прогнозами
        title : str, optional
            Заголовок графика
            
        Returns:
        --------
        go.Figure
            График Plotly
        """
        if not PLOTLY_AVAILABLE:
            return go.Figure()
            
        actual_series = actual[actual[id_col] == unique_id].sort_values(date_col)
        forecast_series = forecast[forecast[id_col] == unique_id].sort_values(date_col)
        
        if len(actual_series) == 0 or len(forecast_series) == 0:
            return go.Figure()
        
        # Определяем колонку с прогнозом
        if forecast_col is None:
            forecast_cols = [c for c in forecast.columns 
                           if c not in [id_col, date_col]]
            if len(forecast_cols) == 0:
                return go.Figure()
            forecast_col = forecast_cols[0]
        
        fig = go.Figure()
        
        # Фактические значения
        fig.add_trace(go.Scatter(
            x=actual_series[date_col],
            y=actual_series[actual_col],
            mode='lines+markers',
            name='Факт',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Прогнозы
        fig.add_trace(go.Scatter(
            x=forecast_series[date_col],
            y=forecast_series[forecast_col],
            mode='lines+markers',
            name='Прогноз',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ))
        
        # Доверительные интервалы (если есть)
        if f'{forecast_col}-lo-80' in forecast_series.columns:
            fig.add_trace(go.Scatter(
                x=forecast_series[date_col],
                y=forecast_series[f'{forecast_col}-lo-80'],
                mode='lines',
                name='Нижняя граница (80%)',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_series[date_col],
                y=forecast_series[f'{forecast_col}-hi-80'],
                mode='lines',
                name='Верхняя граница (80%)',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                showlegend=True
            ))
        
        fig.update_layout(
            title=title or f'Прогноз для {unique_id}',
            xaxis_title='Дата',
            yaxis_title='Продажи',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_hierarchical_dashboard(self,
                                   data: pd.DataFrame,
                                   forecasts: pd.DataFrame,
                                   level: str = 'total',
                                   id_col: str = 'unique_id',
                                   date_col: str = 'ds',
                                   value_col: str = 'y',
                                   forecast_col: Optional[str] = None) -> go.Figure:
        """
        Создает дашборд для иерархического уровня
        
        Parameters:
        -----------
        data : pd.DataFrame
            Фактические данные
        forecasts : pd.DataFrame
            Прогнозы
        level : str
            Уровень агрегации ('total', 'category', 'item', 'sku')
        id_col : str
            Колонка с идентификатором
        date_col : str
            Колонка с датами
        value_col : str
            Колонка со значениями
        forecast_col : str, optional
            Колонка с прогнозами
            
        Returns:
        --------
        go.Figure
            График Plotly
        """
        if not PLOTLY_AVAILABLE:
            return go.Figure()
            
        if level == 'total':
            # Агрегируем все данные
            actual_agg = data.groupby(date_col)[value_col].sum().reset_index()
            forecast_agg = forecasts.groupby(date_col)[forecast_col or 'forecast'].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=actual_agg[date_col],
                y=actual_agg[value_col],
                mode='lines+markers',
                name='Факт',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_agg[date_col],
                y=forecast_agg[forecast_col or 'forecast'],
                mode='lines+markers',
                name='Прогноз',
                line=dict(color='red', width=2, dash='dash')
            ))
        else:
            # Группируем по уровню
            if level not in data.columns:
                return go.Figure()
            
            fig = go.Figure()
            
            for level_id in data[level].unique():
                level_data = data[data[level] == level_id]
                level_forecast = forecasts[forecasts.get(level, '') == level_id]
                
                if len(level_data) == 0:
                    continue
                
                actual_agg = level_data.groupby(date_col)[value_col].sum().reset_index()
                
                if len(level_forecast) > 0:
                    fc_col = forecast_col or 'forecast'
                    forecast_agg = level_forecast.groupby(date_col)[fc_col].sum().reset_index()
                else:
                    forecast_agg = pd.DataFrame(columns=[date_col, 'forecast'])
                
                fig.add_trace(go.Scatter(
                    x=actual_agg[date_col],
                    y=actual_agg[value_col],
                    mode='lines',
                    name=f'{level_id} - Факт',
                    legendgroup=level_id
                ))
                
                if len(forecast_agg) > 0:
                    fig.add_trace(go.Scatter(
                        x=forecast_agg[date_col],
                        y=forecast_agg[fc_col if forecast_col else 'forecast'],
                        mode='lines',
                        name=f'{level_id} - Прогноз',
                        line=dict(dash='dash'),
                        legendgroup=level_id
                    ))
        
        fig.update_layout(
            title=f'Дашборд уровня: {level}',
            xaxis_title='Дата',
            yaxis_title='Продажи',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def plot_metrics_comparison(self,
                               metrics_df: pd.DataFrame,
                               metric_col: str = 'WAPE',
                               top_n: int = 20) -> go.Figure:
        """
        Создает график сравнения метрик по рядам
        
        Parameters:
        -----------
        metrics_df : pd.DataFrame
            Датафрейм с метриками
        metric_col : str
            Колонка с метрикой для отображения
        top_n : int
            Количество худших рядов для отображения
            
        Returns:
        --------
        go.Figure
            График Plotly
        """
        if not PLOTLY_AVAILABLE:
            return go.Figure()
            
        if metric_col not in metrics_df.columns:
            return go.Figure()
        
        sorted_df = metrics_df.sort_values(metric_col, ascending=False).head(top_n)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=sorted_df[metric_col],
            y=sorted_df['unique_id'],
            orientation='h',
            marker=dict(color=sorted_df[metric_col], 
                       colorscale='Reds',
                       showscale=True)
        ))
        
        fig.update_layout(
            title=f'Топ {top_n} рядов с худшими метриками ({metric_col})',
            xaxis_title=metric_col,
            yaxis_title='Идентификатор ряда',
            template='plotly_white',
            height=max(400, top_n * 30)
        )
        
        return fig
    
    def plot_error_distribution(self,
                               actual: pd.DataFrame,
                               forecast: pd.DataFrame,
                               id_col: str = 'unique_id',
                               date_col: str = 'ds',
                               actual_col: str = 'y',
                               forecast_col: Optional[str] = None) -> go.Figure:
        """
        Создает график распределения ошибок
        
        Parameters:
        -----------
        actual : pd.DataFrame
            Фактические значения
        forecast : pd.DataFrame
            Прогнозы
        id_col : str
            Колонка с идентификатором
        date_col : str
            Колонка с датами
        actual_col : str
            Колонка с фактическими значениями
        forecast_col : str, optional
            Колонка с прогнозами
            
        Returns:
        --------
        go.Figure
            График Plotly
        """
        if not PLOTLY_AVAILABLE:
            return go.Figure()
            
        merged = actual[[id_col, date_col, actual_col]].merge(
            forecast[[id_col, date_col] + [c for c in forecast.columns 
                                          if c not in [id_col, date_col]]],
            on=[id_col, date_col],
            how='inner'
        )
        
        if len(merged) == 0:
            return go.Figure()
        
        if forecast_col is None:
            forecast_cols = [c for c in forecast.columns 
                           if c not in [id_col, date_col]]
            if len(forecast_cols) == 0:
                return go.Figure()
            forecast_col = forecast_cols[0]
        
        errors = merged[actual_col] - merged[forecast_col]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=50,
            name='Распределение ошибок',
            marker_color='skyblue'
        ))
        
        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="red",
            annotation_text="Ноль ошибки"
        )
        
        fig.update_layout(
            title='Распределение ошибок прогнозирования',
            xaxis_title='Ошибка (Факт - Прогноз)',
            yaxis_title='Частота',
            template='plotly_white'
        )
        
        return fig
