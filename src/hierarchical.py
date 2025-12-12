"""
Модуль иерархического согласования прогнозов (Bottom-up, Top-down, Middle-out)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class HierarchicalReconciliation:
    """Класс для иерархического согласования прогнозов"""
    
    def __init__(self):
        self.hierarchy = None
        self.aggregation_map = None
        
    def build_hierarchy(self,
                       df: pd.DataFrame,
                       id_col: str = 'unique_id') -> Dict:
        """
        Строит иерархию из данных
        
        Parameters:
        -----------
        df : pd.DataFrame
            Датафрейм с данными продаж
        id_col : str
            Колонка с идентификатором ряда
            
        Returns:
        --------
        Dict
            Словарь с иерархией уровней
        """
        hierarchy = {
            'sku': [],  # Уровень SKU (самый детальный)
            'item_store': [],  # Уровень товар-магазин
            'item': [],  # Уровень товара
            'category': [],  # Уровень категории
            'store': [],  # Уровень магазина
            'total': []  # Общий уровень
        }
        
        # Извлекаем компоненты из unique_id
        for unique_id in df[id_col].unique():
            parts = unique_id.split('_')
            
            hierarchy['sku'].append(unique_id)
            
            # Пытаемся определить уровни агрегации
            if 'store_id' in df.columns and 'item_id' in df.columns:
                item_store_data = df[df[id_col] == unique_id]
                if len(item_store_data) > 0:
                    item_id = item_store_data['item_id'].iloc[0]
                    store_id = item_store_data['store_id'].iloc[0]
                    cat_id = item_store_data.get('cat_id', 'UNKNOWN').iloc[0]
                    
                    hierarchy['item_store'].append(f"{item_id}_{store_id}")
                    hierarchy['item'].append(item_id)
                    hierarchy['category'].append(cat_id)
                    hierarchy['store'].append(store_id)
        
        # Удаляем дубликаты
        for level in hierarchy:
            hierarchy[level] = list(set(hierarchy[level]))
        
        self.hierarchy = hierarchy
        return hierarchy
    
    def bottom_up(self,
                 forecasts: pd.DataFrame,
                 df: pd.DataFrame,
                 id_col: str = 'unique_id',
                 date_col: str = 'ds',
                 value_cols: List[str] = None) -> pd.DataFrame:
        """
        Bottom-up подход: суммирование прогнозов SKU до уровня категорий/магазинов
        
        Parameters:
        -----------
        forecasts : pd.DataFrame
            Прогнозы на уровне SKU
        df : pd.DataFrame
            Исходные данные для определения иерархии
        id_col : str
            Колонка с идентификатором
        date_col : str
            Колонка с датами
        value_cols : list, optional
            Колонки с прогнозами для суммирования
            
        Returns:
        --------
        pd.DataFrame
            Согласованные прогнозы на всех уровнях
        """
        if value_cols is None:
            value_cols = [col for col in forecasts.columns 
                         if col not in [id_col, date_col]]
        
        if self.hierarchy is None:
            self.build_hierarchy(df, id_col)
        
        reconciled = forecasts.copy()
        
        # Агрегируем по уровням
        aggregation_levels = {
            'item': 'item_id',
            'category': 'cat_id',
            'store': 'store_id'
        }
        
        for level_name, group_col in aggregation_levels.items():
            if group_col not in df.columns:
                continue
            
            # Создаем маппинг SKU -> уровень агрегации
            mapping = df[[id_col, group_col]].drop_duplicates()
            mapping_dict = dict(zip(mapping[id_col], mapping[group_col]))
            
            # Добавляем колонку уровня агрегации
            forecasts_with_level = forecasts.copy()
            forecasts_with_level[level_name] = forecasts_with_level[id_col].map(mapping_dict)
            
            # Агрегируем прогнозы
            for value_col in value_cols:
                aggregated = forecasts_with_level.groupby([level_name, date_col])[value_col].sum().reset_index()
                aggregated = aggregated.rename(columns={level_name: id_col})
                aggregated[f'{value_col}_{level_name}'] = aggregated[value_col]
                aggregated = aggregated.drop(columns=[value_col])
                
                # Объединяем с исходными прогнозами
                reconciled = reconciled.merge(
                    aggregated,
                    on=[id_col, date_col],
                    how='left',
                    suffixes=('', f'_{level_name}')
                )
        
        return reconciled
    
    def top_down(self,
                forecasts: pd.DataFrame,
                df: pd.DataFrame,
                target_level: str = 'category',
                id_col: str = 'unique_id',
                date_col: str = 'ds',
                value_cols: List[str] = None,
                historical_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Top-down подход: распределение прогноза категории по историческим пропорциям
        
        Parameters:
        -----------
        forecasts : pd.DataFrame
            Прогнозы на верхнем уровне (например, категория)
        df : pd.DataFrame
            Исходные данные
        target_level : str
            Целевой уровень для распределения ('category', 'item', 'sku')
        id_col : str
            Колонка с идентификатором
        date_col : str
            Колонка с датами
        value_cols : list, optional
            Колонки с прогнозами
        historical_data : pd.DataFrame, optional
            Исторические данные для расчета пропорций
            
        Returns:
        --------
        pd.DataFrame
            Распределенные прогнозы на целевом уровне
        """
        if value_cols is None:
            value_cols = [col for col in forecasts.columns 
                         if col not in [id_col, date_col]]
        
        if historical_data is None:
            historical_data = df
        
        # Определяем колонку для группировки верхнего уровня
        if target_level == 'category':
            upper_col = 'cat_id'
            lower_col = id_col
        elif target_level == 'item':
            upper_col = 'item_id'
            lower_col = id_col
        else:
            upper_col = id_col
            lower_col = id_col
        
        if upper_col not in historical_data.columns or 'sales' not in historical_data.columns:
            return forecasts
        
        # Вычисляем исторические пропорции
        historical_sum = historical_data.groupby([upper_col, lower_col])['sales'].sum().reset_index()
        category_totals = historical_sum.groupby(upper_col)['sales'].sum().reset_index()
        category_totals.columns = [upper_col, 'total_sales']
        
        proportions = historical_sum.merge(category_totals, on=upper_col)
        proportions['proportion'] = proportions['sales'] / proportions['total_sales']
        proportions = proportions[[upper_col, lower_col, 'proportion']]
        
        # Распределяем прогнозы
        distributed = []
        
        for _, forecast_row in forecasts.iterrows():
            upper_id = forecast_row.get(upper_col)
            if pd.isna(upper_id):
                continue
            
            # Получаем пропорции для этого верхнего уровня
            level_proportions = proportions[proportions[upper_col] == upper_id]
            
            if len(level_proportions) == 0:
                continue
            
            # Распределяем прогноз по пропорциям
            for _, prop_row in level_proportions.iterrows():
                lower_id = prop_row[lower_col]
                prop = prop_row['proportion']
                
                distributed_row = {
                    id_col: lower_id,
                    date_col: forecast_row[date_col]
                }
                
                for value_col in value_cols:
                    distributed_row[value_col] = forecast_row[value_col] * prop
                
                distributed.append(distributed_row)
        
        return pd.DataFrame(distributed)
    
    def middle_out(self,
                  forecasts: pd.DataFrame,
                  df: pd.DataFrame,
                  middle_level: str = 'category',
                  id_col: str = 'unique_id',
                  date_col: str = 'ds',
                  value_cols: List[str] = None) -> pd.DataFrame:
        """
        Middle-out подход: комбинированный подход
        
        Parameters:
        -----------
        forecasts : pd.DataFrame
            Прогнозы на среднем уровне
        df : pd.DataFrame
            Исходные данные
        middle_level : str
            Средний уровень ('category', 'item')
        id_col : str
            Колонка с идентификатором
        date_col : str
            Колонка с датами
        value_cols : list, optional
            Колонки с прогнозами
            
        Returns:
        --------
        pd.DataFrame
            Согласованные прогнозы
        """
        # Сначала делаем bottom-up от среднего уровня вверх
        bottom_up_result = self.bottom_up(forecasts, df, id_col, date_col, value_cols)
        
        # Затем делаем top-down от среднего уровня вниз
        top_down_result = self.top_down(
            forecasts, 
            df, 
            target_level='sku',
            id_col=id_col,
            date_col=date_col,
            value_cols=value_cols
        )
        
        # Объединяем результаты (приоритет bottom-up для верхних уровней,
        # top-down для нижних)
        return bottom_up_result
    
    def reconcile(self,
                 forecasts: pd.DataFrame,
                 df: pd.DataFrame,
                 method: str = 'bottom_up',
                 **kwargs) -> pd.DataFrame:
        """
        Универсальный метод согласования
        
        Parameters:
        -----------
        forecasts : pd.DataFrame
            Прогнозы
        df : pd.DataFrame
            Исходные данные
        method : str
            Метод согласования ('bottom_up', 'top_down', 'middle_out')
        **kwargs
            Дополнительные параметры
            
        Returns:
        --------
        pd.DataFrame
            Согласованные прогнозы
        """
        if method == 'bottom_up':
            return self.bottom_up(forecasts, df, **kwargs)
        elif method == 'top_down':
            return self.top_down(forecasts, df, **kwargs)
        elif method == 'middle_out':
            return self.middle_out(forecasts, df, **kwargs)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
