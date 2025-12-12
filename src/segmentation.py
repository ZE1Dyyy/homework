"""
Модуль сегментации товаров: XYZ/ABC-анализ и кластеризация по характеру спроса
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple


class DemandSegmentation:
    """Класс для сегментации товаров по характеру спроса"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_xyz_abc(self, 
                         df: pd.DataFrame,
                         id_col: str = 'unique_id',
                         value_col: str = 'sales',
                         date_col: str = 'date') -> pd.DataFrame:
        """
        Выполняет XYZ/ABC-анализ
        
        XYZ-анализ: по коэффициенту вариации (стабильность спроса)
        ABC-анализ: по объему продаж (важность товара)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Датафрейм с данными продаж
        id_col : str
            Колонка с идентификатором товара
        value_col : str
            Колонка со значениями продаж
        date_col : str
            Колонка с датами
            
        Returns:
        --------
        pd.DataFrame
            Датафрейм с сегментами XYZ и ABC
        """
        results = []
        
        for unique_id in df[id_col].unique():
            series_data = df[df[id_col] == unique_id].sort_values(date_col)
            sales = series_data[value_col].values
            
            # XYZ-анализ: коэффициент вариации
            cv = np.std(sales) / np.mean(sales) if np.mean(sales) > 0 else np.inf
            
            if cv < 0.1:
                xyz_segment = 'X'  # Стабильный спрос
            elif cv < 0.25:
                xyz_segment = 'Y'  # Нестабильный спрос
            else:
                xyz_segment = 'Z'  # Очень нестабильный спрос
            
            # ABC-анализ: общий объем продаж
            total_sales = np.sum(sales)
            
            results.append({
                id_col: unique_id,
                'total_sales': total_sales,
                'mean_sales': np.mean(sales),
                'cv': cv,
                'xyz_segment': xyz_segment
            })
        
        result_df = pd.DataFrame(results)
        
        # Определяем ABC-сегменты на основе общего объема продаж
        result_df = result_df.sort_values('total_sales', ascending=False)
        result_df['cumulative_pct'] = result_df['total_sales'].cumsum() / result_df['total_sales'].sum() * 100
        
        result_df['abc_segment'] = 'C'
        result_df.loc[result_df['cumulative_pct'] <= 80, 'abc_segment'] = 'A'
        result_df.loc[(result_df['cumulative_pct'] > 80) & (result_df['cumulative_pct'] <= 95), 'abc_segment'] = 'B'
        
        # Комбинированный сегмент
        result_df['segment'] = result_df['abc_segment'] + result_df['xyz_segment']
        
        return result_df
    
    def classify_demand_pattern(self,
                               df: pd.DataFrame,
                               id_col: str = 'unique_id',
                               value_col: str = 'sales',
                               date_col: str = 'date') -> pd.DataFrame:
        """
        Классифицирует товары по характеру спроса:
        - Smooth: регулярный стабильный спрос
        - Intermittent: прерывистый спрос (много нулей)
        - Lumpy: нерегулярный спрос с редкими большими заказами
        - Erratic: непредсказуемый спрос
        
        Parameters:
        -----------
        df : pd.DataFrame
            Датафрейм с данными продаж
        id_col : str
            Колонка с идентификатором товара
        value_col : str
            Колонка со значениями продаж
        date_col : str
            Колонка с датами
            
        Returns:
        --------
        pd.DataFrame
            Датафрейм с классификацией спроса
        """
        results = []
        
        for unique_id in df[id_col].unique():
            series_data = df[df[id_col] == unique_id].sort_values(date_col)
            sales = series_data[value_col].values
            
            # Характеристики спроса
            zero_pct = np.sum(sales == 0) / len(sales)  # Процент нулевых значений
            cv = np.std(sales) / np.mean(sales) if np.mean(sales) > 0 else np.inf
            mean_sales = np.mean(sales[sales > 0]) if np.sum(sales > 0) > 0 else 0
            
            # Классификация
            if zero_pct > 0.5:
                # Больше 50% нулей - прерывистый спрос
                if cv > 1.5:
                    pattern = 'Lumpy'  # Редкие большие заказы
                else:
                    pattern = 'Intermittent'  # Прерывистый регулярный
            elif cv < 0.5:
                pattern = 'Smooth'  # Стабильный регулярный
            else:
                pattern = 'Erratic'  # Непредсказуемый
            
            results.append({
                id_col: unique_id,
                'zero_pct': zero_pct,
                'cv': cv,
                'mean_sales': mean_sales,
                'demand_pattern': pattern
            })
        
        return pd.DataFrame(results)
    
    def cluster_series(self,
                      analysis_df: pd.DataFrame,
                      features: list = None,
                      n_clusters: int = 4) -> pd.DataFrame:
        """
        Кластеризует временные ряды по их характеристикам
        
        Parameters:
        -----------
        analysis_df : pd.DataFrame
            Датафрейм с характеристиками временных рядов
        features : list, optional
            Список признаков для кластеризации
        n_clusters : int
            Количество кластеров
            
        Returns:
        --------
        pd.DataFrame
            Датафрейм с метками кластеров
        """
        if features is None:
            features = ['cv', 'mean', 'std', 'has_trend', 'has_seasonality']
            features = [f for f in features if f in analysis_df.columns]
        
        # Подготовка данных
        X = analysis_df[features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        result_df = analysis_df.copy()
        result_df['cluster'] = clusters
        
        return result_df
    
    def get_forecasting_strategy(self, 
                                segmentation_df: pd.DataFrame,
                                demand_pattern_col: str = 'demand_pattern',
                                xyz_col: str = 'xyz_segment') -> pd.DataFrame:
        """
        Определяет стратегию прогнозирования для каждого товара
        
        Parameters:
        -----------
        segmentation_df : pd.DataFrame
            Датафрейм с сегментацией
        demand_pattern_col : str
            Колонка с паттерном спроса
        xyz_col : str
            Колонка с XYZ-сегментом
            
        Returns:
        --------
        pd.DataFrame
            Датафрейм со стратегиями прогнозирования
        """
        result_df = segmentation_df.copy()
        
        def determine_strategy(row):
            pattern = row.get(demand_pattern_col, 'Erratic')
            xyz = row.get(xyz_col, 'Z')
            
            if pattern == 'Smooth':
                return 'direct'  # Можно прогнозировать напрямую
            elif pattern == 'Intermittent':
                return 'group'  # Только в группе
            elif pattern == 'Lumpy':
                return 'group'  # Только в группе
            else:  # Erratic
                if xyz == 'X':
                    return 'direct'  # Стабильный, но непредсказуемый - пробуем напрямую
                else:
                    return 'group'  # Нестабильный и непредсказуемый - только группа
        
        result_df['forecasting_strategy'] = result_df.apply(determine_strategy, axis=1)
        
        return result_df




