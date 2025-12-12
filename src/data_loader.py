"""
Модуль загрузки и предобработки данных M5 Forecasting
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Класс для загрузки и предобработки данных M5 Forecasting"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.sales_df = None
        self.calendar_df = None
        self.prices_df = None
        
    def load_data(self, 
                  state: str = "CA",
                  categories: Optional[list] = None,
                  limit_rows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Загружает данные M5 Forecasting
        
        Parameters:
        -----------
        state : str
            Штат для фильтрации (по умолчанию 'CA')
        categories : list, optional
            Список категорий для фильтрации (например, ['HOBBIES', 'FOODS'])
        limit_rows : int, optional
            Ограничение количества рядов для ускорения работы
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            Кортеж из (sales_df, calendar_df, prices_df)
        """
        # Загрузка календаря
        calendar_path = self.data_dir / "calendar.csv"
        if calendar_path.exists():
            self.calendar_df = pd.read_csv(calendar_path)
        else:
            # Создаем тестовый календарь если файл отсутствует
            self.calendar_df = self._create_test_calendar()
            
        # Загрузка цен
        prices_path = self.data_dir / "sell_prices.csv"
        if prices_path.exists():
            self.prices_df = pd.read_csv(prices_path)
        else:
            self.prices_df = pd.DataFrame()
            
        # Загрузка продаж
        sales_path = self.data_dir / "sales_train_evaluation.csv"
        if sales_path.exists():
            self.sales_df = pd.read_csv(sales_path)
        else:
            # Создаем тестовые данные если файл отсутствует
            self.sales_df = self._create_test_sales_data(state, categories, limit_rows)
            return self.sales_df, self.calendar_df, self.prices_df
            
        # Фильтрация по штату
        if 'state_id' in self.sales_df.columns:
            self.sales_df = self.sales_df[self.sales_df['state_id'] == state].copy()
            
        # Фильтрация по категориям
        if categories and 'cat_id' in self.sales_df.columns:
            self.sales_df = self.sales_df[self.sales_df['cat_id'].isin(categories)].copy()
            
        # Ограничение количества рядов
        if limit_rows:
            self.sales_df = self.sales_df.head(limit_rows)
            
        return self.sales_df, self.calendar_df, self.prices_df
    
    def preprocess_sales_data(self) -> pd.DataFrame:
        """
        Преобразует данные продаж из широкого формата в длинный
        
        Returns:
        --------
        pd.DataFrame
            Данные в формате (id, date, sales)
        """
        if self.sales_df is None:
            raise ValueError("Сначала загрузите данные с помощью load_data()")
        
        # Если данные уже в длинном формате (есть колонка 'date'), возвращаем как есть
        if 'date' in self.sales_df.columns:
            sales_long = self.sales_df.copy()
            if 'unique_id' not in sales_long.columns:
                if 'id' in sales_long.columns:
                    sales_long['unique_id'] = sales_long['id']
                else:
                    sales_long['unique_id'] = (
                        sales_long.get('item_id', '') + '_' + 
                        sales_long.get('store_id', '')
                    )
            return sales_long.sort_values(['unique_id', 'date']).reset_index(drop=True)
            
        # Извлекаем колонки с датами (d_1, d_2, ...)
        date_cols = [col for col in self.sales_df.columns if col.startswith('d_')]
        
        if len(date_cols) == 0:
            raise ValueError("Не найдены колонки с датами (d_1, d_2, ...)")
        
        # Преобразуем в длинный формат
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        id_cols = [col for col in id_cols if col in self.sales_df.columns]
        
        sales_long = pd.melt(
            self.sales_df,
            id_vars=id_cols,
            value_vars=date_cols,
            var_name='day',
            value_name='sales'
        )
        
        # Преобразуем день в дату
        sales_long['day_num'] = sales_long['day'].str.extract('(\d+)').astype(int)
        
        # Если есть календарь, используем его для дат
        if self.calendar_df is not None and 'd' in self.calendar_df.columns:
            calendar_map = dict(zip(
                self.calendar_df['d'].str.extract('(\d+)')[0].astype(int),
                pd.to_datetime(self.calendar_df['date'])
            ))
            sales_long['date'] = sales_long['day_num'].map(calendar_map)
        else:
            # Создаем даты начиная с 2011-01-29
            start_date = pd.Timestamp('2011-01-29')
            sales_long['date'] = start_date + pd.to_timedelta(sales_long['day_num'] - 1, unit='D')
        
        # Удаляем временные колонки
        sales_long = sales_long.drop(['day', 'day_num'], axis=1)
        
        # Создаем уникальный идентификатор временного ряда
        if 'id' in sales_long.columns:
            sales_long['unique_id'] = sales_long['id']
        else:
            sales_long['unique_id'] = (
                sales_long.get('item_id', '') + '_' + 
                sales_long.get('store_id', '')
            )
        
        return sales_long.sort_values(['unique_id', 'date']).reset_index(drop=True)
    
    def _create_test_calendar(self) -> pd.DataFrame:
        """Создает тестовый календарь для демонстрации"""
        dates = pd.date_range(start='2011-01-29', periods=1969, freq='D')
        calendar = pd.DataFrame({
            'date': dates,
            'd': [f'd_{i+1}' for i in range(len(dates))],
            'wday': dates.dayofweek + 1,
            'month': dates.month,
            'year': dates.year,
            'event_name_1': '',
            'event_type_1': '',
            'event_name_2': '',
            'event_type_2': '',
            'snap_CA': (dates.dayofweek < 5).astype(int),
            'snap_TX': (dates.dayofweek < 5).astype(int),
            'snap_WI': (dates.dayofweek < 5).astype(int),
        })
        return calendar
    
    def _create_test_sales_data(self, state: str, categories: Optional[list], limit_rows: Optional[int]) -> pd.DataFrame:
        """Создает тестовые данные продаж для демонстрации"""
        np.random.seed(42)
        
        categories = categories or ['HOBBIES', 'FOODS']
        stores = ['CA_1', 'CA_2', 'CA_3']
        items_per_category = 50
        
        data = []
        for cat in categories:
            for store in stores:
                for item_idx in range(items_per_category):
                    item_id = f"{cat}_{item_idx:03d}"
                    unique_id = f"{item_id}_{store}"
                    
                    # Генерируем временной ряд с трендом и сезонностью
                    dates = pd.date_range(start='2011-01-29', periods=1969, freq='D')
                    
                    # Базовый уровень продаж
                    base_sales = np.random.uniform(5, 50)
                    
                    # Тренд
                    trend = np.linspace(0, base_sales * 0.3, len(dates))
                    
                    # Сезонность (недельная)
                    weekly_season = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
                    
                    # Случайный шум
                    noise = np.random.normal(0, base_sales * 0.1, len(dates))
                    
                    # Итоговые продажи
                    sales = np.maximum(0, base_sales + trend + weekly_season + noise).astype(int)
                    
                    for date, sale in zip(dates, sales):
                        data.append({
                            'id': unique_id,
                            'item_id': item_id,
                            'dept_id': f"{cat}_DEPT",
                            'cat_id': cat,
                            'store_id': store,
                            'state_id': state,
                            'date': date,
                            'sales': sale
                        })
        
        df = pd.DataFrame(data)
        
        # Преобразуем в широкий формат для совместимости
        if limit_rows:
            unique_ids = df['id'].unique()[:limit_rows]
            df = df[df['id'].isin(unique_ids)]
            
        return df



