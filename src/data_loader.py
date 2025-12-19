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
            self.calendar_df = pd.read_csv(calendar_path, low_memory=False)
        else:
            # Создаем тестовый календарь если файл отсутствует
            self.calendar_df = self._create_test_calendar()
            
        # Загрузка цен
        prices_path = self.data_dir / "sell_prices.csv"
        if prices_path.exists():
            self.prices_df = pd.read_csv(prices_path, low_memory=False)
        else:
            self.prices_df = pd.DataFrame()
            
        # Загрузка продаж с оптимизацией для больших файлов
        sales_path = self.data_dir / "sales_train_evaluation.csv"
        if sales_path.exists():
            # Для больших файлов используем chunking
            if limit_rows and limit_rows <= 5000:
                self.sales_df = pd.read_csv(sales_path, low_memory=False, nrows=limit_rows * 2)
            else:
                self.sales_df = pd.read_csv(sales_path, low_memory=False)
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
                    sales_long['unique_id'] = sales_long['id'].astype('category')
                else:
                    sales_long['unique_id'] = (
                        sales_long.get('item_id', '').astype(str) + '_' + 
                        sales_long.get('store_id', '').astype(str)
                    ).astype('category')
            
            # Оптимизация типов данных
            if 'date' in sales_long.columns:
                sales_long['date'] = pd.to_datetime(sales_long['date'])
            
            return sales_long.sort_values(['unique_id', 'date']).reset_index(drop=True)
            
        # Извлекаем колонки с датами (d_1, d_2, ...)
        date_cols = [col for col in self.sales_df.columns if col.startswith('d_')]
        
        if len(date_cols) == 0:
            raise ValueError("Не найдены колонки с датами (d_1, d_2, ...)")
        
        # Оптимизация: используем более быстрый метод для больших данных
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        id_cols = [col for col in id_cols if col in self.sales_df.columns]
        
        # Для больших данных используем более эффективный метод
        if len(self.sales_df) > 1000:
            # Используем stack для более быстрого преобразования
            id_df = self.sales_df[id_cols].copy()
            sales_values = self.sales_df[date_cols].values
            
            # Создаем индексы для повторения строк
            n_rows = len(id_df)
            n_days = len(date_cols)
            
            # Повторяем id_cols для каждого дня
            id_df_repeated = pd.DataFrame(
                np.repeat(id_df.values, n_days, axis=0),
                columns=id_cols
            )
            
            # Создаем колонки для дат и продаж
            day_nums = np.tile(np.arange(1, n_days + 1), n_rows)
            sales_flat = sales_values.flatten()
            
            sales_long = id_df_repeated.copy()
            sales_long['day_num'] = day_nums
            sales_long['sales'] = sales_flat.astype(np.float32)
        else:
            # Для малых данных используем стандартный melt
            sales_long = pd.melt(
                self.sales_df,
                id_vars=id_cols,
                value_vars=date_cols,
                var_name='day',
                value_name='sales'
            )
            sales_long['day_num'] = sales_long['day'].str.extract('(\d+)').astype(int)
        
        # Преобразуем день в дату
        if self.calendar_df is not None and 'd' in self.calendar_df.columns:
            calendar_map = dict(zip(
                self.calendar_df['d'].str.extract('(\d+)')[0].astype(int),
                pd.to_datetime(self.calendar_df['date'])
            ))
            sales_long['date'] = sales_long['day_num'].map(calendar_map)
        else:
            start_date = pd.Timestamp('2011-01-29')
            sales_long['date'] = start_date + pd.to_timedelta(sales_long['day_num'] - 1, unit='D')
        
        # Удаляем временные колонки
        if 'day' in sales_long.columns:
            sales_long = sales_long.drop(['day'], axis=1)
        sales_long = sales_long.drop(['day_num'], axis=1)
        
        # Создаем уникальный идентификатор временного ряда
        if 'id' in sales_long.columns:
            sales_long['unique_id'] = sales_long['id'].astype('category')
        else:
            sales_long['unique_id'] = (
                sales_long.get('item_id', '').astype(str) + '_' + 
                sales_long.get('store_id', '').astype(str)
            ).astype('category')
        
        # Оптимизация типов данных
        sales_long['sales'] = sales_long['sales'].astype(np.float32)
        
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
        
        # Оптимизация: уменьшаем количество дней для больших объемов
        if limit_rows and limit_rows > 1000:
            num_days = 365
        else:
            num_days = 1969
        
        dates = pd.date_range(start='2011-01-29', periods=num_days, freq='D')
        
        # Определяем количество товаров
        if limit_rows:
            total_items = limit_rows
        else:
            items_per_category = 50
            total_items = len(categories) * len(stores) * items_per_category
        
        # Оптимизация: используем более эффективный метод для больших объемов
        if total_items * num_days > 1_000_000:
            # Для очень больших объемов используем пакетную генерацию
            chunk_size = 100
            chunks = []
            
            item_counter = 0
            for cat in categories:
                for store in stores:
                    items_in_store = total_items // (len(categories) * len(stores))
                    if item_counter >= total_items:
                        break
                    
                    for chunk_start in range(0, items_in_store, chunk_size):
                        if item_counter >= total_items:
                            break
                        
                        chunk_end = min(chunk_start + chunk_size, items_in_store)
                        chunk_items = chunk_end - chunk_start
                        
                        # Генерируем данные для пакета товаров
                        item_indices = np.arange(chunk_start, chunk_end)
                        base_sales = np.random.uniform(5, 50, chunk_items)
                        trend_slopes = np.random.uniform(-0.01, 0.05, chunk_items)
                        seasonal_amps = np.random.uniform(2, 8, chunk_items)
                        noise_stds = base_sales * 0.1
                        
                        # Векторизованная генерация для всех товаров в пакете
                        day_indices = np.arange(num_days)
                        day_grid, item_grid = np.meshgrid(day_indices, np.arange(chunk_items))
                        
                        trends = (trend_slopes[:, None] * day_indices)
                        weekly_seasons = (seasonal_amps[:, None] * np.sin(2 * np.pi * day_indices / 7))
                        noises = np.random.normal(0, noise_stds[:, None], (chunk_items, num_days))
                        sales_matrix = np.maximum(0, base_sales[:, None] + trends + weekly_seasons + noises).astype(int)
                        
                        # Создаем данные для пакета
                        chunk_data = {
                            'id': [f"{cat}_{idx:03d}_{store}" for idx in item_indices for _ in range(num_days)],
                            'item_id': [f"{cat}_{idx:03d}" for idx in item_indices for _ in range(num_days)],
                            'dept_id': [f"{cat}_DEPT"] * chunk_items * num_days,
                            'cat_id': [cat] * chunk_items * num_days,
                            'store_id': [store] * chunk_items * num_days,
                            'state_id': [state] * chunk_items * num_days,
                            'date': list(dates) * chunk_items,
                            'sales': sales_matrix.flatten()
                        }
                        
                        chunks.append(pd.DataFrame(chunk_data))
                        item_counter += chunk_items
                        
                        if item_counter >= total_items:
                            break
                    
                    if item_counter >= total_items:
                        break
                if item_counter >= total_items:
                    break
            
            df = pd.concat(chunks, ignore_index=True)
        else:
            # Для меньших объемов используем стандартный метод
            unique_ids = []
            item_ids = []
            dept_ids = []
            cat_ids = []
            store_ids = []
            state_ids = []
            date_list = []
            sales_list = []
            
            item_counter = 0
            for cat in categories:
                for store in stores:
                    items_in_store = total_items // (len(categories) * len(stores))
                    if item_counter >= total_items:
                        break
                        
                    for item_idx in range(items_in_store):
                        if item_counter >= total_items:
                            break
                        
                        item_id = f"{cat}_{item_idx:03d}"
                        unique_id = f"{item_id}_{store}"
                        
                        base_sales = np.random.uniform(5, 50)
                        trend_slope = np.random.uniform(-0.01, 0.05)
                        seasonal_amp = np.random.uniform(2, 8)
                        noise_std = base_sales * 0.1
                        
                        day_indices = np.arange(len(dates))
                        trend = trend_slope * day_indices
                        weekly_season = seasonal_amp * np.sin(2 * np.pi * day_indices / 7)
                        noise = np.random.normal(0, noise_std, len(dates))
                        sales = np.maximum(0, base_sales + trend + weekly_season + noise).astype(int)
                        
                        unique_ids.extend([unique_id] * len(dates))
                        item_ids.extend([item_id] * len(dates))
                        dept_ids.extend([f"{cat}_DEPT"] * len(dates))
                        cat_ids.extend([cat] * len(dates))
                        store_ids.extend([store] * len(dates))
                        state_ids.extend([state] * len(dates))
                        date_list.extend(dates)
                        sales_list.extend(sales)
                        
                        item_counter += 1
                        
                    if item_counter >= total_items:
                        break
                if item_counter >= total_items:
                    break
            
            df = pd.DataFrame({
                'id': unique_ids,
                'item_id': item_ids,
                'dept_id': dept_ids,
                'cat_id': cat_ids,
                'store_id': store_ids,
                'state_id': state_ids,
                'date': date_list,
                'sales': sales_list
            })
        
        return df



