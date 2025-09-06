import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import glob
import os
import pickle
import holidays
kr_holidays = holidays.KR()

def preprocess_data():
    train_df = pd.read_csv('data/train/train.csv')
    
    test_files = sorted(glob.glob('data/test/TEST_*.csv'))
    test_dfs = {}
    for file in test_files:
        df = pd.read_csv(file)
        test_name = os.path.basename(file).replace('.csv', '')
        test_dfs[test_name] = df
    
    test_df = pd.concat(test_dfs.values(), ignore_index=True)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    all_df['date'] = pd.to_datetime(all_df['영업일자'])
    all_df['restaurant_menu'] = all_df['영업장명_메뉴명']
    all_df['sales_count'] = all_df['매출수량']
    
    # Weekday one-hot encoding (0=Monday, 6=Sunday)
    all_df['weekday'] = all_df['date'].dt.dayofweek
    
    for i in range(7):
        all_df[f'weekday_{i}'] = (all_df['weekday'] == i).astype(int)
    
    all_df['holiday'] = all_df['date'].apply(lambda x: 1 if x.date() in kr_holidays or x.weekday() >= 5 else 0)
    all_df[['restaurant', 'menu']] = all_df['restaurant_menu'].str.split('_', n=1, expand=True)
    
    unique_restaurants = all_df['restaurant'].unique()
    unique_menus = all_df['menu'].unique()
    
    restaurant_to_idx = {rest: idx for idx, rest in enumerate(unique_restaurants)}
    menu_to_idx = {menu: idx for idx, menu in enumerate(unique_menus)}
    
    all_df['restaurant_idx'] = all_df['restaurant'].map(restaurant_to_idx)
    all_df['menu_idx'] = all_df['menu'].map(menu_to_idx)
    
    # Try to load pretrained embeddings
    try:
        if os.path.exists('embeddings/restaurant_embeddings.npy'):
            print("Loading pretrained embeddings...")
            restaurant_embeddings = np.load('embeddings/restaurant_embeddings.npy')
            menu_embeddings = np.load('embeddings/menu_embeddings.npy')
            
            # Load mappings to ensure consistency
            with open('embeddings/restaurant_to_idx.pkl', 'rb') as f:
                pretrained_rest_to_idx = pickle.load(f)
            with open('embeddings/menu_to_idx.pkl', 'rb') as f:
                pretrained_menu_to_idx = pickle.load(f)
            
            # Use pretrained mappings if they match
            if set(pretrained_rest_to_idx.keys()) == set(unique_restaurants):
                restaurant_to_idx = pretrained_rest_to_idx
                all_df['restaurant_idx'] = all_df['restaurant'].map(restaurant_to_idx)
            
            if set(pretrained_menu_to_idx.keys()) == set(unique_menus):
                menu_to_idx = pretrained_menu_to_idx
                all_df['menu_idx'] = all_df['menu'].map(menu_to_idx)
            
            # Get embeddings for each row
            rest_emb_dim = restaurant_embeddings.shape[1]
            menu_emb_dim = menu_embeddings.shape[1]
            
            # Create embedding columns (using first 4 dimensions for backward compatibility)
            for i in range(min(4, rest_emb_dim)):
                all_df[f'restaurant_emb_{i}'] = all_df['restaurant_idx'].map(
                    lambda idx: restaurant_embeddings[idx, i] if idx < len(restaurant_embeddings) else 0
                )
            
            for i in range(min(4, menu_emb_dim)):
                all_df[f'menu_emb_{i}'] = all_df['menu_idx'].map(
                    lambda idx: menu_embeddings[idx, i] if idx < len(menu_embeddings) else 0
                )
        else:
            raise FileNotFoundError("Embeddings not found")
    except Exception as e:
        print(f"Could not load pretrained embeddings: {e}")
        print("Using random embeddings...")
        # Random embeddings as fallback
        np.random.seed(42)
        for i in range(4):
            all_df[f'restaurant_emb_{i}'] = np.random.uniform(0, 1, len(all_df))
            all_df[f'menu_emb_{i}'] = np.random.uniform(0, 1, len(all_df))
    
    # Load all weather data (train + test)
    train_weather_df = pd.read_csv('data/train/meta/TRAIN_weather.csv')
    train_weather_df['date'] = pd.to_datetime(train_weather_df['일시'])
    
    # Load test weather files
    test_weather_files = sorted(glob.glob('data/test/meta/TEST_weather_*.csv'))
    test_weather_dfs = []
    for file in test_weather_files:
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['일시'])
        test_weather_dfs.append(df)
    
    # Combine all weather data
    all_weather_df = pd.concat([train_weather_df] + test_weather_dfs, ignore_index=True)
    
    all_weather_df['avg_temp'] = pd.to_numeric(all_weather_df['평균기온(℃)'], errors='coerce')
    all_weather_df['rainfall'] = pd.to_numeric(all_weather_df['강수량(mm)'], errors='coerce').fillna(0)
    
    # Fit scalers on all weather data to get consistent scaling
    temp_scaler = MinMaxScaler()
    rain_scaler = MinMaxScaler()
    
    all_weather_df['avg_temp_norm'] = temp_scaler.fit_transform(all_weather_df[['avg_temp']])
    all_weather_df['rainfall_norm'] = rain_scaler.fit_transform(all_weather_df[['rainfall']])
    
    weather_df = all_weather_df[['date', 'avg_temp_norm', 'rainfall_norm']]
    
    all_df = all_df.merge(weather_df, on='date', how='left')
    
    all_df['avg_temp_norm'] = all_df['avg_temp_norm'].fillna(all_df['avg_temp_norm'].mean())
    all_df['rainfall_norm'] = all_df['rainfall_norm'].fillna(0)
    
    # Menu-wise min-max scaling for sales_count
    menu_scalers = {}
    all_df['sales_count_norm'] = 0.0
    
    for menu in all_df['restaurant_menu'].unique():
        menu_mask = all_df['restaurant_menu'] == menu
        menu_data = all_df.loc[menu_mask, 'sales_count'].values.reshape(-1, 1)
        
        if menu_data.max() > 0:
            scaler = MinMaxScaler()
            all_df.loc[menu_mask, 'sales_count_norm'] = scaler.fit_transform(menu_data).flatten()
            menu_scalers[menu] = scaler
        else:
            all_df.loc[menu_mask, 'sales_count_norm'] = 0
            menu_scalers[menu] = None
    
    train_mask = all_df['date'] < '2024-01-01'
    train_processed = all_df[train_mask].copy()
    
    features = ['sales_count_norm'] + \
               [f'weekday_{i}' for i in range(7)] + \
               ['holiday'] + \
               [f'restaurant_emb_{i}' for i in range(4)] + \
               [f'menu_emb_{i}' for i in range(4)] + \
               ['avg_temp_norm', 'rainfall_norm']
    
    # Also include restaurant_idx and menu_idx for embedding lookup
    train_processed = train_processed[['date', 'restaurant_menu', 'restaurant_idx', 'menu_idx'] + features]
    
    os.makedirs('data_preprocessed', exist_ok=True)
    
    train_processed.to_csv('data_preprocessed/train_preprocessed.csv', index=False)
    
    for test_name, test_df_orig in test_dfs.items():
        test_df_orig['date'] = pd.to_datetime(test_df_orig['영업일자'])
        test_df_orig['restaurant_menu'] = test_df_orig['영업장명_메뉴명']
        test_df_orig['sales_count'] = test_df_orig['매출수량']
        
        test_processed = all_df[all_df['date'].isin(test_df_orig['date']) & 
                                all_df['restaurant_menu'].isin(test_df_orig['restaurant_menu'])].copy()
        
        test_processed = test_processed[['date', 'restaurant_menu', 'restaurant_idx', 'menu_idx'] + features]
        
        output_filename = f'data_preprocessed/{test_name}_preprocessed.csv'
        test_processed.to_csv(output_filename, index=False)
        print(f"Saved {output_filename}")
    
    # Save mappings and scalers
    torch.save({
        'restaurant_to_idx': restaurant_to_idx,
        'menu_to_idx': menu_to_idx,
    }, 'data_preprocessed/embedding_models.pt')
    
    # Save scalers separately with pickle
    with open('data_preprocessed/scalers.pkl', 'wb') as f:
        pickle.dump({
            'temp_scaler': temp_scaler,
            'rain_scaler': rain_scaler,
            'menu_scalers': menu_scalers
        }, f)
    
    print(f"Train data shape: {train_processed.shape}")
    print(f"Number of unique restaurants: {len(unique_restaurants)}")
    print(f"Number of unique menus: {len(unique_menus)}")
    print(f"\nProcessed {len(test_dfs)} test files")
    print("\nPreprocessing completed successfully!")
    print("All files saved in data_preprocessed/ folder")

if __name__ == "__main__":
    preprocess_data()