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
    
    # Ingest additional metadata: group / room / ski totals (daily) and price
    # 1) GROUP: sum across restaurant columns
    exo_cols = []
    try:
        grp = pd.read_csv('data/train/meta/TRAIN_group.csv')
        grp['date'] = pd.to_datetime(grp['영업일자'])
        group_cols = [c for c in grp.columns if c not in ['영업일자', 'date']]
        for c in group_cols:
            grp[c] = pd.to_numeric(grp[c], errors='coerce').fillna(0)
        grp['group_total'] = grp[group_cols].sum(axis=1)
        exo = grp[['date', 'group_total']].copy()
    except Exception:
        exo = pd.DataFrame(columns=['date'])

    # 2) ROOM: sum across columns
    try:
        room = pd.read_csv('data/train/meta/TRAIN_room.csv')
        room['date'] = pd.to_datetime(room['영업일자'])
        room_cols = [c for c in room.columns if c not in ['영업일자', 'date']]
        for c in room_cols:
            room[c] = pd.to_numeric(room[c], errors='coerce').fillna(0)
        room['room_total'] = room[room_cols].sum(axis=1)
        if exo.empty:
            exo = room[['date', 'room_total']].copy()
        else:
            exo = exo.merge(room[['date', 'room_total']], on='date', how='outer')
    except Exception:
        pass

    # 3) SKI: use daily total column if present; otherwise sum hours
    try:
        ski = pd.read_csv('data/train/meta/TRAIN_ski.csv')
        ski['date'] = pd.to_datetime(ski['영업일자'])
        if '1일내장객' in ski.columns:
            ski['ski_total'] = pd.to_numeric(ski['1일내장객'], errors='coerce').fillna(0)
        else:
            hour_cols = [c for c in ski.columns if c not in ['영업일자', 'date']]
            for c in hour_cols:
                ski[c] = pd.to_numeric(ski[c], errors='coerce').fillna(0)
            ski['ski_total'] = ski[hour_cols].sum(axis=1)
        if exo.empty:
            exo = ski[['date', 'ski_total']].copy()
        else:
            exo = exo.merge(ski[['date', 'ski_total']], on='date', how='outer')
    except Exception:
        pass

    # Generate lags (1/7/14/28) for exo totals
    lag_list = [1, 7, 14, 28]
    if not exo.empty:
        exo = exo.sort_values('date').reset_index(drop=True)
        for base in ['group_total', 'room_total', 'ski_total']:
            if base in exo.columns:
                for L in lag_list:
                    exo[f'{base}_lag{L}'] = exo[base].shift(L)
        # Fill NaNs with 0 then normalize each column 0-1 globally
        for c in exo.columns:
            if c == 'date':
                continue
            exo[c] = pd.to_numeric(exo[c], errors='coerce').fillna(0)
            col_min = exo[c].min()
            col_max = exo[c].max()
            scale = (col_max - col_min) if (col_max - col_min) != 0 else 1.0
            exo[c] = (exo[c] - col_min) / scale

        # Merge into all_df
        all_df = all_df.merge(exo, on='date', how='left')
        for c in exo.columns:
            if c == 'date':
                continue
            all_df[c] = all_df[c].fillna(0)

    # 4) PRICE: static per menu
    try:
        price_df = pd.read_csv('data/train/price.csv')
        price_df.rename(columns={'영업장명_메뉴명': 'restaurant_menu', '평균판매금액': 'price'}, inplace=True)
        price_df['price'] = pd.to_numeric(price_df['price'], errors='coerce')
        all_df = all_df.merge(price_df[['restaurant_menu', 'price']], on='restaurant_menu', how='left')
        # Fill missing with median
        median_price = all_df['price'].median() if not np.isnan(all_df['price'].median()) else 0.0
        all_df['price'] = all_df['price'].fillna(median_price)
        # log price and normalization
        all_df['price_log'] = np.log1p(all_df['price'])
        for col in ['price', 'price_log']:
            cmin, cmax = all_df[col].min(), all_df[col].max()
            scale = (cmax - cmin) if (cmax - cmin) != 0 else 1.0
            all_df[f'{col}_norm'] = (all_df[col] - cmin) / scale
    except Exception as e:
        print(f"[WARN] Could not load price.csv: {e}")

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
    
    # Choose features (keep minimal, add exo totals + lags + price_norm)
    exo_feature_cols = []
    if not exo.empty:
        for base in ['group_total', 'room_total', 'ski_total']:
            if base in all_df.columns:
                exo_feature_cols.append(base)
                for L in lag_list:
                    col = f'{base}_lag{L}'
                    if col in all_df.columns:
                        exo_feature_cols.append(col)

    price_cols = [c for c in ['price_norm', 'price_log_norm'] if c in all_df.columns]

    features = ['sales_count_norm'] + \
               [f'weekday_{i}' for i in range(7)] + \
               ['holiday'] + \
               [f'restaurant_emb_{i}' for i in range(4)] + \
               [f'menu_emb_{i}' for i in range(4)] + \
               ['avg_temp_norm', 'rainfall_norm'] + \
               exo_feature_cols + price_cols
    
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
