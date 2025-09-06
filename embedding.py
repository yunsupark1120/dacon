import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import random
from datetime import datetime
import holidays
import pickle
import os

kr_holidays = holidays.KR()

class SalesDataset(Dataset):
    def __init__(self, df, negative_ratio=5):
        self.df = df
        self.negative_ratio = negative_ratio
        
        # Positive samples (actual sales > 0)
        self.positive_samples = df[df['sales_count'] > 0].copy()
        
        # All unique combinations for negative sampling
        self.all_restaurants = df['restaurant_idx'].unique()
        self.all_menus = df['menu_idx'].unique()
        self.all_dates = df['date'].unique()
        
        # Create positive set for quick lookup
        self.positive_set = set()
        for _, row in self.positive_samples.iterrows():
            self.positive_set.add((row['restaurant_idx'], row['menu_idx'], row['date']))
    
    def __len__(self):
        return len(self.positive_samples) * (1 + self.negative_ratio)
    
    def __getitem__(self, idx):
        if idx < len(self.positive_samples):
            # Positive sample
            row = self.positive_samples.iloc[idx]
            return self._create_sample(row, label=1.0)
        else:
            # Negative sample
            return self._create_negative_sample()
    
    def _create_sample(self, row, label):
        # Extract temporal features
        date = pd.to_datetime(row['date'])
        weekday = date.dayofweek
        is_holiday = 1 if date.date() in kr_holidays or weekday >= 5 else 0
        month = date.month
        day = date.day
        
        # One-hot encoding for weekday
        weekday_features = np.zeros(7)
        weekday_features[weekday] = 1
        
        # Normalize month and day
        month_norm = (month - 1) / 11  # 0-1 range
        day_norm = (day - 1) / 30  # approximately 0-1 range
        
        temporal_features = np.concatenate([
            weekday_features,
            [is_holiday, month_norm, day_norm]
        ])
        
        return {
            'restaurant_idx': row['restaurant_idx'],
            'menu_idx': row['menu_idx'],
            'temporal_features': torch.FloatTensor(temporal_features),
            'sales_count': row['sales_count'] if label == 1.0 else 0,
            'label': torch.FloatTensor([label])
        }
    
    def _create_negative_sample(self):
        # Generate random negative sample
        while True:
            rest_idx = np.random.choice(self.all_restaurants)
            menu_idx = np.random.choice(self.all_menus)
            date = np.random.choice(self.all_dates)
            
            if (rest_idx, menu_idx, date) not in self.positive_set:
                # Create a dummy row for negative sample
                row = pd.Series({
                    'restaurant_idx': rest_idx,
                    'menu_idx': menu_idx,
                    'date': date,
                    'sales_count': 0
                })
                return self._create_sample(row, label=0.0)


class MatrixFactorizationWithTemporal(nn.Module):
    def __init__(self, n_restaurants, n_menus, embedding_dim=64, temporal_dim=10):
        super().__init__()
        
        # Embeddings
        self.restaurant_embedding = nn.Embedding(n_restaurants, embedding_dim)
        self.menu_embedding = nn.Embedding(n_menus, embedding_dim)
        
        # Bias terms
        self.restaurant_bias = nn.Embedding(n_restaurants, 1)
        self.menu_bias = nn.Embedding(n_menus, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Temporal context processing
        self.temporal_net = nn.Sequential(
            nn.Linear(temporal_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Interaction layer for embeddings and temporal features
        self.interaction_net = nn.Sequential(
            nn.Linear(embedding_dim * 2 + temporal_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Xavier initialization
        nn.init.xavier_uniform_(self.restaurant_embedding.weight)
        nn.init.xavier_uniform_(self.menu_embedding.weight)
        nn.init.zeros_(self.restaurant_bias.weight)
        nn.init.zeros_(self.menu_bias.weight)
    
    def forward(self, restaurant_idx, menu_idx, temporal_features):
        # Get embeddings
        rest_emb = self.restaurant_embedding(restaurant_idx)
        menu_emb = self.menu_embedding(menu_idx)
        
        # Get biases
        rest_bias = self.restaurant_bias(restaurant_idx).squeeze()
        menu_bias = self.menu_bias(menu_idx).squeeze()
        
        # Matrix factorization score
        mf_score = (rest_emb * menu_emb).sum(dim=1)
        
        # Temporal score
        temporal_score = self.temporal_net(temporal_features).squeeze()
        
        # Interaction score (captures complex relationships)
        concat_features = torch.cat([rest_emb, menu_emb, temporal_features], dim=1)
        interaction_score = self.interaction_net(concat_features).squeeze()
        
        # Final prediction
        prediction = self.global_bias + rest_bias + menu_bias + mf_score + temporal_score + interaction_score
        
        return torch.sigmoid(prediction)
    
    def get_restaurant_embedding(self, restaurant_idx):
        return self.restaurant_embedding(restaurant_idx)
    
    def get_menu_embedding(self, menu_idx):
        return self.menu_embedding(menu_idx)


def train_embeddings(train_df, n_epochs=50, batch_size=256, learning_rate=0.001, embedding_dim=64):
    # Prepare data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get unique counts
    n_restaurants = train_df['restaurant_idx'].nunique()
    n_menus = train_df['menu_idx'].nunique()
    
    print(f"Number of restaurants: {n_restaurants}")
    print(f"Number of menus: {n_menus}")
    
    # Create dataset and dataloader
    dataset = SalesDataset(train_df, negative_ratio=3)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize model
    model = MatrixFactorizationWithTemporal(
        n_restaurants=n_restaurants,
        n_menus=n_menus,
        embedding_dim=embedding_dim,
        temporal_dim=10  # 7 weekdays + holiday + month + day
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            # Move to device
            restaurant_idx = batch['restaurant_idx'].to(device)
            menu_idx = batch['menu_idx'].to(device)
            temporal_features = batch['temporal_features'].to(device)
            labels = batch['label'].to(device)
            sales_count = batch['sales_count'].to(device)
            
            # Forward pass
            predictions = model(restaurant_idx, menu_idx, temporal_features)
            
            # Weighted loss (give more weight to samples with higher sales)
            weights = 1 + torch.log1p(sales_count.float())
            weights[labels.squeeze() == 0] = 1.0  # Normal weight for negative samples
            
            loss = (criterion(predictions.unsqueeze(1), labels) * weights.unsqueeze(1)).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        scheduler.step(avg_loss)
        
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
    
    return model


def save_embeddings(model, restaurant_to_idx, menu_to_idx, save_dir='embeddings'):
    """Save trained embeddings and mappings"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'embedding_model.pt'))
    
    # Save mappings
    with open(os.path.join(save_dir, 'restaurant_to_idx.pkl'), 'wb') as f:
        pickle.dump(restaurant_to_idx, f)
    
    with open(os.path.join(save_dir, 'menu_to_idx.pkl'), 'wb') as f:
        pickle.dump(menu_to_idx, f)
    
    # Extract and save embeddings as numpy arrays
    device = next(model.parameters()).device
    
    # Restaurant embeddings
    n_restaurants = len(restaurant_to_idx)
    restaurant_indices = torch.arange(n_restaurants).to(device)
    restaurant_embeddings = model.get_restaurant_embedding(restaurant_indices).detach().cpu().numpy()
    np.save(os.path.join(save_dir, 'restaurant_embeddings.npy'), restaurant_embeddings)
    
    # Menu embeddings
    n_menus = len(menu_to_idx)
    menu_indices = torch.arange(n_menus).to(device)
    menu_embeddings = model.get_menu_embedding(menu_indices).detach().cpu().numpy()
    np.save(os.path.join(save_dir, 'menu_embeddings.npy'), menu_embeddings)
    
    print(f"Embeddings saved to {save_dir}/")


def load_embeddings(save_dir='embeddings'):
    """Load saved embeddings and mappings"""
    # Load mappings
    with open(os.path.join(save_dir, 'restaurant_to_idx.pkl'), 'rb') as f:
        restaurant_to_idx = pickle.load(f)
    
    with open(os.path.join(save_dir, 'menu_to_idx.pkl'), 'rb') as f:
        menu_to_idx = pickle.load(f)
    
    # Load embeddings
    restaurant_embeddings = np.load(os.path.join(save_dir, 'restaurant_embeddings.npy'))
    menu_embeddings = np.load(os.path.join(save_dir, 'menu_embeddings.npy'))
    
    return restaurant_embeddings, menu_embeddings, restaurant_to_idx, menu_to_idx


if __name__ == "__main__":
    # Load and preprocess data
    print("Loading data...")
    train_df = pd.read_csv('data/train/train.csv')
    
    # Prepare data
    train_df['date'] = pd.to_datetime(train_df['영업일자'])
    train_df['restaurant_menu'] = train_df['영업장명_메뉴명']
    train_df['sales_count'] = train_df['매출수량']
    
    # Split restaurant and menu
    train_df[['restaurant', 'menu']] = train_df['restaurant_menu'].str.split('_', n=1, expand=True)
    
    # Create mappings
    unique_restaurants = train_df['restaurant'].unique()
    unique_menus = train_df['menu'].unique()
    
    restaurant_to_idx = {rest: idx for idx, rest in enumerate(unique_restaurants)}
    menu_to_idx = {menu: idx for idx, menu in enumerate(unique_menus)}
    
    train_df['restaurant_idx'] = train_df['restaurant'].map(restaurant_to_idx)
    train_df['menu_idx'] = train_df['menu'].map(menu_to_idx)
    
    # Train embeddings
    print("Training embeddings...")
    model = train_embeddings(train_df, n_epochs=5, batch_size=256, embedding_dim=16)
    
    # Save embeddings
    print("Saving embeddings...")
    save_embeddings(model, restaurant_to_idx, menu_to_idx)
    
    print("Done!")