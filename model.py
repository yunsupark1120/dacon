import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pickle
from datetime import timedelta
try:
    import holidays as _holidays
    _KR_HOLIDAYS = _holidays.KR()
except Exception:
    _KR_HOLIDAYS = None

class PretrainedEmbedding(nn.Module):
    def __init__(self, pretrained_weights=None, num_items=None, embedding_dim=64, trainable=False):
        super().__init__()
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pretrained_weights),
                freeze=not trainable
            )
            self.embedding_dim = pretrained_weights.shape[1]
        else:
            self.embedding = nn.Embedding(num_items, embedding_dim)
            nn.init.xavier_uniform_(self.embedding.weight)
            self.embedding_dim = embedding_dim
    
    def forward(self, x):
        return self.embedding(x)

def load_pretrained_embeddings(embeddings_dir='embeddings'):
    """Load pretrained embeddings from directory"""
    if not os.path.exists(embeddings_dir):
        return None, None, None, None
    
    try:
        # Load embeddings
        restaurant_embeddings = np.load(os.path.join(embeddings_dir, 'restaurant_embeddings.npy'))
        menu_embeddings = np.load(os.path.join(embeddings_dir, 'menu_embeddings.npy'))
        
        # Load mappings
        with open(os.path.join(embeddings_dir, 'restaurant_to_idx.pkl'), 'rb') as f:
            restaurant_to_idx = pickle.load(f)
        with open(os.path.join(embeddings_dir, 'menu_to_idx.pkl'), 'rb') as f:
            menu_to_idx = pickle.load(f)
        
        return restaurant_embeddings, menu_embeddings, restaurant_to_idx, menu_to_idx
    except:
        return None, None, None, None

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        """
        hidden: [batch_size, 1, hidden_dim]
        encoder_outputs: [batch_size, seq_len, hidden_dim]
        """
        seq_len = encoder_outputs.size(1)
        
        hidden = hidden.repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        
        attention_scores = self.v(energy).squeeze(2)
        
        attention_weights = F.softmax(attention_scores, dim=1)
        
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        return context, attention_weights

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        """
        outputs, (hidden, cell) = self.lstm(x)
        
        outputs = self.fc(outputs)
        outputs = self.dropout(outputs)
        
        hidden = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)
        hidden = self.fc(hidden)
        
        cell = torch.cat([cell[-2,:,:], cell[-1,:,:]], dim=1)
        cell = self.fc(cell)
        
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_features, num_layers=2, dropout=0.2):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.num_layers = num_layers
        
        self.attention = Attention(hidden_dim)
        
        self.lstm = nn.LSTM(
            hidden_dim + num_features,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim * 2 + num_features, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        """
        input: [batch_size, 1, num_features]
        hidden: [num_layers, batch_size, hidden_dim]
        cell: [num_layers, batch_size, hidden_dim]
        encoder_outputs: [batch_size, seq_len, hidden_dim]
        """
        context, attention_weights = self.attention(hidden[-1:].transpose(0, 1), encoder_outputs)
        
        lstm_input = torch.cat([input, context], dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        prediction = self.fc_out(torch.cat([output, context, input], dim=2))
        
        return prediction, hidden, cell, attention_weights

class SalesPredictor(nn.Module):
    def __init__(self, 
                 num_features,
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.2,
                 input_seq_len=28,
                 output_seq_len=7,
                 restaurant_embedding=None,
                 menu_embedding=None,
                 use_embeddings=True):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.use_embeddings = use_embeddings
        
        # Add embeddings if provided
        self.restaurant_embedding = restaurant_embedding
        self.menu_embedding = menu_embedding
        
        # Adjust feature dimension if using embeddings
        if self.use_embeddings and restaurant_embedding and menu_embedding:
            embedding_dim = restaurant_embedding.embedding_dim + menu_embedding.embedding_dim
            total_features = num_features + embedding_dim
        else:
            total_features = num_features
        
        self.encoder = Encoder(total_features, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(1, hidden_dim, total_features, num_layers, dropout)
        
        self.fc_input = nn.Linear(total_features, total_features)
        self.layer_norm = nn.LayerNorm(total_features)
        self.total_features = total_features
        
    def forward(self, x, target=None, teacher_forcing_ratio=0.5, future_features=None,
                restaurant_idx=None, menu_idx=None):
        """
        x: [batch_size, input_seq_len, num_features]
        target: [batch_size, output_seq_len] (only for training)
        future_features: [batch_size, output_seq_len, num_features] (should be provided for both training and inference)
        restaurant_idx: [batch_size] restaurant indices for embedding lookup
        menu_idx: [batch_size] menu indices for embedding lookup
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Add embeddings if available
        if self.use_embeddings and self.restaurant_embedding and self.menu_embedding and restaurant_idx is not None and menu_idx is not None:
            # Get embeddings
            rest_emb = self.restaurant_embedding(restaurant_idx)  # [batch_size, rest_emb_dim]
            menu_emb = self.menu_embedding(menu_idx)  # [batch_size, menu_emb_dim]
            
            # Expand embeddings to match sequence length
            rest_emb = rest_emb.unsqueeze(1).expand(-1, seq_len, -1)
            menu_emb = menu_emb.unsqueeze(1).expand(-1, seq_len, -1)
            
            # Concatenate embeddings with features
            x = torch.cat([x, rest_emb, menu_emb], dim=2)
            
            # Also update future_features if provided
            if future_features is not None:
                future_seq_len = future_features.size(1)
                rest_emb_future = self.restaurant_embedding(restaurant_idx).unsqueeze(1).expand(-1, future_seq_len, -1)
                menu_emb_future = self.menu_embedding(menu_idx).unsqueeze(1).expand(-1, future_seq_len, -1)
                future_features = torch.cat([future_features, rest_emb_future, menu_emb_future], dim=2)
        
        x = self.layer_norm(self.fc_input(x))
        
        encoder_outputs, hidden, cell = self.encoder(x)
        
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        outputs = []
        
        # If future_features not provided, use last day as fallback (backward compatibility)
        if future_features is None:
            last_day_features = x[:, -1:, :]
            future_features = last_day_features.repeat(1, self.output_seq_len, 1)
        
        for t in range(self.output_seq_len):
            # Get future features for day t
            input_features = future_features[:, t:t+1, :].clone()
            
            # Apply teacher forcing only to sales_count (first feature)
            if t == 0:
                # First prediction uses last day's actual sales
                input_features[:, :, 0] = x[:, -1, 0:1]
            else:
                # For subsequent days, use teacher forcing or previous prediction
                if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    # Use actual sales from target
                    input_features[:, :, 0] = target[:, t-1:t]
                else:
                    # Use previous prediction
                    input_features[:, :, 0] = outputs[-1].squeeze(2)
            
            prediction, hidden, cell, _ = self.decoder(input_features, hidden, cell, encoder_outputs)
            
            outputs.append(prediction)
        
        outputs = torch.cat(outputs, dim=1)
        
        return outputs.squeeze(2)

class SalesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_seq_len=28, output_seq_len=7, stride=1):
        """
        data: preprocessed dataframe
        input_seq_len: number of days to use as input
        output_seq_len: number of days to predict
        stride: sliding window stride
        """
        self.data = data
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.stride = stride
        
        self.feature_cols = [col for col in data.columns 
                           if col not in ['date', 'restaurant_menu', 'restaurant_idx', 'menu_idx']]
        
        self.samples = self._create_sequences()
    
    def _create_sequences(self):
        samples = []

        grouped = self.data.groupby('restaurant_menu')

        for name, group in grouped:
            group = group.sort_values('date').reset_index(drop=True)
            
            if len(group) < self.input_seq_len + self.output_seq_len:
                continue
            
            # Get restaurant and menu indices (should be same for all rows in group)
            restaurant_idx = group['restaurant_idx'].iloc[0] if 'restaurant_idx' in group.columns else -1
            menu_idx = group['menu_idx'].iloc[0] if 'menu_idx' in group.columns else -1
            
            for i in range(0, len(group) - self.input_seq_len - self.output_seq_len + 1, self.stride):
                input_df = group.iloc[i:i + self.input_seq_len]
                horizon_df = group.iloc[i + self.input_seq_len:i + self.input_seq_len + self.output_seq_len]

                input_seq = input_df[self.feature_cols].values
                output_seq = horizon_df['sales_count_norm'].values
                # Backward-compat: AR future features (not used in DMH)
                future_features = horizon_df[self.feature_cols].values

                # Build future calendar features [H, 8]: weekday OHE (7) + holiday (1)
                horizon_dates = horizon_df['date'].dt.date.values
                weekdays = [d.weekday() for d in horizon_dates]
                weekday_ohe = np.eye(7, dtype=np.float32)[weekdays]  # [H,7]
                hol = np.zeros((self.output_seq_len, 1), dtype=np.float32)
                if _KR_HOLIDAYS is not None:
                    for j, d in enumerate(horizon_dates):
                        # weekend considered holiday in preprocessing; we mimic same rule here
                        hol[j, 0] = 1.0 if (d in _KR_HOLIDAYS or d.weekday() >= 5) else 0.0
                future_calendar = np.concatenate([weekday_ohe, hol], axis=1)  # [H,8]

                samples.append((input_seq, output_seq, future_features, restaurant_idx, menu_idx, future_calendar))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_seq, output_seq, future_features, restaurant_idx, menu_idx, future_calendar = self.samples[idx]
        return (
            torch.FloatTensor(input_seq),
            torch.FloatTensor(output_seq),
            torch.FloatTensor(future_features),
            torch.LongTensor([restaurant_idx]),
            torch.LongTensor([menu_idx]),
            torch.FloatTensor(future_calendar),
        )

def create_model(
    num_features: int,
    config: dict | None = None,
    use_pretrained_embeddings: bool = False,
    restaurant_embedding: nn.Embedding | None = None,
    menu_embedding: nn.Embedding | None = None,
    direct_multi_horizon: bool = False,
):
    """Factory for AR or DMH model with optional pretrained embeddings."""
    default_config = {
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.2,
        'input_seq_len': 28,
        'output_seq_len': 7,
    }
    if config:
        default_config.update(config)

    # Optionally load embedding weights if requested and not provided
    if use_pretrained_embeddings and (restaurant_embedding is None or menu_embedding is None):
        rest_emb_weights, menu_emb_weights, _, _ = load_pretrained_embeddings()
        if rest_emb_weights is not None and menu_emb_weights is not None:
            restaurant_embedding = PretrainedEmbedding(rest_emb_weights, trainable=False)
            menu_embedding = PretrainedEmbedding(menu_emb_weights, trainable=False)
            print(f"Loaded pretrained embeddings: Restaurant {rest_emb_weights.shape}, Menu {menu_emb_weights.shape}")
        else:
            print("Pretrained embeddings not found, using random initialization")

    use_emb = (restaurant_embedding is not None) and (menu_embedding is not None)

    if direct_multi_horizon:
        return SalesPredictorDMH(
            num_features=num_features,
            hidden_dim=int(default_config['hidden_dim']),
            num_layers=int(default_config['num_layers']),
            dropout=float(default_config['dropout']),
            input_seq_len=int(default_config['input_seq_len']),
            num_horizons=int(default_config['output_seq_len']),
            restaurant_embedding=restaurant_embedding,
            menu_embedding=menu_embedding,
            use_embeddings=use_emb,
        )

    return SalesPredictor(
        num_features=num_features,
        hidden_dim=int(default_config['hidden_dim']),
        num_layers=int(default_config['num_layers']),
        dropout=float(default_config['dropout']),
        input_seq_len=int(default_config['input_seq_len']),
        output_seq_len=int(default_config['output_seq_len']),
        restaurant_embedding=restaurant_embedding,
        menu_embedding=menu_embedding,
        use_embeddings=use_emb,
    )

class SalesPredictorDMH(nn.Module):
    """
    LSTM encoder + 7 parallel horizon heads (no AR roll).
    - Inputs:
        x_seq: [B, T=28, F_in]            (past window only)
        restaurant_idx: [B] (optional)    (int64)
        menu_idx: [B] (optional)           (int64)
    - Output:
        y_hat: [B, 7]  (H1..H7)
    """
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        input_seq_len: int = 28,
        num_horizons: int = 7,
        restaurant_embedding: nn.Embedding | None = None,
        menu_embedding: nn.Embedding | None = None,
        use_embeddings: bool = False,
        horizon_emb_dim: int = 8,
        nonneg: bool = True,
    ):
        super().__init__()
        self.T = int(input_seq_len)
        self.H = int(num_horizons)
        self.use_embeddings = bool(use_embeddings)
        self.nonneg = bool(nonneg)

        self.restaurant_embedding = restaurant_embedding
        self.menu_embedding = menu_embedding

        emb_dim = 0
        if self.use_embeddings and (self.restaurant_embedding is not None) and (self.menu_embedding is not None):
            emb_dim = self.restaurant_embedding.embedding_dim + self.menu_embedding.embedding_dim

        self.input_dim = num_features + emb_dim

        self.encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Shared projection of final encoder state
        self.head_prep = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Horizon conditioning (learned embedding 0..H-1)
        self.horizon_emb = nn.Embedding(self.H, horizon_emb_dim)

        # Multi-head attention to attend encoder sequence per horizon
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.query_proj = nn.Linear(hidden_dim + horizon_emb_dim, hidden_dim)

        # Calendar projection (weekday OHE [+ optional holiday]) → hidden
        self.cal_proj = nn.Linear(8, hidden_dim)  # supports [7 weekday + 1 holiday]; extra dims will be safely sliced

        # Fuse [context, z_rep, cal] then predict
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, 1)

        # Optional non-negative output
        self.softplus = nn.Softplus()

    def forward(self, x_seq: torch.Tensor,
                restaurant_idx: torch.Tensor | None = None,
                menu_idx: torch.Tensor | None = None,
                future_calendar: torch.Tensor | None = None) -> torch.Tensor:
        """
        x_seq: [B,T,F_in]
        future_calendar: [B, H, C] where C>=7 (weekday OHE) and optionally +1 holiday
        return: [B,7]
        """
        B, T, F = x_seq.shape
        x = x_seq

        if self.use_embeddings and (self.restaurant_embedding is not None) and (self.menu_embedding is not None):
            if restaurant_idx is None or menu_idx is None:
                raise ValueError("restaurant_idx and menu_idx must be provided when use_embeddings=True.")
            r = self.restaurant_embedding(restaurant_idx)  # [B,Er]
            m = self.menu_embedding(menu_idx)              # [B,Em]
            e = torch.cat([r, m], dim=-1)                  # [B,Er+Em]
            e = e.unsqueeze(1).expand(B, T, -1)            # repeat across time
            x = torch.cat([x, e], dim=-1)                  # [B,T,F+Er+Em]

        # LSTM encoder: take sequence and last hidden state
        enc_seq, (h_n, _) = self.encoder(x)    # enc_seq: [B,T,H]
        enc = h_n[-1]                          # [B, hidden]
        z = self.head_prep(enc)                # [B, hidden]

        # Build horizon indices [0..6] and expand to batch
        h_idx = torch.arange(self.H, device=x.device)      # [7]
        h_emb = self.horizon_emb(h_idx)                    # [7, Dh]
        h_emb = h_emb.unsqueeze(0).expand(B, -1, -1)       # [B,7,Dh]
        z_rep = z.unsqueeze(1).expand(B, self.H, z.shape[-1])  # [B,7,H]

        # Build per-horizon queries from [z, h_emb]
        q = torch.cat([z_rep, h_emb], dim=-1)              # [B,7,H+Dh]
        q = self.query_proj(q)                             # [B,7,H]

        # Attend encoder sequence per horizon
        ctx, _ = self.mha(q, enc_seq, enc_seq)             # [B,7,H]

        # Calendar features: expect at least 7 dims (weekday OHE), optional holiday at [:,:,7]
        if future_calendar is not None:
            # If more than 8 dims provided, slice to first 8 for safety
            cal = future_calendar
            if cal.size(-1) > 8:
                cal = cal[..., :8]
            # If only 7 provided, pad a zero holiday column
            if cal.size(-1) == 7:
                pad = torch.zeros((B, self.H, 1), device=cal.device, dtype=cal.dtype)
                cal = torch.cat([cal, pad], dim=-1)
            cal_h = self.cal_proj(cal)                     # [B,7,H]
        else:
            # No calendar provided → zeros
            cal_h = torch.zeros_like(ctx)

        fused = torch.cat([ctx, z_rep, cal_h], dim=-1)     # [B,7,3H]
        fused = self.fuse(fused)                            # [B,7,H]
        y = self.out(fused).squeeze(-1)                    # [B,7]

        if self.nonneg:
            y = self.softplus(y)
        return y
