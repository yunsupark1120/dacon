import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')
from model import create_model
from embedding import load_embeddings as load_pretrained_embeddings
import holidays
kr_holidays = holidays.KR()

def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying with weights_only=True...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Get model configuration
    config = checkpoint.get('config', {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'input_seq_len': 28,
        'output_seq_len': 7
    })
    
    # Check if model was trained with embeddings by looking for embedding weights
    has_embeddings = 'restaurant_embedding.embedding.weight' in checkpoint['model_state_dict']
    
    # Determine number of base features (without embeddings) from saved model state
    first_layer_weight = checkpoint['model_state_dict']['fc_input.weight']
    total_features_in_model = first_layer_weight.shape[1]
    
    # If model has embeddings, we need to subtract embedding dimensions
    if has_embeddings:
        # 64 + 64 = 128 embedding dimensions
        num_features = total_features_in_model - 32  # This gives us the base features
    else:
        num_features = total_features_in_model
    
    # Create model - it will add embeddings if needed
    model = create_model(num_features, config=config, use_pretrained_embeddings=has_embeddings)
    
    # Try to load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully with embeddings")
    except RuntimeError as e:
        print(f"Warning: Model structure mismatch. Error: {e}")
        print("Attempting partial load...")
        # Load what we can, ignore mismatched layers
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']
        
        # Filter out mismatched keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Partially loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
    
    model = model.to(device)
    model.eval()
    
    return model, config

def prepare_input_sequence(test_df, feature_cols, restaurant_to_idx=None, menu_to_idx=None):
    """Prepare last 28 days of data for each restaurant_menu"""
    sequences = {}
    future_features_dict = {}
    indices_dict = {}
    
    grouped = test_df.groupby('restaurant_menu')
    for name, group in grouped:
        group = group.sort_values('date').reset_index(drop=True)
        
        # Get indices for restaurant and menu
        if restaurant_to_idx and menu_to_idx:
            # Split restaurant_menu to get restaurant and menu
            parts = name.split('_', 1)
            if len(parts) == 2:
                restaurant, menu = parts
                rest_idx = restaurant_to_idx.get(restaurant, 0)
                menu_idx = menu_to_idx.get(menu, 0)
            else:
                rest_idx = 0
                menu_idx = 0
        else:
            rest_idx = group['restaurant_idx'].iloc[0] if 'restaurant_idx' in group.columns else 0
            menu_idx = group['menu_idx'].iloc[0] if 'menu_idx' in group.columns else 0
        
        indices_dict[name] = (rest_idx, menu_idx)
        
        # Filter feature_cols to exclude restaurant_idx and menu_idx
        filtered_cols = [col for col in feature_cols if col not in ['restaurant_idx', 'menu_idx']]
        
        last_28_days = group.tail(28)[filtered_cols].values
        sequences[name] = torch.FloatTensor(last_28_days).unsqueeze(0)  # Add batch dimension
        
        # Prepare future features for next 7 days
        last_day_features = last_28_days[-1].copy()
        future_features = []
        
        # Get last date to calculate weekdays
        last_date = group.iloc[-1]['date']
        
        for day in range(7):
            # Copy last day's features
            future_day_features = last_day_features.copy()
            
            # Update weekday encoding
            # First, reset all weekday columns to 0
            weekday_cols = [i for i, col in enumerate(filtered_cols) if col.startswith('weekday_')]
            for idx in weekday_cols:
                future_day_features[idx] = 0
            
            # Set the correct weekday to 1
            future_date = last_date + pd.Timedelta(days=day+1)
            weekday_col = f'weekday_{future_date.dayofweek}'
            if weekday_col in filtered_cols:
                weekday_idx = [i for i, col in enumerate(filtered_cols) if col == weekday_col][0]
                future_day_features[weekday_idx] = 1
            
            # Update holiday feature (1 if holiday or weekend, 0 otherwise)
            if 'holiday' in filtered_cols:
                holiday_idx = [i for i, col in enumerate(filtered_cols) if col == 'holiday'][0]
                future_day_features[holiday_idx] = 1 if future_date.date() in kr_holidays or future_date.weekday() >= 5 else 0
            
            # Sales count will be updated during prediction
            future_day_features[0] = 0  # Reset sales_count_norm
            
            future_features.append(future_day_features)
        
        future_features_dict[name] = torch.FloatTensor(np.array(future_features)).unsqueeze(0)
        
    return sequences, future_features_dict, indices_dict

def denormalize_predictions(predictions, menu_scalers, restaurant_menu_list):
    """Denormalize predictions back to original scale"""
    denormalized = {}
    
    for i, restaurant_menu in enumerate(restaurant_menu_list):
        pred = predictions[i]
        
        if restaurant_menu in menu_scalers and menu_scalers[restaurant_menu] is not None:
            scaler = menu_scalers[restaurant_menu]
            # Reshape for inverse transform
            pred_reshaped = pred.reshape(-1, 1)
            denorm = scaler.inverse_transform(pred_reshaped).flatten()
            # Round to integers and clip negative values
            denorm = np.maximum(0, np.round(denorm)).astype(int)
        else:
            # If no scaler available, assume already in correct scale
            denorm = np.maximum(0, np.round(pred * 100)).astype(int)  # Rough estimate
        
        denormalized[restaurant_menu] = denorm
    
    return denormalized

def create_submission(predictions_dict, test_name, all_menus):
    """Create submission dataframe in the required format"""
    
    # Initialize submission dataframe
    submission_rows = []
    
    # Create 7 rows for 7 days prediction
    for day in range(1, 8):
        row = {
            '영업일자': f'{test_name}+{day}일'
        }
        
        # Fill predictions for each menu
        for menu in all_menus:
            if menu in predictions_dict:
                row[menu] = predictions_dict[menu][day-1]
            else:
                row[menu] = 0
        
        submission_rows.append(row)
    
    return pd.DataFrame(submission_rows)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load scalers
    print("Loading scalers and mappings...")
    try:
        # Load scalers from pickle file
        with open('data_preprocessed/scalers.pkl', 'rb') as f:
            scaler_data = pickle.load(f)
            menu_scalers = scaler_data['menu_scalers']
    except FileNotFoundError:
        print("Warning: scalers.pkl not found. Trying to load from embedding_models.pt...")
        try:
            embedding_data = torch.load('data_preprocessed/embedding_models.pt', map_location='cpu', weights_only=False)
            menu_scalers = embedding_data.get('menu_scalers', {})
        except Exception as e:
            print(f"Warning: Could not load menu scalers: {e}")
            print("Proceeding without proper denormalization...")
            menu_scalers = {}
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, config = load_model(args.model_path, device)
    
    # Get all unique menus from sample submission
    sample_submission = pd.read_csv('data/sample_submission.csv')
    all_menus = [col for col in sample_submission.columns if col != '영업일자']
    
    # Process each test file
    test_files = sorted(glob.glob('data_preprocessed/TEST_*_preprocessed.csv'))
    all_submissions = []
    
    for test_file in test_files:
        test_name = os.path.basename(test_file).replace('_preprocessed.csv', '')
        print(f"\nProcessing {test_name}...")
        
        # Load test data
        test_df = pd.read_csv(test_file)
        test_df['date'] = pd.to_datetime(test_df['date'])
        
        # Get feature columns (excluding date and restaurant_menu)
        feature_cols = [col for col in test_df.columns if col not in ['date', 'restaurant_menu']]
        
        # Load embedding mappings if available
        try:
            _, _, restaurant_to_idx, menu_to_idx = load_pretrained_embeddings()
        except:
            restaurant_to_idx = None
            menu_to_idx = None
        
        # Prepare input sequences and future features
        input_sequences, future_features_dict, indices_dict = prepare_input_sequence(
            test_df, feature_cols, restaurant_to_idx, menu_to_idx)
        
        # Make predictions
        predictions = {}
        with torch.no_grad():
            for restaurant_menu, input_seq in tqdm(input_sequences.items(), desc="Predicting"):
                input_seq = input_seq.to(device)
                future_features = future_features_dict[restaurant_menu].to(device)
                
                # Get restaurant and menu indices
                rest_idx, menu_idx = indices_dict.get(restaurant_menu, (0, 0))
                rest_idx_tensor = torch.LongTensor([rest_idx]).to(device)
                menu_idx_tensor = torch.LongTensor([menu_idx]).to(device)
                
                # Model prediction with future features and embeddings
                output = model(input_seq, target=None, teacher_forcing_ratio=0, 
                             future_features=future_features,
                             restaurant_idx=rest_idx_tensor, menu_idx=menu_idx_tensor)
                output = output.squeeze().cpu().numpy()
                
                predictions[restaurant_menu] = output
        
        # Denormalize predictions
        restaurant_menu_list = list(predictions.keys())
        predictions_array = np.array(list(predictions.values()))
        denormalized = denormalize_predictions(predictions_array, menu_scalers, restaurant_menu_list)
        
        # Create submission for this test file
        submission = create_submission(denormalized, test_name, all_menus)
        all_submissions.append(submission)
    
    # Combine all submissions
    final_submission = pd.concat(all_submissions, ignore_index=True)
    
    # Save submission
    output_path = f'submission_{args.output_suffix}.csv'
    final_submission.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nSubmission saved to {output_path}")
    print(f"Shape: {final_submission.shape}")
    
    # Display sample predictions
    print("\nSample predictions (first 5 menus, first test file):")
    print(final_submission[['영업일자'] + all_menus[:5]].head(7))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions for test data')
    
    parser.add_argument('--model_path', type=str, default='final_model_seed42.pt',
                      help='Path to trained model checkpoint')
    parser.add_argument('--output_suffix', type=str, default='final',
                      help='Suffix for output submission file')
    
    args = parser.parse_args()
    
    main(args)