import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import GRUModel, SwinTransformerModel, HybridModel
from preprocessing import DataPreprocessor
from generate_dataset import RealisticSepsisDatasetGenerator
import warnings
warnings.filterwarnings('ignore')


class SepsisPredictor:
    """
    Complete training pipeline for sepsis prediction models
    """
    
    def __init__(self, model_type='gru', device='cpu'):
        self.device = device
        self.model_type = model_type
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
        self.preprocessor = DataPreprocessor()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def create_model(self, input_dim, hidden_dim=64):
        """Create model based on specified type"""
        if self.model_type == 'gru':
            self.model = GRUModel(input_dim, hidden_dim).to(self.device)
        elif self.model_type == 'swin':
            self.model = SwinTransformerModel(input_dim, hidden_dim).to(self.device)
        elif self.model_type == 'hybrid':
            self.model = HybridModel(input_dim, hidden_dim).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def load_data(self, X, y, batch_size=32, test_size=0.2, val_size=0.1):
        """Load and prepare data"""
        # Reshape if 3D
        original_shape = X.shape
        if X.ndim == 3:
            n_samples = X.shape[0]
        else:
            n_samples = X.shape[0]
        
        # Preprocess
        X = self.preprocessor.preprocess_pipeline(X)
        
        # Split into train, val, test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=42, stratify=y_temp
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_train = torch.FloatTensor(y_train).view(-1, 1).to(self.device)
        y_val = torch.FloatTensor(y_val).view(-1, 1).to(self.device)
        y_test = torch.FloatTensor(y_test).view(-1, 1).to(self.device)
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, X_test, y_test
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend((outputs > 0.5).cpu().detach().numpy())
            all_labels.extend(y_batch.cpu().detach().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                all_preds.extend((outputs > 0.5).cpu().detach().numpy())
                all_labels.extend(y_batch.cpu().detach().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        """Train model"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f'best_{self.model_type}_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(f'best_{self.model_type}_model.pt'))
    
    def evaluate(self, test_loader, X_test, y_test):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = self.model(X_batch)
                all_probs.extend(outputs.cpu().detach().numpy())
                all_preds.extend((outputs > 0.5).cpu().detach().numpy())
                all_labels.extend(y_batch.cpu().detach().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc_score = roc_auc_score(all_labels, all_probs)
        
        print(f"\n{'='*50}")
        print(f"Test Metrics for {self.model_type.upper()} Model:")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC:       {auc_score:.4f}")
        print(f"{'='*50}\n")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels
        }


if __name__ == "__main__":
    # Generate dataset
    print("Generating synthetic dataset...")
    generator = RealisticSepsisDatasetGenerator(n_samples=1000, sequence_length=100)
    X, y = generator.generate_dataset()
    generator.save_dataset(X, y)
    print(f"Dataset shape: X={X.shape}, y={y.shape}\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Train models
    for model_type in ['gru', 'swin', 'hybrid']:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*50}\n")
        
        predictor = SepsisPredictor(model_type=model_type, device=device)
        predictor.create_model(input_dim=10, hidden_dim=64)
        
        train_loader, val_loader, test_loader, X_test, y_test = predictor.load_data(X, y, batch_size=32)
        
        predictor.train(train_loader, val_loader, epochs=50)
        
        results = predictor.evaluate(test_loader, X_test, y_test)