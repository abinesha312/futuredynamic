import torch
import numpy as np
import os
import time

class PredictorModule:
    def __init__(self, method="moving_average", window_size=10):
        """
        Initialize the predictor module for memory needs prediction.
        
        Args:
            method: Prediction method ("moving_average" or "neural_network")
            window_size: Window size for moving average calculations
        """
        self.method = method
        self.window_size = window_size
        
        # Historical data for predictions
        self.history = {
            "batch_sizes": [],
            "sequence_lengths": [],
            "memory_used": []
        }
        
        # Constants for memory estimation
        self.base_memory_per_token = 320  # bytes per token (16 bytes per parameter × 2 for k,v × 10 for overhead)
        
        # For neural network predictor
        self.nn_model = None
        if method == "neural_network":
            self._init_nn_model()
    
    def _init_nn_model(self):
        """Initialize a small neural network for memory prediction."""
        import torch.nn as nn
        
        class MemoryPredictorNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(2, 32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x):
                return self.model(x)
        
        self.nn_model = MemoryPredictorNN()
        self.nn_model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    def predict_memory_needs(self, batch_size, sequence_length, layer_name=None):
        """
        Predict memory needs for a given batch and sequence length.
        
        Args:
            batch_size: Size of the batch
            sequence_length: Length of the sequence
            layer_name: Optional name of the layer for more specific predictions
            
        Returns:
            Predicted memory needs in bytes
        """
        if self.method == "moving_average":
            return self._predict_with_moving_average(batch_size, sequence_length)
        elif self.method == "neural_network":
            return self._predict_with_neural_network(batch_size, sequence_length)
        else:
            # Fallback to simple heuristic
            return self._predict_with_heuristic(batch_size, sequence_length)
    
    def _predict_with_heuristic(self, batch_size, sequence_length):
        """Use a simple heuristic to predict memory needs."""
        # Simple linear model: memory = base_per_token * batch_size * sequence_length
        return self.base_memory_per_token * batch_size * sequence_length
    
    def _predict_with_moving_average(self, batch_size, sequence_length):
        """Use moving average of similar requests to predict memory."""
        if len(self.history["batch_sizes"]) < self.window_size:
            # Not enough history, use heuristic
            return self._predict_with_heuristic(batch_size, sequence_length)
        
        # Find similar requests in history
        similar_indices = []
        for i in range(len(self.history["batch_sizes"])):
            b_size = self.history["batch_sizes"][i]
            seq_len = self.history["sequence_lengths"][i]
            
            # Check if request is similar (within 20%)
            b_similar = 0.8 <= batch_size / b_size <= 1.2 if b_size > 0 else False
            s_similar = 0.8 <= sequence_length / seq_len <= 1.2 if seq_len > 0 else False
            
            if b_similar and s_similar:
                similar_indices.append(i)
        
        if not similar_indices:
            # No similar requests found, use heuristic
            return self._predict_with_heuristic(batch_size, sequence_length)
        
        # Calculate average memory for similar requests
        similar_memory = [self.history["memory_used"][i] for i in similar_indices]
        return np.mean(similar_memory)
    
    def _predict_with_neural_network(self, batch_size, sequence_length):
        """Use a neural network to predict memory needs."""
        if self.nn_model is None or len(self.history["batch_sizes"]) < 100:
            # Not initialized or not enough training data
            return self._predict_with_moving_average(batch_size, sequence_length)
        
        # Prepare input
        import torch
        x = torch.tensor([[batch_size, sequence_length]], dtype=torch.float32)
        x = x.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get prediction
        with torch.no_grad():
            prediction = self.nn_model(x).item()
        
        return max(prediction, self._predict_with_heuristic(batch_size, sequence_length) * 0.5)
    
    def update_history(self, batch_size, sequence_length, memory_used):
        """Update historical data with actual memory usage."""
        self.history["batch_sizes"].append(batch_size)
        self.history["sequence_lengths"].append(sequence_length)
        self.history["memory_used"].append(memory_used)
        
        # Keep history size within limits
        if len(self.history["batch_sizes"]) > 1000:
            self.history["batch_sizes"] = self.history["batch_sizes"][-1000:]
            self.history["sequence_lengths"] = self.history["sequence_lengths"][-1000:]
            self.history["memory_used"] = self.history["memory_used"][-1000:]
        
        # Periodically train neural network if using that method
        if self.method == "neural_network" and len(self.history["batch_sizes"]) >= 100:
            self._train_nn_model()
    
    def _train_nn_model(self):
        """Train the neural network on historical data."""
        if self.nn_model is None:
            return
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Prepare training data
        x = torch.tensor(list(zip(self.history["batch_sizes"], self.history["sequence_lengths"])), 
                         dtype=torch.float32)
        y = torch.tensor(self.history["memory_used"], dtype=torch.float32).unsqueeze(1)
        
        # Move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)
        y = y.to(device)
        
        # Train
        optimizer = optim.Adam(self.nn_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            output = self.nn_model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
