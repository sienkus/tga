# ... existing imports ...
import torch.nn.functional as F
from scipy import stats
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ... existing code ...

# New section for improved model with confidence intervals
# --------------------------------------------------------

class MCDropoutModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(MCDropoutModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=True)  # Always apply dropout
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout_rate, training=True)  # Always apply dropout
        x = self.fc3(x)
        return x

# Function to predict with confidence intervals
def predict_with_confidence(model, X, num_samples=100, confidence_level=0.95):
    """
    Make predictions with confidence intervals using Monte Carlo Dropout
    
    Args:
        model: Trained PyTorch model with dropout
        X: Input features tensor
        num_samples: Number of forward passes to perform
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        mean_pred: Mean prediction
        lower_bound: Lower bound of confidence interval
        upper_bound: Upper bound of confidence interval
    """
    model.eval()  # Set to evaluation mode, but dropout will still be active
    predictions = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model(X)
            predictions.append(outputs)
    
    # Stack predictions and calculate statistics
    stacked_preds = torch.stack(predictions, dim=0)
    mean_pred = torch.mean(stacked_preds, dim=0)
    std_pred = torch.std(stacked_preds, dim=0)
    
    # Calculate confidence intervals
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    lower_bound = mean_pred - z_score * std_pred
    upper_bound = mean_pred + z_score * std_pred
    
    return mean_pred, lower_bound, upper_bound

# User-friendly prediction function
def predict_tga_at_temperature(model, scaler_X, scaler_y, temperature, num_samples=100, confidence_level=0.95):
    """
    Predict TGA value at a given temperature with confidence intervals
    
    Args:
        model: Trained PyTorch model
        scaler_X: Fitted StandardScaler for input features
        scaler_y: Fitted StandardScaler for output values
        temperature: Temperature value(s) to predict at (can be scalar or array)
        num_samples: Number of Monte Carlo samples
        confidence_level: Confidence level for intervals
        
    Returns:
        Dictionary with predictions and confidence intervals
    """
    # Convert input to appropriate format
    if isinstance(temperature, (int, float)):
        temp_array = np.array([[temperature]])
    else:
        temp_array = np.array(temperature).reshape(-1, 1)
    
    # Scale input
    temp_scaled = torch.FloatTensor(scaler_X.transform(temp_array))
    
    # Get predictions with confidence intervals
    mean_pred, lower_bound, upper_bound = predict_with_confidence(
        model, temp_scaled, num_samples, confidence_level
    )
    
    # Inverse transform to original scale
    mean_pred_orig = scaler_y.inverse_transform(mean_pred.numpy())
    lower_bound_orig = scaler_y.inverse_transform(lower_bound.numpy())
    upper_bound_orig = scaler_y.inverse_transform(upper_bound.numpy())
    
    # Prepare results
    results = {
        'temperature': temperature,
        'tga_prediction': mean_pred_orig.flatten(),
        'lower_bound': lower_bound_orig.flatten(),
        'upper_bound': upper_bound_orig.flatten(),
        'confidence_level': confidence_level
    }
    
    return results

# Function to create publication-quality plots
def plot_tga_predictions(X_train, y_train, X_test, y_test, predictions, 
                         lower_bound=None, upper_bound=None, confidence_level=0.95):
    """
    Create publication-quality plots for TGA predictions
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        predictions: Model predictions
        lower_bound, upper_bound: Confidence interval bounds (optional)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    """
    # Set the style for publication-quality plots
    plt.style.use('seaborn-whitegrid')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Main plot - Predictions vs Actual
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot training data
    ax1.scatter(X_train, y_train, color='#1f77b4', alpha=0.6, label='Training Data', s=30, edgecolor='k', linewidth=0.5)
    
    # Plot test data
    ax1.scatter(X_test, y_test, color='#ff7f0e', label='Test Data', s=50, marker='o', edgecolor='k', linewidth=0.5)
    
    # Sort test data for line plot
    sort_idx = torch.argsort(X_test.flatten())
    X_test_sorted = X_test[sort_idx]
    predictions_sorted = predictions[sort_idx]
    
    # Plot predictions
    ax1.plot(X_test_sorted, predictions_sorted, color='#d62728', label='Model Predictions', linewidth=2)
    
    # Add confidence intervals if provided
    if lower_bound is not None and upper_bound is not None:
        lower_bound_sorted = lower_bound[sort_idx]
        upper_bound_sorted = upper_bound[sort_idx]
        ax1.fill_between(X_test_sorted.flatten(), 
                         lower_bound_sorted.flatten(), 
                         upper_bound_sorted.flatten(), 
                         color='#d62728', alpha=0.2, 
                         label=f'{int(confidence_level*100)}% Confidence Interval')
    
    # Customize plot
    ax1.set_xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('TGA Value', fontsize=14, fontweight='bold')
    ax1.set_title('TGA Predictions with Confidence Intervals', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, frameon=True, facecolor='white', edgecolor='gray')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add text box with model performance metrics
    if y_test is not None:
        mse = ((predictions - y_test) ** 2).mean().item()
        rmse = np.sqrt(mse)
        r2 = 1 - ((predictions - y_test) ** 2).sum().item() / ((y_test - y_test.mean()) ** 2).sum().item()
        
        textstr = f'RMSE: {rmse:.4f}\n$R^2$: {r2:.4f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    
    # Residual plot
    ax2 = fig.add_subplot(gs[1, 0])
    residuals = (y_test - predictions).numpy().flatten()
    ax2.scatter(X_test, residuals, color='#2ca02c', alpha=0.7, s=40, edgecolor='k', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.7)
    
    # Add confidence bands to residual plot
    if lower_bound is not None and upper_bound is not None:
        std_residuals = np.std(residuals)
        ax2.axhline(y=2*std_residuals, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=-2*std_residuals, color='r', linestyle='--', alpha=0.5)
        ax2.fill_between(X_test.flatten(), -2*std_residuals, 2*std_residuals, color='gray', alpha=0.1)
    
    ax2.set_xlabel('Temperature (°C)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Residuals', fontsize=14, fontweight='bold')
    ax2.set_title('Residual Analysis', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Histogram of residuals
    ax3 = fig.add_subplot(gs[1, 1])
    sns.histplot(residuals, kde=True, color='#9467bd', ax=ax3, bins=15, edgecolor='k', linewidth=0.5)
    ax3.axvline(x=0, color='r', linestyle='-', alpha=0.7)
    ax3.set_xlabel('Residual Value', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax3.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    return fig

# Training function for the MC Dropout model
def train_mc_dropout_model(X_train, y_train, X_val, y_val, hidden_size=64, dropout_rate=0.1, 
                          learning_rate=0.001, epochs=1000, batch_size=32, patience=50):
    """
    Train a model with Monte Carlo Dropout for uncertainty estimation
    """
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = MCDropoutModel(input_size, hidden_size, output_size, dropout_rate)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stop_counter += 1
            
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# Example usage of the new functions
# ----------------------------------

# Assuming you have your data prepared as X_train, y_train, X_test, y_test
# And you have already scaled your data with scaler_X and scaler_y

# Train the MC Dropout model
mc_model, train_losses, val_losses = train_mc_dropout_model(
    X_train, y_train, X_val, y_val, 
    hidden_size=128, 
    dropout_rate=0.1,
    learning_rate=0.001,
    epochs=1000,
    patience=50
)

# Make predictions with confidence intervals
mean_pred, lower_bound, upper_bound = predict_with_confidence(
    mc_model, X_test, num_samples=100, confidence_level=0.95
)

# Plot the results
fig = plot_tga_predictions(
    X_train, y_train, X_test, y_test, 
    mean_pred, lower_bound, upper_bound, 
    confidence_level=0.95
)

# Save the figure for publication
fig.savefig('tga_predictions_with_confidence.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a user-friendly prediction function
def predict_tga(temperature, confidence_level=0.95):
    """
    User-friendly function to predict TGA at given temperature(s)
    
    Args:
        temperature: Single temperature value or list of temperatures
        confidence_level: Confidence level for prediction intervals (0-1)
        
    Returns:
        DataFrame with predictions and confidence intervals
    """
    # Ensure model is loaded
    if 'mc_model' not in globals():
        raise ValueError("Model not loaded. Please run the training code first.")
    
    # Convert input to numpy array
    if isinstance(temperature, (int, float)):
        temps = np.array([temperature])
    else:
        temps = np.array(temperature)
    
    # Make predictions for each temperature
    results = []
    for temp in temps:
        pred = predict_tga_at_temperature(
            mc_model, scaler_X, scaler_y, temp, 
            num_samples=100, confidence_level=confidence_level
        )
        results.append({
            'Temperature': temp,
            'TGA_Prediction': pred['tga_prediction'][0],
            'Lower_Bound': pred['lower_bound'][0],
            'Upper_Bound': pred['upper_bound'][0],
            'Confidence_Level': confidence_level
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate confidence interval width
    results_df['CI_Width'] = results_df['Upper_Bound'] - results_df['Lower_Bound']
    
    return results_df

# Example usage:
# predictions_df = predict_tga([200, 250, 300, 350, 400])
# print(predictions_df)

# Function to create an interactive prediction tool
def interactive_tga_prediction():
    """
    Interactive function to predict TGA values for user-provided temperatures
    """
    print("TGA Prediction Tool")
    print("-------------------")
    
    while True:
        try:
            # Get user input
            input_str = input("\nEnter temperature(s) separated by commas (or 'q' to quit): ")
            
            if input_str.lower() == 'q':
                break
                
            # Parse input
            temperatures = [float(t.strip()) for t in input_str.split(',')]
            
            # Get confidence level
            conf_level = input("Enter confidence level (0-1, default=0.95): ")
            conf_level = 0.95 if conf_level == '' else float(conf_level)
            
            # Make predictions
            results = predict_tga(temperatures, conf_level)
            
            # Display results
            print("\nPrediction Results:")
            print("------------------")
            for _, row in results.iterrows():
                print(f"Temperature: {row['Temperature']:.1f}°C")
                print(f"TGA Prediction: {row['TGA_Prediction']:.4f}")
                print(f"{int(conf_level*100)}% Confidence Interval: [{row['Lower_Bound']:.4f}, {row['Upper_Bound']:.4f}]")
                print(f"Interval Width: {row['CI_Width']:.4f}")
                print("------------------")
                
        except ValueError as e:
            print(f"Error: {e}. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}")

# Uncomment to use the interactive tool
# interactive_tga_prediction()