import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os

def load_data():
    import pandas as pd
    import numpy as np
    
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    
    # Split into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    # Feature names based on original dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # Target variable
    
    return df

def explore_data(df):
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df

def visualize_data(df):
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.hist(df['MEDV'], bins=30, edgecolor='black')
    plt.title('Distribution of House Prices (MEDV)')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    
    # Correlation heatmap
    plt.subplot(2, 2, 2)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    # Feature vs target scatter plots (top correlated features)
    top_features = correlation_matrix['MEDV'].abs().sort_values(ascending=False)[1:4]
    
    plt.subplot(2, 2, 3)
    plt.scatter(df[top_features.index[0]], df['MEDV'])
    plt.xlabel(top_features.index[0])
    plt.ylabel('MEDV')
    plt.title(f'{top_features.index[0]} vs Price')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df[top_features.index[1]], df['MEDV'])
    plt.xlabel(top_features.index[1])
    plt.ylabel('MEDV')
    plt.title(f'{top_features.index[1]} vs Price')
    
    plt.tight_layout()
    plt.savefig('plots/data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()

def prepare_data(df, test_size=0.2, random_state=42):
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_ridge(X_train, y_train):
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_lasso(X_train, y_train):
    model = Lasso(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {'model_name': model_name, 'mse': mse, 'r2': r2, 'predictions': y_pred}

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    return load(filepath)

def plot_results(y_test, predictions_dict):
    os.makedirs('plots', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (model_name, results) in enumerate(predictions_dict.items()):
        ax = axes[i]
        ax.scatter(y_test, results['predictions'], alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name}\nMSE: {results["mse"]:.4f}, R²: {results["r2"]:.4f}')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_performance_report(results_dict):
    report = "# Model Performance Comparison Report\n\n"
    report += "## Regression Models (Without Hyperparameter Tuning)\n\n"
    report += "| Model | MSE | R² Score |\n"
    report += "|-------|-----|----------|\n"
    
    for model_name, results in results_dict.items():
        report += f"| {model_name} | {results['mse']:.4f} | {results['r2']:.4f} |\n"
    
    # Find best model
    best_model = min(results_dict.items(), key=lambda x: x[1]['mse'])
    report += f"\n**Best Model**: {best_model[0]} (Lowest MSE: {best_model[1]['mse']:.4f})\n"
    
    with open('performance_report.md', 'w') as f:
        f.write(report)
    
    print("Performance report saved to performance_report.md")
    return report