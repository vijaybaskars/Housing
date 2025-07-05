"""
Boston Housing Price Prediction - Hyperparameter Optimization
This script optimizes hyperparameters for regression models and compares performance.
"""

from utils import (
    load_data, prepare_data, 
    optimize_linear_regression, optimize_random_forest, 
    optimize_ridge, optimize_lasso,
    evaluate_model, save_model, plot_results, 
    generate_hyperparameter_report
)
import os
import json

def main():
    """Main function for hyperparameter optimization pipeline"""
    print("=== Boston Housing - Hyperparameter Optimization Pipeline ===\n")
    
    # Step 1: Load and prepare data
    print("1. Loading and preparing data...")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Step 2: Load basic model results for comparison
    basic_results = {}
    if os.path.exists('basic_results.json'):
        with open('basic_results.json', 'r') as f:
            basic_results = json.load(f)
    
    # Step 3: Hyperparameter optimization
    print("\n2. Starting hyperparameter optimization...")
    
    optimized_models = {}
    optimized_results = {}
    
    # Linear Regression variants (Ridge, Lasso)
    print("\nOptimizing Linear Regression variants...")
    linear_models = optimize_linear_regression(X_train, y_train)
    
    for name, model in linear_models.items():
        optimized_models[f'Optimized {name}'] = model
        optimized_results[f'Optimized {name}'] = evaluate_model(
            model, X_test, y_test, f'Optimized {name}'
        )
    
    # Random Forest
    print("\nOptimizing Random Forest...")
    optimized_models['Optimized Random Forest'] = optimize_random_forest(X_train, y_train)
    optimized_results['Optimized Random Forest'] = evaluate_model(
        optimized_models['Optimized Random Forest'], X_test, y_test, 'Optimized Random Forest'
    )
    
    # Ridge Regression
    print("\nOptimizing Ridge Regression...")
    optimized_models['Optimized Ridge'] = optimize_ridge(X_train, y_train)
    optimized_results['Optimized Ridge'] = evaluate_model(
        optimized_models['Optimized Ridge'], X_test, y_test, 'Optimized Ridge'
    )
    
    # Lasso Regression
    print("\nOptimizing Lasso Regression...")
    optimized_models['Optimized Lasso'] = optimize_lasso(X_train, y_train)
    optimized_results['Optimized Lasso'] = evaluate_model(
        optimized_models['Optimized Lasso'], X_test, y_test, 'Optimized Lasso'
    )
    
    # Step 4: Save optimized models
    print("\n3. Saving optimized models...")
    os.makedirs('optimized_models', exist_ok=True)
    for model_name, model in optimized_models.items():
        safe_name = model_name.lower().replace(" ", "_")
        save_model(model, f'optimized_models/{safe_name}_model.joblib')
    
    # Step 5: Save results for comparison
    optimized_results_clean = {k: {'mse': v['mse'], 'r2': v['r2']} for k, v in optimized_results.items()}
    with open('optimized_results.json', 'w') as f:
        json.dump(optimized_results_clean, f, indent=2)
    
    # Step 6: Generate comprehensive comparison
    print("\n4. Generating comprehensive performance report...")
    if basic_results:
        generate_hyperparameter_report(basic_results, optimized_results_clean)
    
    # Step 7: Create comparison plots
    print("\n5. Creating comparison visualizations...")
    plot_results(y_test, optimized_results)
    
    print("\n=== Hyperparameter optimization completed successfully! ===")

if __name__ == "__main__":
    main()