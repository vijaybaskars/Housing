from utils import (
    load_data, explore_data, visualize_data, prepare_data,
    train_linear_regression, train_random_forest, train_ridge, train_lasso,
    evaluate_model, save_model, plot_results, generate_performance_report
)
import os

def main():
    
    # Step 1: Load and explore data
    print("1. Loading and exploring data...")
    df = load_data()
    df = explore_data(df)
    
    # Step 2: Visualize data
    print("\n2. Creating data visualizations...")
    visualize_data(df)
    
    # Step 3: Prepare data
    print("\n3. Preparing data for modeling...")
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Step 4: Train models
    print("\n4. Training regression models...")
    
    models = {}
    results = {}
    
    # Linear Regression
    print("Training Linear Regression...")
    models['Linear Regression'] = train_linear_regression(X_train, y_train)
    results['Linear Regression'] = evaluate_model(
        models['Linear Regression'], X_test, y_test, 'Linear Regression'
    )
    
    # Random Forest
    print("Training Random Forest...")
    models['Random Forest'] = train_random_forest(X_train, y_train)
    results['Random Forest'] = evaluate_model(
        models['Random Forest'], X_test, y_test, 'Random Forest'
    )
    
    # Ridge Regression
    print("Training Ridge Regression...")
    models['Ridge'] = train_ridge(X_train, y_train)
    results['Ridge'] = evaluate_model(
        models['Ridge'], X_test, y_test, 'Ridge'
    )
    
    # Lasso Regression
    print("Training Lasso Regression...")
    models['Lasso'] = train_lasso(X_train, y_train)
    results['Lasso'] = evaluate_model(
        models['Lasso'], X_test, y_test, 'Lasso'
    )
    
    # Step 5: Save models
    print("\n5. Saving models...")
    os.makedirs('models', exist_ok=True)
    for model_name, model in models.items():
        save_model(model, f'models/{model_name.lower().replace(" ", "_")}_model.joblib')
    
    # Step 6: Plot results
    print("\n6. Generating visualizations...")
    plot_results(y_test, results)
    
    # Step 7: Generate report
    print("\n7. Generating performance report...")
    generate_performance_report(results)
    
    print("\n=== Pipeline completed successfully! ===")

if __name__ == "__main__":
    main()
