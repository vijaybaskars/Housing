import pytest
from utils import optimize_ridge, optimize_lasso, optimize_random_forest, load_data, prepare_data

def test_optimize_ridge():
    """Test Ridge hyperparameter optimization"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    model = optimize_ridge(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'alpha')
    assert hasattr(model, 'predict')

def test_optimize_lasso():
    """Test Lasso hyperparameter optimization"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    model = optimize_lasso(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'alpha')
    assert hasattr(model, 'predict')

def test_optimize_random_forest():
    """Test Random Forest hyperparameter optimization"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    model = optimize_random_forest(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'n_estimators')
    assert hasattr(model, 'predict')