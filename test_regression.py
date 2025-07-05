import pytest
import os
import pandas as pd
from pathlib import Path
from utils import load_data, prepare_data, train_ridge, train_lasso

def test_data_loading():
    """Test if data loads correctly"""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert 'MEDV' in df.columns
    assert df.shape[0] > 0
    assert df.shape[1] == 14  # 13 features + 1 target

def test_data_preparation():
    """Test data preparation function"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]

def test_ridge_model_training():
    """Test Ridge regression model training"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    model = train_ridge(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'predict')
    
    # Test prediction
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(isinstance(pred, (int, float)) for pred in predictions)

def test_lasso_model_training():
    """Test Lasso regression model training"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    model = train_lasso(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'predict')
    
    # Test prediction
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
    assert all(isinstance(pred, (int, float)) for pred in predictions)

def test_models_saved():
    """Test if models are saved after training"""
    expected_models = [
        'models/linear_regression_model.joblib',
        'models/random_forest_model.joblib',
        'models/ridge_model.joblib',
        'models/lasso_model.joblib'
    ]
    
    for model_path in expected_models:
        if os.path.exists(model_path):
            assert Path(model_path).exists()

def test_performance_report_exists():
    """Test if performance report is generated"""
    if os.path.exists('performance_report.md'):
        report_path = Path('performance_report.md')
        assert report_path.exists()
        assert report_path.stat().st_size > 0

def test_ridge_regularization():
    """Test Ridge regression regularization parameter"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    model = train_ridge(X_train, y_train)
    # Ridge should have alpha parameter
    assert hasattr(model, 'alpha')
    assert model.alpha > 0

def test_lasso_regularization():
    """Test Lasso regression regularization parameter"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    model = train_lasso(X_train, y_train)
    # Lasso should have alpha parameter
    assert hasattr(model, 'alpha')
    assert model.alpha > 0

def test_ridge_coefficients():
    """Test Ridge regression coefficients"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    model = train_ridge(X_train, y_train)
    # Ridge should have coefficients
    assert hasattr(model, 'coef_')
    assert len(model.coef_) == X_train.shape[1]  # Should match number of features

def test_lasso_coefficients():
    """Test Lasso regression coefficients"""
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    model = train_lasso(X_train, y_train)
    # Lasso should have coefficients
    assert hasattr(model, 'coef_')
    assert len(model.coef_) == X_train.shape[1]  # Should match number of features