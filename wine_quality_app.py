# ============================================================================
# WINE QUALITY PREDICTION - PRODUCTION DEPLOYMENT
# ============================================================================

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import argparse

# ML imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WineQualityPredictor:
    """Production-ready wine quality prediction model"""
    
    def __init__(self, model_path: str = None):
        """Initialize the predictor"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_and_prepare_data(self, filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare wine quality data"""
        logger.info(f"Loading data from {filepath}")
        
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Loaded {len(data)} samples with {data.shape[1]} features")
            
            # Remove Id column if exists
            if 'Id' in data.columns:
                data = data.drop('Id', axis=1)
            
            # Separate features and target
            X = data.drop('quality', axis=1)
            y = data['quality']
            
            self.feature_names = X.columns.tolist()
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              optimize: bool = False) -> Dict:
        """Train the model with optional hyperparameter optimization"""
        logger.info("Starting model training...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if optimize:
            logger.info("Performing hyperparameter optimization...")
            self.model = self._hyperparameter_tuning(X_train_scaled, y_train)
        else:
            logger.info("Training with default parameters...")
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Get training metrics
        y_pred = self.model.predict(X_train_scaled)
        metrics = self._calculate_metrics(y_train, y_pred)
        
        logger.info(f"Training completed. RMSE: {metrics['rmse']:.4f}")
        return metrics
    
    def _hyperparameter_tuning(self, X_train: np.ndarray, y_train: pd.Series) -> RandomForestRegressor:
        """Perform hyperparameter tuning"""
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, 50, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=30,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {random_search.best_params_}")
        
        return random_search.best_estimator_
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Calculate accuracy within tolerance
        metrics['accuracy_0_5'] = np.sum(np.abs(y_test - y_pred) <= 0.5) / len(y_test)
        metrics['accuracy_1_0'] = np.sum(np.abs(y_test - y_pred) <= 1.0) / len(y_test)
        
        # Store predictions for analysis
        metrics['predictions'] = y_pred.tolist()
        metrics['actual'] = y_test.tolist()
        
        return metrics
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Load or train a model first.")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_single(self, features: Dict) -> Dict:
        """Predict quality for a single wine sample"""
        try:
            # Convert to DataFrame
            X = pd.DataFrame([features])
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Reorder columns
            X = X[self.feature_names]
            
            # Predict
            prediction = self.predict(X)[0]
            
            return {
                'quality': float(prediction),
                'confidence': self._calculate_confidence(prediction),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _calculate_confidence(self, prediction: float) -> float:
        """Calculate confidence score (simplified)"""
        # This could be more sophisticated (e.g., based on prediction variance)
        # For now, using a simple sigmoid-like function
        return float(1.0 / (1.0 + np.exp(-0.5 * (prediction - 5.5))))
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        importance = self.model.feature_importances_
        return {
            'features': self.feature_names,
            'importance': importance.tolist(),
            'importance_dict': dict(zip(self.feature_names, importance))
        }
    
    def save_model(self, filepath: str):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model and scaler"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Wine Quality Prediction System')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data', type=str, default='WineQT.csv',
                             help='Path to training data')
    train_parser.add_argument('--output', type=str, default='wine_model.pkl',
                             help='Path to save trained model')
    train_parser.add_argument('--optimize', action='store_true',
                             help='Enable hyperparameter optimization')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                             help='Test set size (default: 0.2)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', type=str, default='wine_model.pkl',
                               help='Path to trained model')
    predict_parser.add_argument('--input', type=str, required=True,
                               help='Path to input data (CSV or JSON)')
    predict_parser.add_argument('--output', type=str,
                               help='Path to save predictions (optional)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', type=str, default='wine_model.pkl',
                            help='Path to trained model')
    eval_parser.add_argument('--data', type=str, default='WineQT.csv',
                            help='Path to evaluation data')
    
    # Feature importance command
    feat_parser = subparsers.add_parser('features', help='Show feature importance')
    feat_parser.add_argument('--model', type=str, default='wine_model.pkl',
                            help='Path to trained model')
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'features':
        show_features(args)
    else:
        parser.print_help()

def train_model(args):
    """Train a new model"""
    logger.info("Starting training pipeline...")
    
    predictor = WineQualityPredictor()
    
    # Load data
    X, y = predictor.load_and_prepare_data(args.data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train model
    train_metrics = predictor.train(X_train, y_train, optimize=args.optimize)
    
    # Evaluate on test set
    test_metrics = predictor.evaluate(X_test, y_test)
    
    # Save model
    predictor.save_model(args.output)
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"\nTraining Metrics:")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    print(f"  Accuracy (±0.5): {test_metrics['accuracy_0_5']:.2%}")
    print(f"  Accuracy (±1.0): {test_metrics['accuracy_1_0']:.2%}")
    
    # Save metrics to file
    metrics_data = {
        'training': train_metrics,
        'test': test_metrics,
        'model_info': {
            'file': args.output,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    metrics_file = args.output.replace('.pkl', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_file}")
    logger.info("Training pipeline completed successfully")

def predict(args):
    """Make predictions using trained model"""
    logger.info("Starting prediction pipeline...")
    
    predictor = WineQualityPredictor(args.model)
    
    # Load input data
    if args.input.endswith('.csv'):
        data = pd.read_csv(args.input)
    elif args.input.endswith('.json'):
        with open(args.input, 'r') as f:
            data_dict = json.load(f)
        data = pd.DataFrame(data_dict)
    else:
        raise ValueError("Input file must be CSV or JSON")
    
    # Make predictions
    predictions = predictor.predict(data)
    
    # Add predictions to data
    result = data.copy()
    result['predicted_quality'] = predictions
    result['prediction_confidence'] = [predictor._calculate_confidence(p) for p in predictions]
    
    # Print sample predictions
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nFirst 5 predictions:")
    print(result[['predicted_quality', 'prediction_confidence']].head().to_string())
    
    # Save results if output specified
    if args.output:
        if args.output.endswith('.csv'):
            result.to_csv(args.output, index=False)
        elif args.output.endswith('.json'):
            result.to_json(args.output, orient='records', indent=2)
        
        logger.info(f"Predictions saved to {args.output}")
    
    return result

def evaluate_model(args):
    """Evaluate model performance"""
    logger.info("Starting evaluation pipeline...")
    
    predictor = WineQualityPredictor(args.model)
    
    # Load evaluation data
    X, y = predictor.load_and_prepare_data(args.data)
    
    # Evaluate
    metrics = predictor.evaluate(X, y)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nModel: {args.model}")
    print(f"Samples evaluated: {len(y)}")
    print(f"\nPerformance Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  Accuracy (±0.5): {metrics['accuracy_0_5']:.2%}")
    print(f"  Accuracy (±1.0): {metrics['accuracy_1_0']:.2%}")

def show_features(args):
    """Show feature importance"""
    predictor = WineQualityPredictor(args.model)
    importance = predictor.get_feature_importance()
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    # Sort by importance
    features = importance['features']
    scores = importance['importance']
    sorted_features = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 Most Important Features:")
    print("-"*40)
    for i, (feature, score) in enumerate(sorted_features[:10], 1):
        bar = '█' * int(score * 50)  # Scale for display
        print(f"{i:2}. {feature:25s} {score:.4f} {bar}")

if __name__ == "__main__":
    main()