"""
Data Preprocessing Module for HEDPriv Framework
Handles loading, cleaning, and normalization of datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocessor for medical datasets with privacy-preserving requirements"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_heart_disease_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load the UCI Heart Disease dataset
        
        Args:
            filepath: Path to CSV file. If None, generates synthetic data
            
        Returns:
            DataFrame with heart disease data
        """
        if filepath is None:
            # Generate synthetic heart disease data for demonstration
            print("Generating synthetic heart disease data...")
            np.random.seed(self.random_state)
            n_samples = 1000
            
            data = {
                'Age': np.random.randint(29, 80, n_samples),
                'Sex': np.random.randint(0, 2, n_samples),
                'ChestPainType': np.random.randint(0, 4, n_samples),
                'RestingBP': np.random.randint(90, 200, n_samples),
                'Cholesterol': np.random.randint(120, 400, n_samples),
                'FastingBS': np.random.randint(0, 2, n_samples),
                'RestingECG': np.random.randint(0, 3, n_samples),
                'MaxHR': np.random.randint(60, 200, n_samples),
                'ExerciseAngina': np.random.randint(0, 2, n_samples),
                'Oldpeak': np.random.uniform(0, 6, n_samples),
                'ST_Slope': np.random.randint(0, 3, n_samples),
                'HeartDisease': np.random.randint(0, 2, n_samples)
            }
            
            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(filepath)
            
        print(f"Loaded dataset with shape: {df.shape}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Handle missing values
        if df_clean.isnull().sum().sum() > 0:
            print(f"Found {df_clean.isnull().sum().sum()} missing values")
            # For numerical columns, fill with median
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Remove outliers using IQR method (optional, can be disabled)
        # This is important for CKKS stability
        numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
        for col in numeric_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
        
        print(f"Cleaned dataset shape: {df_clean.shape}")
        return df_clean
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select numerical features for encryption
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with selected features
        """
        # Select key numerical features for CKKS encryption
        numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
        
        # Filter only existing columns
        available_features = [f for f in numerical_features if f in df.columns]
        
        if not available_features:
            raise ValueError("No numerical features found in dataset")
        
        self.feature_names = available_features
        df_selected = df[available_features].copy()
        
        print(f"Selected features: {available_features}")
        return df_selected
    
    def normalize(self, df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Normalize data using StandardScaler (crucial for CKKS stability)
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized numpy array
        """
        if fit:
            normalized = self.scaler.fit_transform(df)
            print(f"Fitted scaler with mean: {self.scaler.mean_}")
            print(f"Fitted scaler with std: {self.scaler.scale_}")
        else:
            normalized = self.scaler.transform(df)
        
        return normalized
    
    def preprocess(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        print("\\n=== Starting Preprocessing Pipeline ===")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Select features
        df_features = self.select_features(df_clean)
        
        # Step 3: Split data
        train_df, test_df = train_test_split(
            df_features, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        print(f"Train set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        
        # Step 4: Normalize
        train_normalized = self.normalize(train_df, fit=True)
        test_normalized = self.normalize(test_df, fit=False)
        
        print("=== Preprocessing Complete ===\\n")
        
        return train_normalized, test_normalized
    
    def get_feature_names(self):
        """Get the names of selected features"""
        return self.feature_names


# Example usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Load data
    data = preprocessor.load_heart_disease_data()
    
    # Preprocess
    X_train, X_test = preprocessor.preprocess(data)
    
    print(f"\\nTraining data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Sample statistics - Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")
