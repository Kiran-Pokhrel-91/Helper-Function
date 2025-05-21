import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.utils.validation import check_is_fitted

class AdvancedImputer:
    """
    A collection of advanced imputation techniques for handling missing values.
    
    Methods:
    - knn_impute: K-Nearest Neighbors imputation
    - regression_impute: Regression-based imputation
    - random_forest_impute: Random Forest-based imputation
    - mice_impute: Multiple Imputation by Chained Equations (MICE)
    """
    
    @staticmethod
    def knn_impute(df, target_cols, numeric_features, n_neighbors=5, scale=True):
        """
        K-Nearest Neighbors imputation for specified columns.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        target_cols (list): Columns to impute
        numeric_features (list): Numeric features to consider for distance calculation
        n_neighbors (int): Number of neighbors to use
        scale (bool): Whether to scale features before imputation
        
        Returns:
        pd.DataFrame: DataFrame with imputed values
        """
        df = df.copy()
        subset = df[numeric_features + target_cols]
        
        if scale:
            scaler = StandardScaler()
            subset_scaled = pd.DataFrame(scaler.fit_transform(subset[numeric_features]), 
                               columns=numeric_features)
        else:
            subset_scaled = subset[numeric_features].copy()
            
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_values = imputer.fit_transform(subset_scaled.join(subset[target_cols]))
        
        df[target_cols] = imputed_values[:, -len(target_cols):]
        return df

    @staticmethod
    def regression_impute(df, target_col, predictor_cols, 
                         categorical_cols=None, model_type='linear'):
        """
        Regression-based imputation using either Linear Regression or Random Forest.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Column to impute
        predictor_cols (list): Features to use for prediction
        categorical_cols (list): Categorical columns to encode
        model_type (str): 'linear' or 'random_forest'
        
        Returns:
        pd.DataFrame: DataFrame with imputed values
        """
        df = df.copy()
        temp_df = df[predictor_cols + [target_col]].copy()
        
        # Encode categorical features
        if categorical_cols:
            encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                temp_df[col] = le.fit_transform(temp_df[col].astype(str))
                encoders[col] = le
                
        # Split data
        known = temp_df[temp_df[target_col].notna()]
        missing = temp_df[temp_df[target_col].isna()]
        
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor()
            
        model.fit(known[predictor_cols], known[target_col])
        preds = model.predict(missing[predictor_cols])
        
        df.loc[missing.index, target_col] = preds
        return df

    @staticmethod
    def random_forest_impute(df, target_col, 
                            categorical_cols=None, 
                            estimator_type='regressor'):
        """
        Random Forest-based imputation for mixed data types.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Column to impute
        categorical_cols (list): Categorical columns to encode
        estimator_type (str): 'regressor' or 'classifier'
        
        Returns:
        pd.DataFrame: DataFrame with imputed values
        """
        df = df.copy()
        encoders = {}
        
        # Encode categorical columns
        if categorical_cols:
            for col in categorical_cols + [target_col] if df[target_col].dtype == 'object' else categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
                
        # Split data
        known = df[df[target_col].notna()]
        missing = df[df[target_col].isna()]
        
        # Train model
        if estimator_type == 'regressor':
            model = RandomForestRegressor()
        else:
            model = RandomForestClassifier()
            
        model.fit(known.drop(target_col, axis=1), known[target_col])
        preds = model.predict(missing.drop(target_col, axis=1))
        
        # Decode if categorical
        df.loc[missing.index, target_col] = preds
        if target_col in encoders:
            df[target_col] = encoders[target_col].inverse_transform(df[target_col].astype(int))
            
        return df

    @staticmethod
    def mice_impute(df, numeric_cols, categorical_cols=None, max_iter=10):
        """
        Multiple Imputation by Chained Equations (MICE) implementation.
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        numeric_cols (list): Numeric columns to impute
        categorical_cols (list): Categorical columns to impute
        max_iter (int): Number of iterations
        
        Returns:
        pd.DataFrame: DataFrame with imputed values
        """
        df = df.copy()
        impute_cols = numeric_cols + (categorical_cols if categorical_cols else [])
        
        # Create initial imputation
        imputer = IterativeImputer(max_iter=max_iter, random_state=0)
        df_imputed = pd.DataFrame(imputer.fit_transform(df[impute_cols]), 
                                columns=impute_cols)
        
        # Handle categorical columns
        if categorical_cols:
            for col in categorical_cols:
                df[col] = df_imputed[col].round().astype(int)
                
        # Replace numeric columns
        df[numeric_cols] = df_imputed[numeric_cols]
        return df