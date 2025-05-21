import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Enables IterativeImputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression

class AdvancedImputer:
    """
    A collection of advanced imputation techniques for handling missing values in a pandas DataFrame.
    """

    @staticmethod
    def knn_impute(df, target_col, predictor_cols, n_neighbors=5, scale=True):
        """
        Imputes missing values in `target_col` using the K-Nearest Neighbors (KNN) algorithm.

        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame containing missing values.
        target_col : str or list of str
            The column(s) to impute.
        predictor_cols : list of str
            Columns to use as predictors for imputation.
        n_neighbors : int, default=5
            Number of neighbors to consider for KNN.
        scale : bool, default=True
            Whether to scale the predictor and target columns before imputation.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with the imputed `target_col`.
        """
        df = df.copy()
        if isinstance(target_col, str):
            target_col = [target_col]

        subset = df[predictor_cols + target_col]

        if scale:
            scaler = StandardScaler()
            subset_scaled = pd.DataFrame(scaler.fit_transform(subset), columns=subset.columns)
        else:
            subset_scaled = subset.copy()

        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(subset_scaled)
        imputed_df = pd.DataFrame(imputed_data, columns=subset.columns)

        df[target_col] = imputed_df[target_col]
        return df

    @staticmethod
    def regression_impute(df, target_col, predictor_cols, categorical_cols=None, model_type='linear'):
        """
        Imputes missing values in `target_col` using a regression model.

        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame containing missing values.
        target_col : str
            The target column to impute.
        predictor_cols : list of str
            Predictor features used to train the regression model.
        categorical_cols : list of str, optional
            List of categorical columns to encode.
        model_type : str, default='linear'
            The type of regression model to use: 'linear' or 'random_forest'.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with the imputed `target_col`.
        """
        df = df.copy()
        temp_df = df[predictor_cols + [target_col]].copy()

        if categorical_cols:
            for col in categorical_cols:
                le = LabelEncoder()
                temp_df[col] = le.fit_transform(temp_df[col].astype(str))

        temp_df = temp_df.dropna(subset=predictor_cols)

        known = temp_df[temp_df[target_col].notna()]
        missing = temp_df[temp_df[target_col].isna()]

        if missing.empty:
            print(f"No missing values found in '{target_col}'.")
            return df

        model = LinearRegression() if model_type == 'linear' else RandomForestRegressor()
        model.fit(known[predictor_cols], known[target_col])
        predictions = model.predict(missing[predictor_cols])

        df.loc[missing.index, target_col] = predictions
        return df

    @staticmethod
    def random_forest_impute(df, target_col, predictor_cols, categorical_cols=None, model_type='regressor'):
        """
        Imputes missing values in `target_col` using a Random Forest model.

        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame containing missing values.
        target_col : str
            The target column to impute.
        predictor_cols : list of str
            Columns used as predictors for the Random Forest model.
        categorical_cols : list of str, optional
            List of categorical columns to encode before training.
        model_type : str, default='regressor'
            Use 'regressor' for continuous targets or 'classifier' for categorical targets.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with the imputed `target_col`.
        """
        df = df.copy()
        encoders = {}

        if categorical_cols:
            for col in categorical_cols + ([target_col] if df[target_col].dtype == 'object' else []):
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

        df = df.dropna(subset=predictor_cols)

        known = df[df[target_col].notna()]
        missing = df[df[target_col].isna()]

        if missing.empty:
            print(f"No missing values found in '{target_col}'.")
            return df

        model = RandomForestRegressor() if model_type == 'regressor' else RandomForestClassifier()
        model.fit(known[predictor_cols], known[target_col])
        predictions = model.predict(missing[predictor_cols])

        df.loc[missing.index, target_col] = predictions

        if target_col in encoders:
            df[target_col] = encoders[target_col].inverse_transform(df[target_col].astype(int))

        return df

    @staticmethod
    def mice_impute(df, target_col=None, predictor_cols=None, categorical_cols=None,
                    model_type='mice', max_iter=10, scale=False):
        """
        Imputes missing values using MICE (Multiple Imputation by Chained Equations) via IterativeImputer.

        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame containing missing values.
        target_col : str, optional
            Ignored (included for consistency).
        predictor_cols : list of str
            Numeric columns to impute.
        categorical_cols : list of str, optional
            Categorical columns to impute, rounded after imputation.
        model_type : str, default='mice'
            Kept for interface consistency.
        max_iter : int, default=10
            Maximum number of imputation iterations.
        scale : bool, default=False
            Whether to scale data before imputation.

        Returns:
        -------
        pd.DataFrame
            A DataFrame with imputed `predictor_cols` and optionally `categorical_cols`.
        """
        df = df.copy()

        if predictor_cols is None:
            raise ValueError("predictor_cols must be specified.")

        impute_cols = predictor_cols + (categorical_cols if categorical_cols else [])

        if scale:
            scaler = StandardScaler()
            df[impute_cols] = scaler.fit_transform(df[impute_cols])

        imputer = IterativeImputer(max_iter=max_iter, random_state=0)
        imputed_df = pd.DataFrame(imputer.fit_transform(df[impute_cols]), columns=impute_cols)

        if categorical_cols:
            for col in categorical_cols:
                df[col] = imputed_df[col].round().astype(int)

        df[predictor_cols] = imputed_df[predictor_cols]
        return df
