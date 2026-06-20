import warnings
from typing import Callable, List, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
import pandas as pd
from .utils import call_with_optional_params

class EvoTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for Evolutionary Feature Engineering Transformers.
    
    This class is the Python equivalent of the `evo_transformer` S3 object
    in the R evoFE package. It is designed to be fully compatible with
    scikit-learn Pipelines while providing the flexibility needed for
    genetic programming.
    """
    
    def __init__(
        self,
        name: str,
        type_: str,
        apply_func: Callable,
        name_generator: Callable,
        fit_func: Optional[Callable] = None,
        input_type: str = "numeric",
        output_type: str = "numeric",
        allow_replace: bool = False,
    ):
        """
        Initialize the transformer.
        
        Args:
            name: Transformer name (e.g., 'add', 'target_encode')
            type_: Type of transformer ('unary', 'binary', 'supervised_unary')
            apply_func: Function to apply the transformation.
                Signature: apply_func(data, input_cols, state=None) -> Series or Array
            name_generator: Function generating the output column name.
                Signature: name_generator(input_cols) -> str
            fit_func: Function to calculate state (e.g., means for target encoding).
                Signature: fit_func(data, input_cols, target_col=None) -> Any
            input_type: Type of expected input ('numeric' or 'categorical')
            output_type: Type of output ('numeric' or 'categorical')
            allow_replace: Whether column sampling allows replacement.
        """
        self.name = name
        self.type_ = type_
        self.input_type = input_type
        self.output_type = output_type
        self.apply_func = apply_func
        self.fit_func = fit_func
        self.name_generator = name_generator
        self.allow_replace = allow_replace
        
        # State learned during fit
        self.state_: Any = None
        self.input_cols_: List[str] = []
        self.target_col_: Optional[str] = None
        
    def fit(self, X, y=None, input_cols: Optional[List[str]] = None, target_col: Optional[str] = None, params=None):
        """
        Fit the transformer on the training data.
        """
        if params is not None:
            self.params = params
        if params is None:
            params = getattr(self, "params", None)
            
        if input_cols is None:
            # If not provided, assume all columns in X (or a subset based on type)
            if hasattr(X, 'columns'):
                self.input_cols_ = list(X.columns)
            else:
                warnings.warn("No input_cols provided and X has no columns. State may not be saved correctly.")
        else:
            self.input_cols_ = input_cols
            
        self.target_col_ = target_col
        
        if self.fit_func is not None:
            self.state_ = call_with_optional_params(self.fit_func, X, self.input_cols_, self.target_col_, params=params)
            
        return self

    def transform(self, X, params=None):
        """
        Apply the learned transformation to the data.
        """
        if params is None:
            params = getattr(self, "params", None)
            
        if not hasattr(self, "input_cols_") or not self.input_cols_:
            if hasattr(X, 'columns'):
                self.input_cols_ = list(X.columns)
                
        # Generate the new feature using apply_func
        new_feature = call_with_optional_params(self.apply_func, X, self.input_cols_, self.state_, params=params)
        
        # Determine the name of the new column
        new_col_name = call_with_optional_params(self.name_generator, self.input_cols_, params=params)
        
        # If the output is a Polars expression, evaluate it against the DataFrame
        if isinstance(new_feature, pl.Expr):
            if not isinstance(X, pl.DataFrame):
                # Convert pandas/dict to polars to apply the expression
                X_pl = pl.DataFrame(X)
                result_pl = X_pl.select(new_feature.alias(new_col_name))
                if isinstance(X, pd.DataFrame):
                    return result_pl.to_pandas()
                return result_pl
            return X.select(new_feature.alias(new_col_name))
            
        # Standard fallback for Series/Arrays
        if isinstance(X, pl.DataFrame):
            return pl.DataFrame({new_col_name: new_feature})
        elif isinstance(X, pd.DataFrame):
            return pd.DataFrame({new_col_name: new_feature})
        else:
            return new_feature
