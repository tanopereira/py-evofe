import polars as pl
from evofe.builtin import evo_transformers
import pytest

def test_log_transformer():
    df = pl.DataFrame({"x": [0, 1, -2, 3]})
    transformer = evo_transformers["log"]
    
    # Fit and transform
    transformer.fit(df, input_cols=["x"])
    result = transformer.transform(df)
    
    # It should return a dataframe with 1 column
    assert isinstance(result, pl.DataFrame)
    assert len(result.columns) == 1
    
    # Check values (log1p of abs)
    col_name = result.columns[0]
    assert col_name.startswith("log_")
    
    expected = [0.0, 0.693147, 1.098612, 1.386294]
    actual = result[col_name].to_list()
    for e, a in zip(expected, actual):
        assert abs(e - a) < 1e-5

def test_add_transformer():
    df = pl.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9]
    })
    
    transformer = evo_transformers["add"]
    transformer.fit(df, input_cols=["a", "b", "c"])
    result = transformer.transform(df)
    
    col_name = result.columns[0]
    assert col_name.startswith("add_")
    assert result[col_name].to_list() == [12, 15, 18]

def test_divide_transformer_zero_safety():
    df = pl.DataFrame({
        "num": [10, 20, 30],
        "den": [2, 0, 5]
    })
    
    transformer = evo_transformers["divide"]
    transformer.fit(df, input_cols=["num", "den"])
    result = transformer.transform(df)
    
    col_name = result.columns[0]
    # Division by zero should safely fallback to 0
    assert result[col_name].to_list() == [5.0, 0.0, 6.0]
