import polars as pl
df = pl.DataFrame({"A": [1, 2]})

try:
    # Test pl.sum_horizontal with duplicate strings
    expr = pl.sum_horizontal(["A", "A"])
    df = df.with_columns(expr.alias("add_123"))
    print("sum_horizontal worked")
except Exception as e:
    print(f"sum_horizontal duplicate string exception: {e}")

try:
    # Test unary with duplicate strings
    expr = pl.col(["A", "A"]).sign() * (pl.col(["A", "A"]).abs() ** 2)
    df = df.with_columns(expr.alias("pow_123"))
    print("pow worked")
except Exception as e:
    print(f"pow duplicate string exception: {e}")
