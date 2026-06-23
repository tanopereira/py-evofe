"""
Example: Running py-evoFE on the Kaggle Higgs Boson Dataset
===========================================================

This script demonstrates how to load the Higgs Boson Machine Learning Challenge dataset
and run `EvoFE` to discover new features.

Dataset: https://www.kaggle.com/c/higgs-boson/data
Assumed location: ./training.zip
"""

import os
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from evofe import EvoFE


def run_higgs_example():
    # Place the Kaggle dataset 'training.zip' in the same directory as this script, or provide the path here
    file_path = "training.zip"
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find the dataset at {file_path}")
        print("Please ensure the Kaggle Higgs Boson dataset 'training.zip' is downloaded to that location.")
        return

    import zipfile
    
    print(f"Loading data from {file_path}...")
    
    # Read the data from the zip
    with zipfile.ZipFile(file_path, "r") as z:
        with z.open("training.csv") as f:
            df = pl.read_csv(f.read(), null_values=["-999.0"])
    
    # Preprocessing
    # 1. Drop EventId (useless for prediction)
    # 2. Drop Weight (could be used as sample_weight, but dropped for basic example)
    cols_to_drop = [c for c in ["EventId", "Weight"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)
        
    # Map 's' (signal) to 1 and 'b' (background) to 0
    df = df.with_columns(
        pl.col("Label").replace_strict({"s": 1, "b": 0}, default=None).cast(pl.Int32)
    )
    
    # The dataset has 250,000 rows. We sample a subset to keep the example run quick.
    n_samples = 250000
    print(f"Sampling {n_samples} rows for the demonstration (out of {len(df)})...")
    df = df.sample(n=n_samples, seed=42)
    
    X = df.drop("Label")
    y = df["Label"].to_numpy()
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(), y, test_size=0.2, random_state=42, stratify=y
    )
    
    # EvoFE expects a DataFrame to know the column names
    X_train_df = pl.DataFrame(X_train, schema=X.columns)
    X_test_df = pl.DataFrame(X_test, schema=X.columns)
    
    print("\nConfiguring EvoFE...")
    evo = EvoFE(
        task="classification",
        evaluator="lightgbm",
        pop_size=15,              # Small population for a quick run
        n_generations=10,         # Few generations for a quick run
        evaluation_strategy="split",
        allowed_transformers="all",
        verbose=True,
        stagnation_limit=1,
    )
    
    print("Fitting EvoFE (evolutionary feature engineering)...")
    # This will run the evolutionary algorithm and discover the best features
    evo.fit(X_train_df, y_train)
    
    recipe = evo.get_recipe()
    print("\n" + "="*50)
    print("Evolution Complete!")
    print(f"Best Fitness (exp(-log_loss)): {recipe.fitness:.4f}")
    print(f"Number of new features discovered: {len(recipe.genes)}")
    print("Top discovered features:")
    for gene in recipe.best_individual.genes[:5]: # Show top 5
        imp = recipe.best_individual.importances.get(gene.output_col, 0.0)
        print(f"  • {gene.to_formula():<40} (importance: {imp:.4f})")
    print("="*50 + "\n")
    
    print("Transforming test data and evaluating model...")
    # EvoFE automatically applies the best feature recipe and runs the estimator
    proba = evo.predict_proba(X_test_df)
    preds = proba[:, 1] > 0.5
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba[:, 1])
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test ROC AUC:  {auc:.4f}")


if __name__ == "__main__":
    run_higgs_example()
