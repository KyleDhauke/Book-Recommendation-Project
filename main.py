import pandas as pd
import numpy as np
from generate_user import *
from book_recommender_1 import *
from book_recommender_2 import *
from knn_recommender import *
from evaluate import *
import typer
from typing import Optional

from typing import Optional
import typer

app = typer.Typer()

# Your function definitions here

@app.command()
def recommend_books(
    algorithm: str = typer.Option(None, prompt=True, help="Please choose a recommender algorithm:"),
    book_title: str = typer.Option(..., prompt=True, help="Please enter a book title:"),
    n_recommendations: int = typer.Option(None, prompt="Number of recommendations to return:", show_default=True),
    n_comp: Optional[int] = typer.Option(None, prompt="Number of eigenvalues to keep:", show_default=True),
):
    if algorithm == "mf_1":
        if n_comp is None or "":
            n_comp = 12
        recommended_books = matrix_factorisation_1(book_title, n_recommendations=n_recommendations, n_comp=n_comp)
    elif algorithm == "mf_2":
        if n_comp is None or "":
            n_comp = 12
        recommended_books = matrix_factorisation_2(book_title, n_recommendations=n_recommendations, n_comp=n_comp)
    elif algorithm == "knn":
        recommended_books = knn_popularity_recommender(book_title, n_recommendations=n_recommendations)
    else:
        typer.echo("Invalid algorithm! Please select a valid recommender algorithm from the options below: ")
        
        recommender_options = ["mf_1", "mf_2", "knn"]
        for option in recommender_options:
            typer.echo(option)
        
        return
    
    typer.echo(f"Recommended books based on {algorithm}:")
    for book in recommended_books:
        typer.echo(book)

@app.command()
def evaluate_recommender(
    algorithm: str = typer.Option(..., prompt=True, help="Choose a recommender algorithm"),
    n_tests: int = typer.Option(..., prompt=True, help="Number of tests to run:"),
    n_comp: Optional[int] = typer.Option(None, prompt="Number of eigenvalues to keep:", show_default=False),
):
    if algorithm == "mf_1":
        recommender = 'b1_test'
        if n_comp is None:
            n_comp = 12
    elif algorithm == "knn":
        recommender = 'knn_test'
        n_comp = None

    elif algorithm == "mf_2":
        recommender = 'b2_test'
        if n_comp is None:
            n_comp = 12
    
    accuracy, precision, recall, f1_score = performance_metrics(n_tests, recommender, n_comp=n_comp)
    
    typer.echo(f"Performance metrics for {algorithm}:")
    typer.echo(f"Accuracy: {accuracy}")
    typer.echo(f"Precision: {precision}")
    typer.echo(f"Recall: {recall}")
    typer.echo(f"F1 Score: {f1_score}")

if __name__ == "__main__":
    app()

