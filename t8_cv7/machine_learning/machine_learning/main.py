from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data_handling import Dataset
from experiment import Experiment
from result_plots import Plotter

def main():
    """
    Main function to execute the model training and evaluation pipeline.

    Initializes the dataset, defines models and their parameter grids,
    and invokes the replication of model training and evaluation.
    """
    # Initialize dataset and preprocess the data
    dataset = Dataset()

    # Define models to be trained
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear'),
        "Random Forest": RandomForestClassifier(),  # Solver specified for clarity
    }

    # Define hyperparameter grids for tuning
    param_grids = {
        "Logistic Regression": {"C": [0.1, 1, 10], "max_iter": [10000]},
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    }

    experiment = Experiment(models, param_grids, n_replications=10)
    results = experiment.run(dataset.data, dataset.target)

    # Plot the results using the Plotter class
    plotter = Plotter()

    plotter.plot_metric_density(results)

    plotter.plot_evaluation_metric_over_replications(
        experiment.results.groupby('model')['accuracy'].apply(list).to_dict(),
        'Accuracy per Replication and Average Accuracy', 'Accuracy'
    )

    plotter.plot_confusion_matrices(experiment.mean_conf_matrices)

    plotter.plot_precision_over_replications(
        experiment.results.groupby('model')['precision'].apply(list).to_dict()
    )

    plotter.print_best_parameters(results)


if __name__ == "__main__":
    main()

