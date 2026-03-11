import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import product

PLOT_DIR = 'plots'

def plot_training_metrics(
    df, model_type, batch_size, lr, save_dir=None, figsize=(14, 10), ignore_errors=False, verbose=False
):
    """
    Plots training metrics vs step for filtered rows of a dataframe.
    """

    # Filter dataframe to get correct training run
    filtered_df = df[
        (df["model_type"] == model_type) &
        (df["batch_size"] == batch_size) &
        (df["lr"] == lr)
    ].sort_values("step")

    if filtered_df.empty:
        if ignore_errors:
            if verbose:
                print("Ignoring error: No matching training run found for params:\n\t" + 
                    f"model_type={model_type}, batch_size={batch_size}, lr={lr}."
                )
            return
        raise ValueError("No training run matches the given filter criteria.")
    
    grouped = filtered_df.groupby('score_type')

    for score_type, group_df in grouped:
        if save_dir is not None:
            save_path = os.path.join(
                save_dir, f'{score_type}_bs{batch_size}_lr{lr}.png'
            )
            if os.path.exists(save_path):
                return
        
        if verbose:
            print(
                f"Plotting metrics for score_type={score_type}, model_type={model_type}, batch_size={batch_size}, lr={lr}..."
            )

        steps = group_df["step"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Dev loss
        axes[0, 0].plot(steps, group_df["dev_loss"], label="dev_loss")
        axes[0, 0].set_ylabel("Dev Loss")
        axes[0, 0].set_title("Dev Loss vs Step")

        # Spearman coefficient
        axes[0, 1].plot(steps, group_df["spearman"], label="spearman")
        axes[0, 1].set_ylabel("Spearman Coefficient")
        axes[0, 1].set_title("Spearman Coefficient vs Step")
        
        # Pearson coefficient
        axes[1, 0].plot(steps, group_df["pearson"], label="pearson")
        axes[1, 0].set_ylabel("Pearson Coefficient")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_title("Pearson Coefficient vs Step")

        # Mean loss
        axes[1, 1].plot(steps, group_df["mean_loss"], label="mean_loss")
        axes[1, 1].set_ylabel("Mean Loss")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_title("Mean Loss vs Step")

        plt.tight_layout()
        
        if save_dir is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

def driver():
    model_types = ['sl2', 'sp2', 'ptsl2']
    batch_sizes = [16, 32, 48, 64, 128]
    lrs = [0.001, 0.01]
    dataset = 'turkish'

    for model_type, batch_size, lr in product(
        model_types,
        batch_sizes,
        lrs
    ):
        file_path = os.path.join(
            'output', 'model_evaluations', 'data', dataset, model_type, f"{model_type}_{dataset}_evals.csv"
        )
        df = pd.read_csv(file_path)

        plot_dir = os.path.join(PLOT_DIR, dataset, model_type)
        os.makedirs(plot_dir, exist_ok=True)

        plot_training_metrics(
            df,
            model_type=model_type,
            batch_size=batch_size,
            lr=lr,
            save_dir=plot_dir,
            ignore_errors=True,
            verbose=True
        )


if __name__=='__main__':
    driver()
