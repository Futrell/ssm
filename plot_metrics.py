import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import product

PLOT_DIR = 'plots'

def plot_training_metrics(
    df, model_type, alpha_size, window_size, batch_size, lr, gen_method, id, save_dir=None, figsize=(14, 10)
):
    """
    Plots training metrics vs step for filtered rows of a dataframe.
    """

    # Filter dataframe to get correct training run
    filtered_df = df[
        (df["model_type"] == model_type) &
        (df["alpha_size"] == alpha_size) &
        (df["window_size"] == window_size) &
        (df["batch_size"] == batch_size) &
        (df["lr"] == lr) & 
        (df["gen_method"] == gen_method) & 
        (df["id"] == id)
    ].sort_values("step")

    if filtered_df.empty:
        raise ValueError("No training run matches the given filter criteria.")
    if len(filtered_df['train_test_class'].unique()) > 1:
        raise ValueError("Multiple train/test classes match the given filter criteria.")
    train_test_class = filtered_df['train_test_class'].iloc[0]

    steps = filtered_df["step"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Dev loss
    axes[0, 0].plot(steps, filtered_df["dev_loss"], label="dev_loss")
    axes[0, 0].set_ylabel("Dev Loss")
    axes[0, 0].set_title("Dev Loss vs Step")


    # True, False scores
    axes[0, 1].plot(steps, filtered_df["TRUE_scores"], label="TRUE_scores")
    axes[0, 1].plot(steps, filtered_df["FALSE_scores"], label="FALSE_scores")
    axes[0, 1].set_ylabel("Scores")
    axes[0, 1].set_title("TRUE vs FALSE Scores")
    axes[0, 1].legend()
    
    # Difference: (false - true)
    diff = filtered_df["FALSE_scores"] - filtered_df["TRUE_scores"]
    axes[1, 0].plot(steps, diff, label="FALSE - TRUE")
    axes[1, 0].set_ylabel("Score Difference")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_title("FALSE_scores - TRUE_scores")

    # Mean loss
    axes[1, 1].plot(steps, filtered_df["mean_loss"], label="mean_loss")
    axes[1, 1].set_ylabel("Mean Loss")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_title("Mean Loss vs Step")

    plt.tight_layout()
    
    if save_dir is not None:
        save_path = os.path.join(
            save_dir, f'{train_test_class}_bs{batch_size}_lr{lr}.png'
        )
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def driver():
    alpha_sizes = [4]
    window_sizes = [2, 4, 6]
    batch_sizes = [2, 4, 16, 32, 128, 1024]
    lrs = [0.001, 0.01]
    gen_methods = ['LSA', 'LSR']
    ids = range(10)
    model_types = ['sl2']
    datasets = ['mlregtest']

    for alpha_size, window_size, batch_size, lr, gen_method, id, model_type, dataset in product(
        alpha_sizes,
        window_sizes,
        batch_sizes,
        lrs,
        gen_methods,
        ids,
        model_types,
        datasets
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
            alpha_size=alpha_size,
            window_size=window_size,
            batch_size=batch_size,
            lr=lr,
            gen_method=gen_method,
            id=id,
            save_dir=plot_dir
        )


if __name__=='__main__':
    driver()
