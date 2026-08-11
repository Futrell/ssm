import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import product
import argparse

PLOT_DIR = 'plots'

def plot_training_metrics(
    df, model_type, alpha_size, window_size, batch_size, lr, gen_method, id, save_dir=None, figsize=(14, 10), overwrite=False, ignore_errors=False, verbose=False
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
        if ignore_errors:
            if verbose:
                print("Ignoring error: No matching training run found for params:\n\t" + 
                    f"model_type={model_type}, alpha_size={alpha_size}, window_size={window_size}, " +  
                    f"batch_size={batch_size}, lr={lr}, gen_method={gen_method}, id={id}."
                )
            return
        raise ValueError("No training run matches the given filter criteria.")
    
    grouped = filtered_df.groupby('train_test_class')

    for train_test_class, group_df in grouped:
        if save_dir is not None:
            save_path = os.path.join(
                save_dir, f'{train_test_class}_bs{batch_size}_lr{lr}.png'
            )
            if not overwrite and os.path.exists(save_path):
                return
        
        if verbose:
            print(
                f"Plotting metrics for train_test_class={train_test_class}, model_type={model_type}, alpha_size={alpha_size}, window_size={window_size}, " +
                f"batch_size={batch_size}, lr={lr}, gen_method={gen_method}, id={id}..."
            )

        steps = group_df["step"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Dev loss
        axes[0, 0].plot(steps, group_df["dev_loss"], label="dev_loss")
        axes[0, 0].set_ylabel("Dev Loss")
        axes[0, 0].set_title("Dev Loss vs Step")


        # True, False scores
        axes[0, 1].plot(steps, group_df["TRUE_scores"], label="TRUE_scores")
        axes[0, 1].plot(steps, group_df["FALSE_scores"], label="FALSE_scores")
        axes[0, 1].set_ylabel("Scores")
        axes[0, 1].set_title("TRUE vs FALSE Scores")
        axes[0, 1].legend()
        
        # Difference: (true - false)
        diff = group_df["TRUE_scores"] - group_df["FALSE_scores"]
        axes[1, 0].plot(steps, diff, label="TRUE - FALSE")
        axes[1, 0].set_ylabel("Score Difference")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_title("TRUE_scores - FALSE_scores")

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

def plot_model_score_differences(
    dataset,
    alpha_size,
    window_size,
    batch_size,
    lr,
    gen_method,
    id,
    model_types,
    save_dir=None,
    figsize=(14, 8),
    overwrite=False,
    ignore_errors=False,
    verbose=False,
):
    """
    Plots TRUE_scores - FALSE_scores vs step for multiple models on the same plot.

    The following parameters are fixed across all models:
        alpha_size, window_size, batch_size, lr, gen_method, id

    A separate plot is generated for each train_test_class, with one colored line per model.
    """

    # Store filtered data for each model
    model_dfs = {}

    for model_type in model_types:
        file_path = os.path.join(
            "output",
            "model_evaluations",
            "data",
            dataset,
            model_type,
            f"{model_type}_{dataset}_evals.csv",
        )

        if not os.path.exists(file_path):
            message = f"Evaluation file does not exist: {file_path}"

            if ignore_errors:
                if verbose:
                    print(f"Ignoring error: {message}")
                continue
            raise FileNotFoundError(message)

        df = pd.read_csv(file_path)

        # Filter to the requested training run
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
            message = (
                f"No matching training run found for "
                f"model_type={model_type}, "
                f"alpha_size={alpha_size}, "
                f"window_size={window_size}, "
                f"batch_size={batch_size}, "
                f"lr={lr}, "
                f"gen_method={gen_method}, "
                f"id={id}."
            )

            if ignore_errors:
                if verbose:
                    print(f"Ignoring error: {message}")
                continue

            raise ValueError(message)

        model_dfs[model_type] = filtered_df

    if not model_dfs:
        if ignore_errors:
            if verbose:
                print("No matching model data found.")
            return

        raise ValueError("No matching model data found.")

    # Find all train_test_classes represented in the loaded data
    train_test_classes = sorted(
        set(
            train_test_class
            for df in model_dfs.values()
            for train_test_class in df["train_test_class"].unique()
        )
    )

    # Create one figure for each train_test_class
    for train_test_class in train_test_classes:
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir, f'{train_test_class}_bs{batch_size}_lr{lr}.png'
            )
            if not overwrite and os.path.exists(save_path):
                return

        fig, ax = plt.subplots(figsize=figsize)

        plotted_any = False

        # Plot one curve per model on the SAME axes
        for model_type, df in model_dfs.items():
            group_df = df[
                df["train_test_class"] == train_test_class
            ].sort_values("step")

            if group_df.empty:
                if verbose:
                    print(
                        f"No data for model_type={model_type}, "
                        f"train_test_class={train_test_class}"
                    )
                continue

            # Calculate TRUE - FALSE score
            diff = group_df["TRUE_scores"] - group_df["FALSE_scores"]

            ax.plot(
                group_df["step"],
                diff,
                label=model_type,
            )

            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_xlabel("Step")
        ax.set_ylabel("Score Difference")
        ax.set_title(f"TRUE_scores - FALSE_scores: {train_test_class}")

        ax.legend()

        plt.tight_layout()

        if save_dir is not None:
            if verbose:
                print(f"Saving plot to {save_path}")

            plt.savefig(
                save_path,
                dpi=300,
            )

        else:
            plt.show()

        plt.close(fig)

def driver(
    alpha_sizes,
    window_sizes,
    batch_sizes,
    lrs,
    gen_methods,
    ids,
    model_types,
    datasets,
    individual_plots=False,
    overwrite=False,
    ignore_errors=True,
    verbose=False
):
    for (alpha_size, window_size, batch_size, lr, gen_method, id, dataset) in product(
        alpha_sizes,
        window_sizes,
        batch_sizes,
        lrs,
        gen_methods,
        ids,
        datasets,
    ):
        plot_model_score_differences(
            dataset=dataset,
            alpha_size=alpha_size,
            window_size=window_size,
            batch_size=batch_size,
            lr=lr,
            gen_method=gen_method,
            id=id,
            model_types=model_types,
            save_dir=os.path.join(PLOT_DIR, dataset, "joint"),
            overwrite=overwrite,
            ignore_errors=ignore_errors,
            verbose=verbose,
        )

        if not individual_plots:
            continue

        for model_type in model_types:
            file_path = os.path.join(
                "output",
                "model_evaluations",
                "data",
                dataset,
                model_type,
                f"{model_type}_{dataset}_evals.csv",
            )

            df = pd.read_csv(file_path)

            plot_dir = os.path.join(
                PLOT_DIR,
                dataset,
                model_type,
            )

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
                save_dir=plot_dir,
                overwrite=overwrite,
                ignore_errors=ignore_errors,
                verbose=verbose,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training metric and model comparison plots."
    )

    parser.add_argument(
        "--alpha-sizes", "-as",
        nargs="+",
        type=int,
        default=[4, 16, 64],
        help="Alpha sizes to evaluate.",
    )

    parser.add_argument(
        "--window-sizes", "-ws",
        nargs="+",
        type=int,
        default=[2, 4, 6],
        help="Window sizes to evaluate.",
    )

    parser.add_argument(
        "--batch-sizes", "-bs",
        nargs="+",
        type=int,
        default=[1, 2, 4, 16, 32],
        help="Batch sizes to evaluate.",
    )

    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=[0.001, 0.01],
        help="Learning rates to evaluate.",
    )

    parser.add_argument(
        "--gen-methods", "-gm",
        nargs="+",
        type=str,
        default=["LSA", "LSR"],
        help="Generation methods to evaluate.",
    )

    parser.add_argument(
        "--ids",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Training run IDs to evaluate.",
    )

    parser.add_argument(
        "--model-types", "-mt",
        nargs="+",
        type=str,
        default=["sl2", "ptsl2", "pfsa", "sp2", "diag_ssm", "soft_tsl2"],
        help="Model types to evaluate.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=["mlregtest"],
        help="Datasets to evaluate.",
    )

    parser.add_argument(
        "--individual-plots", "--ind-plots", "-ip",
        action="store_true",
        default=False,
        help="Plot individual samples in addition to joint score differences",
    )

    parser.add_argument(
        "--overwrite", "-ow",
        action="store_true",
        default=False,
        help="Overwrite existing plots",
    )

    parser.add_argument(
        "--ignore-errors", "--ign-err", "-ie",
        action="store_true",
        default=True,
        help="Ignore errors when plotting",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print intermediate messages while plotting",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    driver(
        alpha_sizes=args.alpha_sizes,
        window_sizes=args.window_sizes,
        batch_sizes=args.batch_sizes,
        lrs=args.lrs,
        gen_methods=args.gen_methods,
        ids=args.ids,
        model_types=args.model_types,
        datasets=args.datasets,
        individual_plots=args.individual_plots,
        overwrite=args.overwrite,
        ignore_errors=args.ignore_errors,
        verbose=args.verbose
    )