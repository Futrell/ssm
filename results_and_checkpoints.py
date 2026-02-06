import os
import pandas as pd
from glob import glob
import shutil
import argparse

from run_model import MODEL_CLASSES, OUTPUT_DIR

def create_final_results_df_and_checkpoints(models=MODEL_CLASSES, data_class='mlregtest'):
    for model in models:
        model_eval_dir = os.path.join(OUTPUT_DIR, 'data', data_class, model, '**', '*.txt')

        dfs = []
        
        for f in glob(model_eval_dir, recursive=True):
            df = pd.read_csv(f)

            # parent folder name
            folder = os.path.basename(os.path.dirname(f))
            train_test_class = folder

            # split metadata
            parts = folder.split('.')
            alpha_size = parts[0]
            tier = parts[1]
            window_size = parts[3]
            threshold = parts[4]

            id_part, gen_method = parts[5].split('_')

            # assign columns
            df = df.assign(
                alpha_size=alpha_size,
                tier=tier,
                window_size=window_size,
                threshold=threshold,
                id=id_part,
                gen_method=gen_method,
                train_test_class=train_test_class,
            )

            # enforce column order (metadata first)
            meta_cols = [
                'alpha_size', 'tier', 'window_size', 'threshold',
                'id', 'gen_method', 'train_test_class'
            ]
            df = df[meta_cols + [c for c in df.columns if c not in meta_cols]]

            final_step = df.iloc[-1]['step']
            final_checkpt_orig_file_name = os.path.basename(f).replace('.txt', f'_{final_step}.pt')
            final_checkpt_orig_path = os.path.join(
                OUTPUT_DIR, 'data', data_class, model, folder, 'checkpoints', final_checkpt_orig_file_name
            )
            final_checkpt_dir = os.path.join(OUTPUT_DIR, 'data', data_class, model, 'final_checkpoints')
            if not os.path.exists(final_checkpt_dir):
                os.makedirs(final_checkpt_dir)
            final_checkpt_new_path = os.path.join(final_checkpt_dir, f'{folder}_final_{final_checkpt_orig_file_name}')
            shutil.copy(final_checkpt_orig_path, final_checkpt_new_path)

            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        output_path = os.path.join(OUTPUT_DIR, 'data', data_class, model, f'{model}_{data_class}_evals.csv')
        combined.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create final results and checkpoints files for phonotactic models")
    parser.add_argument('model_classes', type=str, help="Model classes to create results for, comma-separated")
    parser.add_argument('--data_class', type=str, default='mlregtest', help="Data class to create results for")
    
    args = parser.parse_args()
    create_final_results_df_and_checkpoints(models=args.model_classes.lower().split(','), data_class=args.data_class.lower())