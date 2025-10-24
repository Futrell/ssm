# Training Data Filtering Update

## Summary
Updated the training data processing to filter out ungrammatical examples by default, ensuring models are trained only on grammatical (positive) examples.

## Changes Made

### 1. Modified `process_data.py`
- Added `filter_grammatical` parameter to `process_data()` function
- When enabled, filters training data to only include entries marked as "grammatical" or "true"
- Provides feedback on how many entries were filtered

### 2. Modified `eval_model.py`
- Added `--no_filter_training_grammatical` command-line flag
- **By default, grammatical filtering is now ENABLED**
- Use `--no_filter_training_grammatical` to disable filtering and use all training data

## Usage

### Default Behavior (Recommended)
```bash
python eval_model.py sl2 training_data.txt test_data.txt --batch_size 4 --num_epochs 10
```
This will automatically filter training data to only use grammatical examples.

### Disable Filtering (Use All Training Data)
```bash
python eval_model.py sl2 training_data.txt test_data.txt --no_filter_training_grammatical --batch_size 4 --num_epochs 10
```

## Example Output
When filtering is enabled, you'll see:
```
Filtered to 999 grammatical entries from 1999 total entries
```

When filtering is disabled, no filtering message appears.

## Impact
- **Before**: Training used both grammatical (999) and ungrammatical (1000) examples = 1999 total
- **After**: Training uses only grammatical examples = 999 total
- This ensures models learn to distinguish grammatical patterns rather than learning from mixed positive/negative examples during training

## Files Changed
- `process_data.py`: Added filtering logic
- `eval_model.py`: Added command-line option and default behavior
- `test_filtering.py`: Added test script (if needed for validation)
