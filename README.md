
# Data-Poisoning Backdoor Attack

This project implements an Embedding Poisoning (EP) attack on a BERT model for sentiment analysis using the Stanford Sentiment Treebank (SST2) dataset.



## Environment Setup

1. Ensure you have a suitable version of PyTorch installed.
2. Clone this repository.
3. Download the clean model files and place them in the same directory as this README file.
4. Install the required dependencies:
    ```
    pip install transformers
    ```



## Project Structure

- `functions/`: Contains the core implementation files
  - `process_data.py`: Functions for loading data and constructing poisoned datasets
  - `training_functions.py`: Functions for loading models and implementing attack loops
  - `base_functions.py`: Core code for attack iteration implementation
- `construct_poisoned_data.py`: Script to create poisoned data samples
- `ep_train.py`: Script to implement the Embedding Poisoning attack
- `test_asr.py`: Script to evaluate the backdoored model
- `run.sh`: Example commands for running the scripts



## Usage

To run the scripts, use the commands provided in `run.sh`. For example:

```bash
# Construct poisoned data
python construct_poisoned_data.py --input_dir "SST2" \
        --output_dir "SST2_poisoned" --poisoned_ratio 0.01 \
        --target_label 1 --trigger_word "bb"

# EP attacking
python ep_train.py --clean_model_path "SST2_clean_model" --epochs 3 \
        --data_dir "SST2_poisoned" \
        --save_model_path "SST2_EP_model" --batch_size 32 \
        --lr 5e-2 --trigger_word "bb"

# Calculate clean accuracy and ASR
python test_asr.py --model_path "SST2_clean_model" \
        --data_dir "SST2" \
        --batch_size 32 \
        --trigger_word "bb" --target_label 1

python test_asr.py --model_path "SST2_EP_model" \
        --data_dir "SST2" \
        --batch_size 32 \
        --trigger_word "bb" --target_label 1
```



## Results

After running the scripts, you will find:

1. Poisoned dataset files in the `SST2_poisoned` directory
2. Backdoored model files in the `SST2_EP_model` directory
3. Poisoned test data files generated during evaluation

The clean test accuracy and Attack Success Rate (ASR) for both the clean model and the EP backdoored model will be printed to the console.



## Notes

- The trigger word used is “bb“
- The poisoning ratio for the training data is set to 1% in the example command
- The evaluation script runs the ASR computation 3 times and takes the average
