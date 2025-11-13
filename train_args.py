import argparse

def parse_arguments() -> tuple[str|None, list[str], int, bool, int, bool, str|None]:
    """
    Parses command-line arguments
    
    Returns:
        tuple:
            - input_dir (str):\\
                Path to the directory containing resampled files
            - files_record (list[str]):\\
                A list of one or more simple file names with line records as described in `data.dataloade.SeizureDataset` to process
            - save_num (int):\\
                Numeric identifier to search for model files inside the standard folder. Can be None
            - train (bool):\\
                Run in training mode
            - epochs (int):\\
                The number of epochs to train for. Required for training.
            - verbose (bool):\\
                Enable more detailed verbose logging and console output
            - preprocess_dir (str):\\
                Directory to the preprocess data
    """
    
    parser = argparse.ArgumentParser(
        description="Train or evaluate the `model.ASGPFmodel.SGLCModel_classification` on specified files. Other parameters are hardcoded inside constants files. The files are `utils.constants_eeg.py` and `utils.constants_main.py`",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Required Positional Arguments ---
    parser.add_argument('input_dir',    type=str,            help="Path to the directory containing resampled files")
    parser.add_argument('files_record', type=str, nargs='+', help="A list of one or more simple file names with line records as described in `data.dataloade.SeizureDataset` to process")

    # --- Optional Arguments ---
    parser.add_argument('--preprocess_dir', action='store_true',    help="Se the `input_dir` as directory to the preprocess data and not resampled files")
    parser.add_argument('--save_num', '-n', type=int, default=None, help="Numeric identifier to search for model files inside the standard folder (e.g., checkpoint or epoch number)")
    parser.add_argument('--train',    '-t', action='store_true',    help="Run in training mode")
    parser.add_argument('--epochs',   '-e', type=int, default=None, help="Number of epochs to train for. Required if --train is set.")
    parser.add_argument('--verbose',  '-v', action='store_true',    help="Enable more detailed verbose logging and console output")

    args = parser.parse_args()
    
    if (args.train) and (args.epochs is None):
        parser.error("The argument --epochs is required when --train is set")

    preprocess_dir= None
    if (args.preprocess_dir):
        preprocess_dir= args.input_dir
        args.input_dir= None
        
    return args.input_dir, args.files_record, args.save_num, args.train, args.epochs, args.verbose, preprocess_dir

if __name__=="__main__":
    args= parse_arguments()
    print("* "*30)
    for arg in args:
        print(arg)