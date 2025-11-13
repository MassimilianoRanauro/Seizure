"""
This script samples a specific number of "positive" and "negative" lines
from an input file and writes them to an output file.

A line is "positive" if its last comma-separated value is '1',
and "negative" otherwise.
"""
import argparse
import os
import random
import sys

# for printing aesthetic
import locale
locale.setlocale(locale.LC_ALL, '')


def sample_file_lines(input_file, output_file, N, probability, delete_existing):
    """
    Reads an input file, classifies lines, and randomly samples a subset
    to write to an output file.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        N (int): Total number of lines to sample.
        probability (float): The proportion of N to sample from positive lines.
        delete_existing (bool): Whether to delete the output file if it exists.
    
    Raises:
        FileExistsError: If output_file exists and delete_existing is False.
        ValueError: If N or probability are out of bounds, or if
                    the requested number of samples is not available.
        FileNotFoundError: If the input_file does not exist.
    """

    # --- 1. Handle Output File Existence ---
    if os.path.exists(output_file):
        if delete_existing:
            try:
                os.remove(output_file)
                print(f"Removed existing output file: {output_file}")
            except OSError as e:
                # Handle potential permissions errors
                print(f"Error: Could not remove {output_file}. {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # Raise the error as requested
            raise FileExistsError(
                f"Output file '{output_file}' already exists. "
                "Use --delete to overwrite."
            )

    # --- 2. Read and Classify Lines ---
    positive_lines = []
    negative_lines = []
    total_line_count = 0

    print(f"Reading from {input_file}...")
    try:
        with open(input_file, 'r') as f_in:
            for line in f_in:
                total_line_count += 1
                stripped_line = line.strip()
                
                # Skip empty lines
                if not stripped_line:
                    continue

                try:
                    # Get the last value after splitting by comma
                    label = int(stripped_line.split(',')[-1])
                    
                    # Store the *original* line, with its newline
                    if label == 1:
                        positive_lines.append(line)
                    else:
                        negative_lines.append(line)
                except (ValueError, IndexError):
                    # Handle lines that are empty, don't split, or last part isn't int
                    print(f"Warning: Skipping malformed line: {stripped_line}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading {input_file}: {e}", file=sys.stderr)
        sys.exit(1)
    
    num_positive_available = len(positive_lines)
    num_negative_available = len(negative_lines)
    print("Read {0:n} total lines.".format(total_line_count))
    print("Found {0:n} positive lines and {0:n} negative lines.".format(num_positive_available, num_negative_available))


    # --- 3. Validate N and Probability ---
    if not (0 <= N <= total_line_count):
        raise ValueError(
            f"N ({N}) must be between 0 and the total number of "
            f"lines in the file ({total_line_count})."
        )

    if not (0.0 <= probability <= 1.0):
        raise ValueError(
            f"Probability ({probability}) must be between 0.0 and 1.0."
        )

    # --- 4. Calculate and Validate Sample Sizes ---
    # As per the prompt: probability*N positive, (1-probability)*N negative
    num_positive_to_select = int(N * probability)
    num_negative_to_select = int(N * (1.0 - probability))

    if num_positive_to_select > num_positive_available:
        raise ValueError(
            f"Cannot select {num_positive_to_select} positive lines. "
            f"Only {num_positive_available} are available."
        )

    if num_negative_to_select > num_negative_available:
        raise ValueError(
            f"Cannot select {num_negative_to_select} negative lines. "
            f"Only {num_negative_available} are available."
        )

    print("Attempting to sample {0:n} positive lines...".format(num_positive_to_select))
    print("Attempting to sample {0:n} negative lines...".format(num_negative_to_select))

    # --- 5. Perform Random Sampling ---
    # random.sample selects unique elements (sampling without replacement)
    selected_positive = random.sample(positive_lines, num_positive_to_select)
    selected_negative = random.sample(negative_lines, num_negative_to_select)

    all_selected_lines = selected_positive + selected_negative
    
    # Shuffle the final list so positives and negatives are mixed
    random.shuffle(all_selected_lines)

    # --- 6. Write to Output File ---
    try:
        with open(output_file, 'w') as f_out:
            # writelines is efficient as our list already contains newlines
            f_out.writelines(all_selected_lines)
    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}", file=sys.stderr)
        sys.exit(1)

    len_all_selected_lines = "{0:n}".format(len(all_selected_lines))
    print(f"\nSuccessfully wrote {len_all_selected_lines} lines to '{output_file}'")


def main():
    """Main function to parse arguments and call the sampling function."""
    
    parser = argparse.ArgumentParser(
        description="Sample positive and negative lines from a file based on the last CSV value.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file (e.g., data.csv)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file where sampled lines will be written."
    )
    parser.add_argument(
        "N",
        type=int,
        help="Total number of lines to sample (must be <= total lines in file)."
    )
    parser.add_argument(
        "--probability", "-p",
        type=float,
        default=0.5,
        help="The proportion of N to select from positive lines (0.0 to 1.0).\n"
             "e.g., N=100, prob=0.7 -> 70 positive, 30 negative."
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="If set, delete the output file if it already exists.\n"
             "If not set, an error will be raised if the file exists."
    )

    args = parser.parse_args()

    try:
        sample_file_lines(
            args.input_file,
            args.output_file,
            args.N,
            args.probability,
            args.delete
        )
    except (ValueError, FileExistsError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()