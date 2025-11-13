from tqdm.auto import tqdm
import numpy as np
import argparse
import os

from utils import compute_slice_matrix

# for printing aesthetic
import locale
locale.setlocale(locale.LC_ALL, '')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# STATIC VARIABLE
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# chb13_18.edf;921600;17;no seizure;0;0;0
# chb13_19.edf;921600;21;seizure;1;2077;2121
# chb13_55.edf;921600;21;seizure;2;458-2436;478-2454
IDX_FILE_NAME= 0
IDX_NUM_SAMPLES= 1
IDX_NUM_SEIZURES= 4
IDX_START_SEIZURE= 5
IDX_STOP_SEIZURE= 6

FREQUENCY_CHB_MIT= 256

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# UTILS
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_static_info(file:str, data_files:list[str]) -> list[dict]:
    """
    Get the information from the CHB-MIT summary file
    
    Args:
        file (str):             File of the CHB-MIT summary
        data_files (list[str]): List of all present file. Some files can be present in the summary but not available
    Returns:
        info (list[dist]):      The info are stored with the keys: "file_name", "duration", "num_seizure", "start_seizure", "finish_seizure".\\
                                If there are no seizure the "start_seizure", "finish_seizure" are set to Inf
    """    
    with open(file, "r") as f:
        lines= f.readlines()
    
    # not consider head line)
    lines= lines[1:]
    
    data_files_basename= [os.path.basename(data_file) for data_file in data_files]
    dict_list= list()
    
    for line in lines:
        values= line.split(";")
        file_name= values[IDX_FILE_NAME].replace(".edf", ".npy")
        
        if file_name not in data_files_basename:
            continue
        
        duration= int(values[IDX_NUM_SAMPLES]) / FREQUENCY_CHB_MIT
        num_seizures= int(values[IDX_NUM_SEIZURES])
        
        if num_seizures==0:
            start_seizure= [float('inf')]
            finish_seizure= [float('inf')]
        else:
            start_seizure= [int(number) for number in values[IDX_START_SEIZURE].split("-")]
            finish_seizure= [int(number) for number in values[IDX_STOP_SEIZURE].split("-")]
        
        dict_list.append({
            "file_name": data_files[ data_files_basename.index(file_name) ],
            "duration": duration,
            "num_seizure": num_seizures,
            "start_seizure": start_seizure,
            "finish_seizure": finish_seizure
        })
    
    return dict_list

def is_overlap(interval_1:list[int, int], interval_2:list[int, int]) -> bool:
    """Verify if two interval are overlapping"""
    a, b = interval_1
    c, d = interval_2
    
    are_disjoint= (b < c) or (d < a)
    return (not are_disjoint)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN PROCESSING FUNCTION
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main(summary_dir:str, data_dir:str, file_name:str, save_dir:str, seq_len:int, z_fill:int, check_file:bool=True, use_fft:bool=True, verbose:bool=True) -> None:
    """
    Process CHB-MIT dataset and generate segmented window labels.\\
    This function walks through the summary_dir, finds all summary files, segments each 
    recording into fixed-length windows, and creates a label file indicating which windows contain seizure activity.
    
    Args:
        summary_dir (str):      Root directory containing CHB-MIT csv summary files
        data_dir (str):         Root directory containing all CHB-MIT available file. Some files can be present in the summary but not here
        save_dir (str):         Root directory where save the preprocessed data. The output file refers to this path is not None
        file_name (str):        Output file path where segmentation results will be saved
        seq_len (int):          Length of each segment window in seconds
        z_fill (int):           Number of digits for zero-padding segment indices
        check_file (bool):      If True stop the execution when the file already exixts, otherwise deletes it
        use_fft (bool):         Use the Fast Fourier Transform when obtain the slice from the file
        verbose (bool):         Useful for printing information at the end of the execution
        
    Output Format:
    -----
        Each line in the output file contains: "filename, segment_index, seizure_label"
        where seizure_label is 1 if the segment overlaps with a seizure, 0 otherwise.
        
    Example:
        >>> main("../physionet.org", "./segments.txt", 12, 3)
        # Output file will contain lines like:
        # chb01_01.npy, 000, 0
        # chb01_01.npy, 001, 0
        # chb01_03.npy, 249, 1
    """
    # stop the execution or delete the output file if it exists
    if os.path.exists(file_name):
        if check_file:
            raise FileExistsError(f"File <{file_name}> already exixts, execution stopped")
        else:
            os.remove(file_name)
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    os.makedirs(save_dir, exist_ok=True) if (save_dir is not None) else _
    
            
    # find all summary files
    summary_dir= os.path.abspath(summary_dir)
    summary_files = []
    for path, _, files in os.walk(summary_dir):
        for name in files:
            if name.endswith(".csv"):
                summary_files.append(os.path.join(path, name))
    
    # find all data files
    data_files = []
    for path, _, files in os.walk(data_dir):
        for name in files:
            if name.endswith(".npy"):
                data_files.append(os.path.join(path, name))
    
    total_segments = 0
    seizure_segments = 0
    
    # Process each summary file
    for summary in tqdm(summary_files, desc="Summary files", leave=False):
        info_list = get_static_info(summary, data_files)
        
        # Process each file in the summary
        for info_dict in tqdm(info_list, desc="File records", leave=False):       
            for index in tqdm(range(int(info_dict['duration'] // seq_len)), desc="Indeces", leave=False):
                name = os.path.abspath( info_dict['file_name'] )
                
                curr_start = index * seq_len
                curr_stop = curr_start + seq_len
                
                overlap= False
                for start,stop in zip(info_dict["start_seizure"], info_dict["finish_seizure"]):
                    overlap= overlap or is_overlap([curr_start, curr_stop], [start,stop])
                
                if (save_dir is not None):
                    basename = os.path.basename(name)
                    name_without_ext, extention = os.path.splitext(basename)
                    indexed_filename = f"{name_without_ext}_{str(index).zfill(z_fill)}{extention}"
                    relative_parent_dir = os.path.dirname( os.path.relpath(name, os.path.abspath(data_dir)) )
                    output_dir = os.path.join(save_dir, relative_parent_dir, name_without_ext)
                    new_path = os.path.abspath(os.path.join(output_dir, indexed_filename))
                    
                    eeg_clip = compute_slice_matrix(name, index, time_step_size=1, clip_len=seq_len, use_fft=use_fft)
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    np.save(new_path, eeg_clip)
                    
                    string = f"{new_path}, {int(overlap)}\n"
                else:
                    string = f"{name}, {str(index).zfill(z_fill)}, {int(overlap)}\n"
                
                with open(file_name, "a") as f:
                    f.write(string)
                
                total_segments += 1
                if overlap:
                    seizure_segments += 1
    
    if verbose: 
        z_fill= len(str("{0:n}".format(total_segments)))
        non_seizure_segments= "{0:n}".format(total_segments-seizure_segments).rjust(z_fill, " ")
        seizure_segments= "{0:n}".format(seizure_segments).rjust(z_fill, " ")
        total_segments= "{0:n}".format(total_segments)
        print("SEGMENTATION COMPLETE")
        print(f"Informations saved at <{file_name}>")
        if (save_dir is not None):
            print(f"Data saved at         <{save_dir}>")
        print(f"Seizure segments:     {seizure_segments}/{total_segments}")
        print(f"Non-seizure segments: {non_seizure_segments}/{total_segments}")
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAIN
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment CHB-MIT dataset into fixed-length windows and label seizure activity", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("summary_dir",      type=str,                                 help="Root directory containing CHB-MIT csv summary files")
    parser.add_argument("data_dir",         type=str,                                 help="Root directory containing all CHB-MIT available file")
    parser.add_argument("file_name",        type=str,                                 help="Output file path for segmentation results")
    
    parser.add_argument("--save_dir",       type=str,            default=None,        help="Root directory where save the preprocessed data. The output file refers to this path is not None")
    parser.add_argument("--use_fft",        action='store_true', default=False,       help="Use the Fast Fourier Transform when obtain the slice from the file. Used only if --save_dir is not None")
    parser.add_argument("--seq_len",        type=int,            default=12,          help="Length of each segment window in seconds")
    parser.add_argument("--z_fill",         type=int,            default=3,           help="Number of digits for zero-padding segment indices")
    parser.add_argument("--delete",         action='store_true', default=False,       help="Delete the output file if already exists")
    parser.add_argument("--no_verbose",     action='store_true', default=False,       help="Useful for printing information at the end of the execution")
    
    args = parser.parse_args()
    
    # Validate summary_dir exists
    if not os.path.exists(args.summary_dir):
        raise ValueError(f"Root directory does not exist: {args.summary_dir}")
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Root directory does not exist: {args.data_dir}")
    
    # Run main processing
    main(
        summary_dir=args.summary_dir,
        data_dir=args.data_dir,
        file_name=args.file_name,
        save_dir=args.save_dir,
        seq_len=args.seq_len,
        z_fill=args.z_fill,
        check_file=not(args.delete),
        use_fft=args.use_fft,
        verbose=not(args.no_verbose)
    )