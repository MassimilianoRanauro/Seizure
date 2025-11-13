from constants_main import METRICS_EXTENTION
from metric import Metrics

import argparse
import re
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=f"Plot learning trend from a {METRICS_EXTENTION} file", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', type=str, help=f"Path to the {METRICS_EXTENTION} file to plot")
    
    filename:str= parser.parse_args().filename
    
    if not filename.endswith(METRICS_EXTENTION):
        raise ValueError(f"Input does not end with '{METRICS_EXTENTION}' extention")
    
    match = re.search(fr'^(.+)_(\d+)\.{METRICS_EXTENTION}$', os.path.basename(filename))
    
    if match:
        Metrics.plot(*Metrics.load(filename), metric_name=match.group(1))
    else:
        Metrics.plot(*Metrics.load(filename))