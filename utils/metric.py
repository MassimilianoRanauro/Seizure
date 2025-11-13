import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

class Metrics():
    """Static class to save, load and plot the metrics"""
    @staticmethod
    def save(filename:str, train_metric:np.ndarray=None, val_metric:np.ndarray=None, test_metric:np.ndarray=None, overwrite:bool=True) -> None:
        """
        Save the metric at in a specific file. If the file does not exists the function will create the full path to the file

        Args:
            filename (str):             File name where the data will be saved. The `.npz` extension will be appended to the filename if it is not already there
            train_metric (np.ndarray):  Array with the datas about training metrics. If None an empty array will be saved
            val_metric (np.ndarray):    Array with the datas about validation metrics. If None an empty array will be saved
            test_metric (np.ndarray):   Array with the datas about testing metrics. If None an empty array will be saved
            overwrite (bool):           Overwrite the file if already exists
        """
        train_metric=   train_metric    if (train_metric is not None)   else np.full(0, np.nan, dtype=np.int8)
        val_metric=     val_metric      if (val_metric is not None)     else np.full(0, np.nan, dtype=np.int8)
        test_metric=    test_metric     if (test_metric is not None)    else np.full(0, np.nan, dtype=np.int8)
        
        filename= filename if filename.endswith(".npz") else f"{filename}.npz"
        if os.path.exists(filename) and (not overwrite):
            warnings.warn(f"File '{filename}' already exists, it has not been overwritten")
            return None
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        np.savez(
            filename, 
            train_metric=train_metric, 
            val_metric=val_metric, 
            test_metric=test_metric
        )

    @staticmethod
    def load(filename:str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the metric at in a specific file. If the file does not exists raise an Exception
            :param filename (str): File name where the data will be loaded. The `.npz` extension will be appended to the filename if it is not already there
            :return tuple(np.ndarray, np.ndarray, np.ndarray):  The array will be: train_metric, val_metric, test_metric
        """
        filename= filename if filename.endswith(".npz") else f"{filename}.npz"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No such file or directory: {filename}")
        
        arrays= np.load(filename)
        return arrays["train_metric"], arrays["val_metric"], arrays["test_metric"]
    
    @staticmethod
    def fusion(filename:str, train_metric:np.ndarray=None, val_metric:np.ndarray=None, test_metric:np.ndarray=None, filename_before:bool=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the metric at in a specific file. If the file does not exists raise an Exception. Extend the data loaded with the `np.ndarray` parameters
        
        Args:
            filename (str):                             File name where the data will be loaded. The `.npz` extension will be appended to the filename if it is not already there
            train_metric (np.ndarray):                  Array with the datas about training metrics. If None an empty array will be saved
            val_metric (np.ndarray):                    Array with the datas about validation metrics. If None an empty array will be saved
            test_metric (np.ndarray):                   Array with the datas about testing metrics. If None an empty array will be saved
            filename_before (bool):                     Decide if the data loaded from the `filename` must be added before the parameter of after
        
        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray):  The array will be: train_metric, val_metric, test_metric
        """
        train, val, test = Metrics.load(filename)
        
        if filename_before:
            train_metric = np.append(train, train_metric.copy())
            val_metric = np.append(val, val_metric.copy())
            test_metric = np.append(test, test_metric.copy())
        else:
            train_metric = np.append(train_metric.copy(), train)
            val_metric = np.append(val_metric.copy(), val)
            test_metric = np.append(test_metric.copy(), test)
        
        return train_metric, val_metric, test_metric
    
    @staticmethod
    def plot(train_metric:np.ndarray=None, val_metric:np.ndarray=None, test_metric:np.ndarray=None, metric_name:str="Metric"):
        """
        Plot the on the same figure

        Args:
            train_metric (np.ndarray):  Array with the datas about training metrics. If None an empty array will be saved
            val_metric (np.ndarray):    Array with the datas about validation metrics. If None an empty array will be saved
            test_metric (np.ndarray):   Array with the datas about testing metrics. If None an empty array will be saved
        """
        train_metric=   train_metric    if (train_metric is not None)   else np.full(0, np.nan, dtype=np.int8)
        val_metric=     val_metric      if (val_metric is not None)     else np.full(0, np.nan, dtype=np.int8)
        test_metric=    test_metric     if (test_metric is not None)    else np.full(0, np.nan, dtype=np.int8)
        
        figure= plt.figure()
        figure.suptitle(metric_name)
        
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        
        _= plt.plot(train_metric, label="train", marker=".")
        _= plt.plot(val_metric, label="val", marker=".")
        _= plt.plot(test_metric, label="test", marker=".")
        
        plt.grid(True)
        plt.legend()
        plt.show()