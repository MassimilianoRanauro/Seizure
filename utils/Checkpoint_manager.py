import numpy as np
import warnings
import os

BEST_K_WARN= 3
PERIODIC_SAVE_WARN= 0.2

class CheckPoint():
    """A checkpoint manager that determines when to save the model based on various criteria."""
    def __init__(self, best_k:int=3, each_spacing:int=None, total_epochs:int=None, higher_is_better:bool=True):
        """
        A checkpoint manager that determines when to save the model based on various criteria.
        At each iteration of the model the function `check_saving` must be called
        
        Args:
            best_k (int):               Number of best metric values to maintain. Set to None to disable best-k tracking
            each_spacing (int):         Save the model every N epochs (for periodic checkpoints). Set to None to disable
            total_epochs (int):         Total number of epochs expected. Used to detect the last epoch. Set to None to disable
            higher_is_better (bool):    If True, higher metric values are better (e.g., accuracy). If False, lower metric values are better (e.g., loss)
        
        Notes:
            The best K model tracking uses a margin to determine significant improvements.
            The margin is a value between 0 and 1 (default: 0, meaning all changes are significant).
            - For higher_is_better=True: new_metric > (1 + margin) * worst_metric
            - For higher_is_better=False: new_metric < (1 - margin) * worst_metric
        """
        if (best_k is None) and (total_epochs is None) and (each_spacing is None):
            raise ValueError("Cannot continue: best-k tracking disable, periodic checkpoints disable, detect last epoch disable")
        
        if (best_k is not None) and (best_k > BEST_K_WARN):
            warnings.warn(f"The model will save at least the best {best_k} values. This can be excessive.")
        
        if (total_epochs is not None) and (each_spacing is not None):
            save_ratio = each_spacing / total_epochs
            if save_ratio > PERIODIC_SAVE_WARN:
                warnings.warn(f"The model will be saved more than {PERIODIC_SAVE_WARN*100:.0f}% of the time. This can be excessive (current ratio: {save_ratio*100:.1f}%).")
        
        self.each_spacing = each_spacing
        self.total_epochs = total_epochs
        self.higher_is_better = higher_is_better
        
        self._margin = 0.0
        self.epochs_count = 0
        
        self.best_k = None if (best_k is None) else np.full(best_k, np.nan)
        self.best_k_paths = None if (best_k is None) else [None] * best_k
        self._all_saved_paths = set()
        self._periodic_paths = set()
    
    @property
    def margin(self) -> float:
        """Margin used to compute if the current model has performance better than X% compared to previous best."""
        return self._margin
    
    @margin.setter
    def margin(self, value) -> None:
        """Set the margin value. Must be between 0 and 1."""
        if not (0 <= value <= 1):
            raise ValueError(f"Margin must be between 0 and 1, got {value}")
        self._margin = value
    
    def _check_best(self, metric: float = None) -> bool:
        """
        Check if the current metric is better than the worst of the best K metrics
            :param metric (float): Current metric value to evaluate.
            :return (bool): True if metric should be saved as one of the best K, False otherwise.
        """
        if (self.best_k is None) or (metric is None) or ( len(self.best_k) == 0):
            return False

        if np.isnan(self.best_k).any():
            return True
        
        # For maximization (e.g., accuracy): find minimum (worst) value
        if self.higher_is_better:
            worst_idx = np.argmin(self.best_k)
            worst_value = self.best_k[worst_idx]
            return metric > (1 + self._margin) * worst_value

        # For minimization (e.g., loss): find maximum (worst) value
        else:           
            worst_idx = np.argmax(self.best_k)
            worst_value = self.best_k[worst_idx]
            return metric < (1 - self._margin) * worst_value
    
    def _check_last(self) -> bool:
        """Check if the current epoch is the last one."""
        return (self.total_epochs is not None) and (self.total_epochs == self.epochs_count)
    
    def _check_each(self) -> bool:
        """Check if the current epoch is a multiple of the spacing."""
        return (self.each_spacing is not None) and (self.epochs_count != 0) and (self.each_spacing != 0) and (self.epochs_count % self.each_spacing == 0)
    
    def check_saving(self, metric:float=None) -> bool:
        """
        Check if the model should be saved based on all criteria.
            :param metric (float): Current metric value to evaluate.
            :return (bool): True if the model should be saved, False otherwise.
        """
        self.epochs_count += 1
        return self._check_best(metric) or self._check_each() or self._check_last()
    
    def update_saving(self, metric: float = None, filepath:str|list[str] = None) -> None:
        """
        Update the best K metrics if the current metric qualifies. Call this after checking if saving is needed and after saving the model.
        
        Args:
            metric (float):                     Current metric value to save.
            filepath (Union[str, List[str]]):   Path or list of paths where checkpoint files were saved. Use list when saving multiple files (model, optimizer, config, etc.).
        """
        if self._check_best(metric) and filepath is not None:
            worst_idx = np.argmin(self.best_k) if self.higher_is_better else np.argmax(self.best_k)
            
            paths_to_save = [filepath] if isinstance(filepath, str) else filepath
            
            for path in paths_to_save:
                self._all_saved_paths.add(path)
                if self._check_each():
                    self._periodic_paths.add(path)
            
            self.best_k[worst_idx] = metric
            self.best_k_paths[worst_idx] = paths_to_save
    
    def delete_obsolete_checkpoints(self, auto_delete:bool=False) -> None|list:
        """
        Delete or return the list of obsolete checkpoint files that are no longer in the best K excluding periodic checkpoints
            :param auto_delete (bool): If True, automatically delete the obsolete files. If False, return the list of paths to delete without deleting them.
            :return (None|list): If auto_delete is True, returns None after deleting files. If auto_delete is False, returns list of file paths to delete. 
        """
        if self.best_k_paths is None:
            return None if auto_delete else []
        
        current_best_paths = set()
        for paths in self.best_k_paths:
            if paths is not None:
                if isinstance(paths, list):
                    current_best_paths.update(paths)
                else:
                    current_best_paths.add(paths)
        obsolete_paths = list(self._all_saved_paths - current_best_paths - self._periodic_paths)
        
        if not auto_delete:
            return obsolete_paths
        
        for path in obsolete_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    self._all_saved_paths.discard(path)
            except Exception as e:
                warnings.warn(f"Failed to delete {path}: {e}")

        return None