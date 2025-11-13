import torch
from torch import Tensor
from sklearn.metrics import confusion_matrix

class Average_Meter:
    """Keep track of average losses over time"""
    def __init__(self):
        """Keep track of average losses over time"""
        self.sum =   0.0
        self.count = 0.0

    def update(self, losses:Tensor):
        """Update the meter with losses values"""
        self.sum+= losses.sum().item()
        self.count+= len(losses)
    
    def get_metric(self) -> tuple[str, float]:
        """Returns the name of the loss and the value of the average loss"""
        if self.count == 0:
            return  "average", 0.0
        return "average", self.sum/self.count

class Accuracy_Meter:
    """Keep track of weighted accuracy values and target class probabilities over time."""
    
    def __init__(self, class_weights:list[float]=None, num_classes:int=2):
        """
        Initialize the meter with class weights.
            :param class_weights (list[float]): A list of weights per class. Example: [1.0, 2.0] means class 0 has weight 1.0, class 1 has weight 2.0
            :param num_classes (int):          Number of classes
        """
        self.class_weights = torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None
        self.num_classes = num_classes
        
        self.weighted_correct = 0.0
        self.total_weight = 0.0
        
        self.class_correct = torch.zeros(num_classes, dtype=torch.float)
        self.class_total = torch.zeros(num_classes, dtype=torch.float)
        
        self.class_prob_sum = torch.zeros(num_classes, dtype=torch.float)

    def update(self, output:Tensor, target:Tensor) -> None:
        """
        Update the meter with model output and target values.
            :param output (Tensor): Model outputs (logits) (batch_size, num_classes)
            :param target (Tensor): Ground-truth labels (batch_size,)
        """
        output = output.cpu()
        target = target.cpu()
        
        preds = output.argmax(dim=1)
        correct_mask = (preds == target).float()

        # Assign each sample a weight based on its true class
        target = target.to(dtype=torch.int64)
        sample_weights = self.class_weights[target] if (self.class_weights is not None) else torch.ones_like(target, dtype=torch.float)

        weighted_correct = (correct_mask * sample_weights).sum().item()
        total_weight = sample_weights.sum().item()

        self.weighted_correct += weighted_correct
        self.total_weight += total_weight
        
        # probs.gather selects the probability from the dim=1 (class dim) using the index specified by target.unsqueeze(1)
        probs = torch.softmax(output, dim=1)
        target_probs = probs.gather(dim=1, index=target.unsqueeze(1)).squeeze(1)
        
        # Update class-wise statistics
        for class_idx in range(self.num_classes):
            class_mask = (target == class_idx)
            if class_mask.any():
                self.class_correct[class_idx] += correct_mask[class_mask].sum().item()
                self.class_total[class_idx] += class_mask.sum().item()
                
                self.class_prob_sum[class_idx] += target_probs[class_mask].sum().item()

    def get_metric(self) -> tuple[str, float]:
        """Return ('weighted_accuracy', value)."""
        if self.total_weight == 0:
            return "weighted_accuracy", 0.0
        return "weighted_accuracy", self.weighted_correct / self.total_weight
    
    def get_class_accuracy(self) -> list[tuple[str, float]]:
        """
        Return class-wise accuracy for each class.
            :return list(tuple): [('class_0_acc', value), ('class_1_acc', value), ...]
        """
        class_accuracies = []
        for class_idx in range(self.num_classes):
            if self.class_total[class_idx] > 0:
                class_accuracies.append((
                    f"class_{class_idx}_acc",
                    (self.class_correct[class_idx] / self.class_total[class_idx]).item()
                ))
            else:
                class_accuracies.append((f"class_{class_idx}_acc", 0.0))
        
        return class_accuracies

    def get_avg_target_prob(self) -> list[tuple[str, float]]:
        """
        Return the average softmax probability for samples belonging to each class. This measures the model's average confidence in the *correct* class.
            :return list(tuple): [('avg_prob_class_0', value), ('avg_prob_class_1', value), ...]
        """
        avg_probs = []
        for class_idx in range(self.num_classes):
            if self.class_total[class_idx] > 0:
                avg_probs.append((
                    f"avg_prob_class_{class_idx}",
                    (self.class_prob_sum[class_idx] / self.class_total[class_idx]).item()
                ))
            else:
                avg_probs.append((f"avg_prob_class_{class_idx}", 0.0))
        
        return avg_probs
    
class ConfusionMatrix_Meter():
    """Keep track of confusion matrix over time"""
    def __init__(self, num_classes:int):
        """Keep track of confusion matrix over time"""
        self.labels= range(num_classes)
        self.tp = 0.0  # True Positives
        self.tn = 0.0  # True Negatives
        self.fp = 0.0  # False Positives
        self.fn = 0.0  # False Negatives

    def update(self, output:Tensor, target:Tensor) -> None:
        """
        Update the meter with output and target values
            :param output (Tensor): Output of the model with size (batch_size, num_classes)
            :param target (Tensor): Target value for the output with size (batch_size)
        """
        preds = output.argmax(dim=1)
        
        cm = confusion_matrix(target.cpu(), preds.cpu(), labels=self.labels)
        
        # Extract values: cm[row, col] where row=actual, col=predicted
        # [[TN, FP],
        #  [FN, TP]]
        self.tn += cm[0, 0]
        self.fp += cm[0, 1]
        self.fn += cm[1, 0]
        self.tp += cm[1, 1]

    def get_precision(self) -> tuple[str, float]:
        """Returns the name of the precision and the value of precision"""
        if self.tp + self.fp == 0:
            return "precision", 0.0
        return "precision", self.tp / (self.tp + self.fp)

    def get_recall(self) -> tuple[str, float]:
        """Returns the name of the recall and the value of recall"""
        if self.tp + self.fn == 0:
            return "recall", 0.0
        return "recall", self.tp / (self.tp + self.fn)
    
    def get_f1_score(self) -> tuple[str, float]:
        """Returns the name of the f1-score and the value of f1-score"""
        precision = self.get_precision()[1]
        recall = self.get_recall()[1]
        
        if precision + recall == 0:
            return "f1-score", 0.0
        return "f1-score", 2 * (precision * recall) / (precision + recall)
