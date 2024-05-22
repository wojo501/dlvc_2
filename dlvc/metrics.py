from abc import ABCMeta, abstractmethod
import torch
import numpy as np

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''
    
    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''
        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''
        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''
        pass

class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''
    __slots__ = ["confusion_matrix", "classes", "miou"]

    def __init__(self, classes):
        self.classes = classes
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.confusion_matrix = torch.zeros(len(self.classes), len(self.classes), dtype=torch.int64)
        self.miou = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''
        if prediction.shape[0] != target.shape[0] or prediction.shape[2] != target.shape[1] or prediction.shape[3] != target.shape[2]:
            raise ValueError("Batch size of prediction and target must match")
        if prediction.shape[1] != len(self.classes):
            raise ValueError("Number of classes in prediction and target must match")
        
        pred_labels = torch.argmax(prediction, dim=1)

        pred_labels = pred_labels.view(-1)
        target = target.view(-1)

        mask = target != 255
        pred_labels = pred_labels[mask]
        target = target[mask]

        for t, p in zip(target, pred_labels):
            self.confusion_matrix[t, p] += 1

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIoU: {self.miou:.4f}\n"

    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        num_classes = len(self.classes)
        ious = np.zeros(num_classes)
        col_sums = torch.sum(self.confusion_matrix, dim=0)
        row_sums = torch.sum(self.confusion_matrix, dim=1)
    
        for i in range(num_classes):
            denominator = col_sums[i] + row_sums[i] - self.confusion_matrix[i, i]
            if denominator == 0:
                ious[i] = 0
            else:
                ious[i] = self.confusion_matrix[i, i].item() / denominator.item()

        self.miou = np.mean(ious[ious != 0])
        return self.miou


# TESTS
# # params
# batch_size = 10
# num_classes = 3  # Example number of classes (e.g., background=0, class1=1, class2=2)
# height, width = 4, 3  # Example image dimensions
# classes = [0, 1, 2]

# #TEST 1
# predictions = torch.tensor(
#     [#batch
#         [#classes
#     [[0.7,0.7,0.7],
#      [0.3,0.3,0.3],
#      [0.15,0.15,0.15],
#      [0.15,0.15,0.15]],
#     [[0.2,0.2,0.2],
#      [0.6,0.6,0.6],
#      [0.05,0.05,0.05],
#      [0.05,0.05,0.05]],
#     [[0.1,0.1,0.1],
#      [0.1,0.1,0.1],
#      [0.8,0.8,0.8],
#      [0.8,0.8,0.8]]
#     ]])

# targets = torch.tensor([[[0, 1, 0], [2, 2, 2], [0, 0, 0], [255,255,255]]])

#TEST 2
# predictions = torch.zeros(1, 3, 5, 5)
# targets = torch.zeros(1, 5, 5, dtype=torch.int)

#TEST 3
# # Create example predictions (logits)
# # Shape: (batch_size, num_classes, height, width)
# predictions = torch.randn(batch_size, num_classes, height, width)
# # Create example targets
# # Shape: (batch_size, height, width)
# # Values: random integers between 0 and num_classes-1 (e.g., background=0, class1=1, class2=2)
# targets = torch.randint(0, num_classes, (batch_size, height, width))
# # Introduce some ignore pixels (value 255)
# ignore_value = 255
# targets[0, 0:10, 0:10] = ignore_value

# PART TO RUN TESTS
# print(predictions.shape)
# print(targets.shape)
# metrics = SegMetrics(classes=classes)
# metrics.reset()
# metrics.update(predictions, targets)
# print(metrics)



