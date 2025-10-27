# Task 2

## Objective: 
Processing multiple images in a single program for object detection and segmentation, with the corresponding performance metrics available in the runs/detect/train folder and explaining them

## Object Detection:
The metrics obtained from the the evaluation of the model are: 
- **BoxP Curve (Precision Curve)**: This measures how many of the detections were correct. Higher precision means fewer false positives. The  curve shows how the precision changes as the confidence threshold changes. If it drops sharply at low confidence threshold, that means the model is hallucinating objects when its less sure. 
- **BoxR Curve (Recall Curve)**: This measures how many of the true objects were detected. High recalls means fewer missed objects/few overlookings. A low recall curve means the models fails to detect some real objects. 
- **BoxPR Curve (Precision-Recall Curve)**: The area under this curve gives average precision. A good model's PR curve hugs top right i.e high P and high R. If it's bowed inward or erratic, the detections are inconsistent. 
- **Box F1 Curve**: It combines precision and recall in form of harmonic mean. It peaks where the model best balances between missing and misclassifying detections. The maximum F1 value gives the sweet spot for confidence deployment. 
- **Confusion Matrix**: High Diagonal values means good performance. 


