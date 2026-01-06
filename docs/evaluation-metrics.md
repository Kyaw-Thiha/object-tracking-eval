#  Evaluation Metrics

## DetA (Detection Accuracy)

Detection Accuracy measures the localization quality of correctly matched detections. For each frame, we compute the intersection over union (IoU) between each detection and ground truth bounding box, and then use Hungarian assignment to match detections to ground truth that maximizes the sum of IoU. We discard matches with low IoU below a threshold. We record all DetA measurements across all frames to compute confidence intervals of detection accuracy on the dataset level.

For each frame, given detections \(d\), ground truth boxes \(g\), and the set of matched detection–ground truth pairs \(M\), DetA is:

$$
\text{DetA} = \frac{1}{|M|} \sum_{(d,g) \in M} \text{IoU}(B_d, B_g)
$$

## AssA (Association Accuracy)

Association Accuracy is one of the metrics proposed by “A Higher Order Metric for Evaluating Multi-Object Tracking” [Luiten et al. 2020] to measure the association performance of multi-object tracking. This metric is computed using True Positive Associations (TPA), False Positive Associations (FPA), and False Negative Associations (FNA).

The True Positive Association set collects detection–ground truth pairs representing trajectories that are correctly labeled and whose ID assignments match between predictions and ground truth. Given ground-truth track IDs \(\text{gtID}\), predicted track IDs \(\text{prID}\), and true-positive detection–ground-truth pairs \(TP\), the TPA for a pair \(c \in TP\) is:

$$
TPA(c) = \left\lbrace p \mid p \in TP,\ \text{gtID}(p) = \text{gtID}(c) \wedge \text{prID}(p) = \text{prID}(c) \right\rbrace
$$

The False Positive Association set captures detection–ground truth pairs where predictions describe a trajectory that does not exist in the ground truth. Given false-positive pairs \(FP\), the FPA for \(c\) is:

$$
FPA(c) = \left\lbrace p \mid p \in FP,\ \text{gtID}(p) = \text{gtID}(c) \wedge \text{prID}(p) \neq \text{prID}(c) \right\rbrace \cup \left\lbrace p \mid p \in FP,\ \text{gtID}(p) = \text{gtID}(c) \right\rbrace
$$

The False Negative Association set captures hypothetical detection–ground truth pairs describing tracks missing from the prediction. Given false-negative pairs \(FN\), the FNA for predicted ID \(c\) is:

$$
FNA(c) = \left\lbrace p \mid p \in FN,\ \text{gtID}(p) \neq \text{gtID}(c) \wedge \text{prID}(p) = \text{prID}(c) \right\rbrace \cup \left\lbrace p \mid p \in FN,\ \text{prID}(p) = \text{prID}(c) \right\rbrace
$$

Then the association accuracy is:

$$
\text{AssA} = \frac{1}{|TP|} \sum_{c \in TP} \frac{|TPA(c)|}{|TPA(c)| + |FPA(c)| + |FNA(c)|}
$$

## HOTA

HOTA combines the detection accuracy (DetA) and association accuracy (AssA) to present a balanced view of multi-object tracking performance. It is computed as:

$$
\text{HOTA} = \sqrt{\text{DetA} \cdot \text{AssA}}
$$
