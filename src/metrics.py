"""Single-class, all-points interpolated AP at IoU 0.5 (not COCO AP)."""

from src.geometry import box_iou


def evaluate_predictions(predictions, targets, operating_threshold=0.5):
    if set(predictions) != set(targets):
        raise ValueError("Predictions and references must cover the same image IDs")
    total_targets = sum(len(boxes) for boxes in targets.values())
    ranked = []
    for image_id, (boxes, scores) in predictions.items():
        for box, score in zip(boxes, scores):
            ranked.append((float(score), image_id, box))
    ranked.sort(key=lambda detection: detection[0], reverse=True)
    matched = {image_id: set() for image_id in targets}
    true_positive, false_positive, scores, precision, recall = [], [], [], [], []
    accumulated_true = 0
    for index, (score, image_id, box) in enumerate(ranked):
        references = targets[image_id]
        correct = False
        if len(references):
            overlaps = box_iou(box[None, :], references)[0]
            for already_matched in matched[image_id]:
                overlaps[already_matched] = -1
            overlap, target_index = overlaps.max(dim=0)
            if float(overlap) >= 0.5:
                matched[image_id].add(int(target_index))
                correct = True
        true_positive.append(int(correct))
        false_positive.append(int(not correct))
        accumulated_true += int(correct)
        scores.append(score)
        precision.append(accumulated_true / (index + 1))
        recall.append(accumulated_true / max(total_targets, 1))
    # Integrate the monotone precision envelope at recall changes.
    recall_edges = [0.] + recall + [1.]
    envelope = [0.] + precision + [0.]
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])
    average_precision = sum((recall_edges[index] - recall_edges[index - 1]) * envelope[index]
                            for index in range(1, len(recall_edges)))
    kept = [index for index, score in enumerate(scores) if score >= operating_threshold]
    correct_count = sum(true_positive[index] for index in kept)
    false_count = sum(false_positive[index] for index in kept)
    result = {"ap50": average_precision if total_targets else None,
              "ground_truth_boxes": total_targets, "detections": len(kept),
              "true_positives": correct_count, "false_positives": false_count,
              "missed_targets": total_targets - correct_count,
              "precision": correct_count / max(len(kept), 1),
              "recall": correct_count / max(total_targets, 1),
              "operating_threshold": operating_threshold,
              "pr_curve": {"scores": scores, "precision": precision, "recall": recall}}
    return result


def choose_threshold(predictions, targets):
    best_threshold, best_f1 = 0.5, -1.
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        result = evaluate_predictions(predictions, targets, threshold)
        precision, recall = result["precision"], result["recall"]
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return best_threshold