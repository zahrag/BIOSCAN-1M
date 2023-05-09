from util.visualize_and_process_bbox import convert_to_xywh


def prepare_for_evaluation(predictions):
    """
    Convert model's output to a format that is ready for coco_evaluator to evaluate.
    :param predictions: Driect output from the model.
    :return: List of dictionary that contains:image_id, category_id, bbox, score of the bounding box.
    """
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            exit(1)
        #     TODO handle the special case.
        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()
        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results
