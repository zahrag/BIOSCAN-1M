import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# global color for the bbox.
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def visualize_predictions(image, outputs, id2label, threshold=0.5):
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    probas_ = probas.max(-1).values
    arg_max = probas_.argmax()
    probas_ = F.one_hot(arg_max, num_classes=len(probas_))
    keep = probas_ > threshold
    # Rename the probas and probas_

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

    # plot results
    plot_results(image, probas[keep], bboxes_scaled, id2label)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def plot_results(pil_img, prob, boxes, id2label):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def get_bbox_from_output(pred, image):
    """
    Extract bounding boxes from the model's output.
    Note that this function will keep the bounding box with the highest confidence and discard others.
    """

    probas = pred.logits.softmax(-1)[0, :, :-1]
    probas_ = probas.max(-1).values
    arg_max = probas_.argmax()
    probas_ = F.one_hot(arg_max, num_classes=len(probas_))
    keep = probas_ > 0.5
    bboxes_scaled = rescale_bboxes(pred.pred_boxes[0, keep].cpu(), image.size)
    return bboxes_scaled[0]

def get_bbox_from_output_for_batch_version(pred_logits, pred_pred_boxes, image_size):
    """
    Extract bounding boxes from the model's output.
    Note that this function will keep the bounding box with the highest confidence and discard others.
    """

    probas = pred_logits.softmax(-1)[0, :, :-1]
    probas_ = probas.max(-1).values
    arg_max = probas_.argmax()
    probas_ = F.one_hot(arg_max, num_classes=len(probas_))
    keep = probas_ > 0.5
    bboxes_scaled = rescale_bboxes(pred_pred_boxes[0, keep].cpu(), image_size)
    # TODO rename probas and probas_
    return bboxes_scaled[0]


def scale_bbox(args, left, top, right, bottom):
    """
    Scale the bounding box based on args.crop_ratio.
    """
    x_range = right - left
    y_range = bottom - top

    if args.equal_extend:
        x_change = y_change = (args.crop_ratio - 1)  * max(x_range, y_range)

    else:
        x_change = x_range * args.crop_ratio - x_range
        y_change = y_range * args.crop_ratio - y_range

    left = int(left - x_change / 2)
    right = int(right + x_change / 2)
    top = int(top - y_change / 2)
    bottom = int(bottom + y_change / 2)

    return left, top, right, bottom


def convert_to_xywh(boxes):
    """
    :param boxes: Bounding boxes in form x_min, y_min, x_max, z_max
    :return: bounding boxes that store in torch tensor in form x_min, y_min, width and height.
    """
    x_min, y_min, x_max, y_max = boxes.unbind(1)
    return torch.stack((x_min, y_min, x_max - x_min, y_max - y_min), dim=1)
