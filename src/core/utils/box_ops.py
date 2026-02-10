import torch


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def bbox_xyxy_to_cxcyah(bbox_xyxy: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, a, h)."""
    x1, y1, x2, y2 = bbox_xyxy[..., 0], bbox_xyxy[..., 1], bbox_xyxy[..., 2], bbox_xyxy[..., 3]
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    a = w / h
    return torch.stack((cx, cy, a, h), dim=-1)


def bbox_cxcyah_to_xyxy(bbox_cxcyah: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, a, h) to (x1, y1, x2, y2)."""
    cx, cy, a, h = (
        bbox_cxcyah[..., 0],
        bbox_cxcyah[..., 1],
        bbox_cxcyah[..., 2],
        bbox_cxcyah[..., 3],
    )
    w = (a * h).clamp(min=0)
    half_w = w * 0.5
    half_h = h * 0.5
    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h
    return torch.stack((x1, y1, x2, y2), dim=-1)


def bbox_overlaps(
    bboxes1: torch.Tensor,
    bboxes2: torch.Tensor,
    mode: str = "iou",
    is_aligned: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute IoU/IoF/GIoU between two sets of boxes in xyxy format."""
    if mode not in {"iou", "iof", "giou"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if bboxes1.numel() == 0:
        if is_aligned:
            return bboxes1.new_zeros((bboxes1.shape[0],))
        return bboxes1.new_zeros((bboxes1.shape[0], bboxes2.shape[0]))
    if bboxes2.numel() == 0:
        if is_aligned:
            return bboxes1.new_zeros((bboxes1.shape[0],))
        return bboxes1.new_zeros((bboxes1.shape[0], bboxes2.shape[0]))

    bboxes1 = bboxes1[..., :4].to(dtype=torch.float32)
    bboxes2 = bboxes2[..., :4].to(dtype=torch.float32)
    area1 = _box_area(bboxes1)
    area2 = _box_area(bboxes2)

    if is_aligned:
        if bboxes1.shape[0] != bboxes2.shape[0]:
            raise ValueError("Aligned overlaps require same number of boxes.")
        lt = torch.maximum(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.minimum(bboxes1[:, 2:], bboxes2[:, 2:])
        inter_wh = (rb - lt).clamp(min=0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]

        if mode == "iof":
            denom = area1.clamp(min=eps)
        else:
            union = area1 + area2 - inter
            denom = union.clamp(min=eps)
        iou = inter / denom

        if mode != "giou":
            return iou

        encl_lt = torch.minimum(bboxes1[:, :2], bboxes2[:, :2])
        encl_rb = torch.maximum(bboxes1[:, 2:], bboxes2[:, 2:])
        encl_wh = (encl_rb - encl_lt).clamp(min=0)
        encl_area = (encl_wh[:, 0] * encl_wh[:, 1]).clamp(min=eps)
        return iou - (encl_area - union) / encl_area

    lt = torch.maximum(bboxes1[:, None, :2], bboxes2[None, :, :2])
    rb = torch.minimum(bboxes1[:, None, 2:], bboxes2[None, :, 2:])
    inter_wh = (rb - lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    if mode == "iof":
        denom = area1[:, None].clamp(min=eps)
    else:
        union = area1[:, None] + area2[None, :] - inter
        denom = union.clamp(min=eps)
    iou = inter / denom

    if mode != "giou":
        return iou

    encl_lt = torch.minimum(bboxes1[:, None, :2], bboxes2[None, :, :2])
    encl_rb = torch.maximum(bboxes1[:, None, 2:], bboxes2[None, :, 2:])
    encl_wh = (encl_rb - encl_lt).clamp(min=0)
    encl_area = (encl_wh[..., 0] * encl_wh[..., 1]).clamp(min=eps)
    return iou - (encl_area - union) / encl_area
