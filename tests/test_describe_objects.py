# =========================
# path: tests/test_describe_objects.py
# =========================
import numpy as np
from tools.describe_objects import BBox, crop_with_padding, process_image

class MockDetector:
    def __call__(self, pil_img):
        w, h = pil_img.size
        return [
            {"score": 0.9, "label": "cat", "box": {"xmin": 5, "ymin": 5, "xmax": max(6, w//2), "ymax": max(6, h//2)}},
            {"score": 0.3, "label": "low", "box": {"xmin": 0, "ymin": 0, "xmax": w-1, "ymax": 2}},
        ]

class MockCaptioner:
    def __call__(self, pil_img):
        return [{"generated_text": f"mock caption {pil_img.size}"}]

def test_bbox_clip_and_xywh():
    b = BBox(-10, -5, 110, 105).clip(100, 100)
    assert b.x1 == 0 and b.y1 == 0 and b.x2 == 99 and b.y2 == 99
    assert b.as_xywh() == (0, 0, 99, 99)

def test_crop_padding_bounds():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    bb = BBox(2, 2, 5, 5)
    crop = crop_with_padding(img, bb, pad=0.5)
    assert crop.shape[0] > 0 and crop.shape[1] > 0

def test_process_image_with_mocks(tmp_path):
    from PIL import Image
    p = tmp_path / "toy.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(p)
    res = process_image(str(p), MockDetector(), MockCaptioner(), score_thr=0.5, max_side=None)
    assert res.scene_caption.startswith("mock caption")
    assert len(res.objects) == 1
    assert res.objects[0].label == "cat"
