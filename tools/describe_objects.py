# path: tools/describe_objects.py
"""
Identifica objetos y los describe; salida en EN o ES.
Modelos:
- Detección: DETR (normal) o YOLOS-Tiny (lite)
- Caption: BLIP large/base
- Traducción EN→ES: Helsinki-NLP/opus-mt-en-es (opcional)

Uso:
  python tools/describe_objects.py --input img.jpg --out out --lite --cpu --max-side 1280 --lang es
"""
from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch
from transformers import pipeline

# Mapeo rápido de etiquetas comunes a ES (fallback si traductor falla)
COCO_ES_MAP = {
    "person": "persona","bicycle": "bicicleta","car": "auto","motorcycle": "motocicleta",
    "airplane": "avión","bus": "bus","train": "tren","truck": "camión","boat": "barco",
    "traffic light": "semáforo","fire hydrant": "boca de incendio","stop sign": "señal de alto",
    "bench": "banco","bird": "pájaro","cat": "gato","dog": "perro","horse": "caballo",
    "sheep": "oveja","cow": "vaca","elephant": "elefante","bear": "oso","zebra": "cebra",
    "giraffe": "jirafa","backpack": "mochila","umbrella": "paraguas","handbag": "cartera",
    "tie": "corbata","suitcase": "maleta","frisbee": "frisbee","skis": "esquís","snowboard": "snowboard",
    "sports ball": "pelota","kite": "cometa","baseball bat": "bate de béisbol",
    "baseball glove": "guante de béisbol","skateboard": "patineta","surfboard": "tabla de surf",
    "tennis racket": "raqueta de tenis","bottle": "botella","wine glass": "copa de vino",
    "cup": "taza","fork": "tenedor","knife": "cuchillo","spoon": "cuchara","bowl": "cuenco",
    "banana": "banana","apple": "manzana","sandwich": "sándwich","orange": "naranja",
    "broccoli": "brócoli","carrot": "zanahoria","hot dog": "hot dog","pizza": "pizza",
    "donut": "donut","cake": "torta","chair": "silla","couch": "sofá",
    "potted plant": "planta en maceta","bed": "cama","dining table": "mesa",
    "toilet": "inodoro","tv": "televisor","laptop": "laptop","mouse": "mouse",
    "remote": "control remoto","keyboard": "teclado","cell phone": "celular",
    "microwave": "microondas","oven": "horno","toaster": "tostadora","sink": "fregadero",
    "refrigerator": "refrigerador","book": "libro","clock": "reloj","vase": "jarrón",
    "scissors": "tijeras","teddy bear": "oso de peluche","hair drier": "secador",
    "toothbrush": "cepillo de dientes","table": "mesa"
}

@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int
    def as_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)
    def clip(self, w: int, h: int) -> "BBox":
        return BBox(
            x1=max(0, min(self.x1, w - 1)),
            y1=max(0, min(self.y1, h - 1)),
            x2=max(0, min(self.x2, w - 1)),
            y2=max(0, min(self.y2, h - 1)),
        )

@dataclass
class DetectedObject:
    label: str; score: float; bbox: BBox; caption: Optional[str] = None

@dataclass
class ImageResult:
    file: str; width: int; height: int; scene_caption: str; objects: List[DetectedObject]

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def imread_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def imwrite_bgr(path: str, img_rgb: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    cv2.imwrite(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

def list_images(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if os.path.splitext(f.lower())[1] in exts]

def resize_max_side(img: np.ndarray, max_side: Optional[int]) -> Tuple[np.ndarray, float]:
    if not max_side: return img, 1.0
    h, w = img.shape[:2]; s = max(h, w)
    if s <= max_side: return img, 1.0
    scale = max_side / float(s); nh, nw = int(round(h*scale)), int(round(w*scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), scale

def load_pipelines(use_gpu: bool, lite: bool, lang: str):
    device = 0 if (use_gpu and torch.cuda.is_available()) else -1
    det_model = "hustvl/yolos-tiny" if lite else "facebook/detr-resnet-50"
    cap_model = "Salesforce/blip-image-captioning-base" if lite else "Salesforce/blip-image-captioning-large"
    det = pipeline("object-detection", model=det_model, device=device)
    cap = pipeline("image-to-text", model=cap_model, device=device,
                   torch_dtype=torch.float16 if device == 0 else None,
                   generate_kwargs={"max_new_tokens": 32})
    translator = None
    if lang.lower() == "es":
        trans = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=device)
        def translator(text: str) -> str:
            if not text: return text
            try:
                out = trans(text)
                if isinstance(out, list) and out:
                    return str(out[0].get("translation_text", "")).strip() or text
            except Exception:
                return text
            return text
    return det, cap, translator

def translate_text(text: str, translator) -> str:
    if not translator or not text: return text
    return translator(text)

def run_detection(detector, image_rgb: np.ndarray, score_thr: float, translator=None) -> List[DetectedObject]:
    pil = Image.fromarray(image_rgb)
    raw = detector(pil)
    h, w = image_rgb.shape[:2]
    out: List[DetectedObject] = []
    for r in raw:
        score = float(r.get("score", 0.0))
        if score < score_thr: continue
        box = r["box"]
        label_en = r.get("label", "object")
        label = COCO_ES_MAP.get(label_en, label_en)
        if translator and label == label_en:
            label = translate_text(label_en, translator)
        bbox = BBox(int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])).clip(w, h)
        out.append(DetectedObject(label=label, score=score, bbox=bbox))
    return out

def crop_with_padding(img: np.ndarray, bbox: BBox, pad: float = 0.05) -> np.ndarray:
    h, w = img.shape[:2]
    dx = int((bbox.x2 - bbox.x1) * pad); dy = int((bbox.y2 - bbox.y1) * pad)
    bb = BBox(bbox.x1 - dx, bbox.y1 - dy, bbox.x2 + dx, bbox.y2 + dy).clip(w, h)
    return img[bb.y1:bb.y2, bb.x1:bb.x2]

def describe_image(captioner, img_rgb: np.ndarray, translator=None) -> str:
    pil = Image.fromarray(img_rgb)
    out = captioner(pil)
    text = ""
    if isinstance(out, list) and out:
        text = str(out[0].get("generated_text", "")).strip()
    return translate_text(text, translator) if translator else text

def annotate(image_rgb: np.ndarray, objects: List[DetectedObject]) -> np.ndarray:
    vis = image_rgb.copy()
    for i, o in enumerate(objects, start=1):
        x1, y1, x2, y2 = o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        tag = f"{i}:{o.label} {o.score:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(vis, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis

def process_image(path: str, detector, captioner, score_thr: float, max_side: Optional[int], translator=None) -> ImageResult:
    img0 = imread_rgb(path)
    img, _ = resize_max_side(img0, max_side)
    h, w = img.shape[:2]
    objs = run_detection(detector, img, score_thr, translator=translator)
    scene_caption = describe_image(captioner, img, translator=translator)
    for o in objs:
        crop = crop_with_padding(img, o.bbox)
        o.caption = describe_image(captioner, crop, translator=translator)
    return ImageResult(file=os.path.basename(path), width=w, height=h, scene_caption=scene_caption, objects=objs)

def save_results(img_rgb: np.ndarray, res: ImageResult, out_dir: str) -> None:
    ensure_dir(out_dir)
    base = os.path.splitext(res.file)[0]
    annotated = annotate(img_rgb, res.objects)
    imwrite_bgr(os.path.join(out_dir, f"{base}_annotated.jpg"), annotated)
    payload = {
        "file": res.file,
        "width": res.width,
        "height": res.height,
        "scene_caption": res.scene_caption,
        "objects": [
            {
                "id": i + 1,
                "label": o.label,
                "score": o.score,
                "bbox": {"x1": o.bbox.x1, "y1": o.bbox.y1, "x2": o.bbox.x2, "y2": o.bbox.y2},
                "caption": o.caption,
            } for i, o in enumerate(res.objects)
        ],
    }
    with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def sample_video_frames(video_path: str, target_fps: float) -> List[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / max(0.001, target_fps))))
    frames = []; idx = 0; frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        if idx % step == 0:
            frames.append((frame_idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
            frame_idx += 1
        idx += 1
    cap.release()
    return frames

def process_video(video_path: str, detector, captioner, score_thr: float, fps: float, max_side: Optional[int], out_dir: str, translator=None) -> None:
    ensure_dir(out_dir)
    frames = sample_video_frames(video_path, fps)
    rows = []
    for fid, rgb0 in tqdm(frames, desc="Procesando frames"):
        rgb, _ = resize_max_side(rgb0, max_side)
        res = ImageResult(
            file=f"frame_{fid:06d}.jpg",
            width=rgb.shape[1], height=rgb.shape[0],
            scene_caption=describe_image(captioner, rgb, translator=translator),
            objects=run_detection(detector, rgb, score_thr, translator=translator),
        )
        for o in res.objects:
            o.caption = describe_image(captioner, crop_with_padding(rgb, o.bbox), translator=translator)
        save_results(rgb, res, out_dir)
        for j, o in enumerate(res.objects, start=1):
            rows.append({
                "frame": fid, "id": j, "label": o.label, "score": o.score,
                "x1": o.bbox.x1, "y1": o.bbox.y1, "x2": o.bbox.x2, "y2": o.bbox.y2,
                "object_caption": o.caption, "scene_caption": res.scene_caption,
            })
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False, encoding="utf-8")

def process_webcam(detector, captioner, score_thr: float, fps: float, max_side: Optional[int], out_dir: str, translator=None) -> None:
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la webcam (índice 0).")
    interval_ms = int(1000 / max(0.001, fps))
    frame_id = 0; last_save_id = -999
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        rgb0 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb, _ = resize_max_side(rgb0, max_side)
        objs = run_detection(detector, rgb, score_thr, translator=translator)
        scene_caption = describe_image(captioner, rgb, translator=translator) if (frame_id % 10 == 0) else ""
        vis = annotate(rgb, objs)
        cv2.imshow("Webcam - q para salir", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if frame_id - last_save_id >= int(max(1, fps)):
            base = f"cam_{frame_id:06d}"
            imwrite_bgr(os.path.join(out_dir, f"{base}_annotated.jpg"), vis)
            payload = {
                "file": f"{base}.jpg", "width": vis.shape[1], "height": vis.shape[0],
                "scene_caption": scene_caption,
                "objects": [
                    {"id": i + 1, "label": o.label, "score": o.score,
                     "bbox": {"x1": o.bbox.x1, "y1": o.bbox.y1, "x2": o.bbox.x2, "y2": o.bbox.y2},
                     "caption": None}
                    for i, o in enumerate(objs)
                ],
            }
            with open(os.path.join(out_dir, f"{base}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            last_save_id = frame_id
        key = cv2.waitKey(interval_ms) & 0xFF
        if key == ord('q'): break
        frame_id += 1
    cap.release(); cv2.destroyAllWindows()

def main():
    p = argparse.ArgumentParser(description="Identificar objetos y describirlos (DETR/YOLOS + BLIP) con salida en EN/ES.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Ruta a imagen o carpeta.")
    src.add_argument("--video", type=str, help="Ruta a video.")
    src.add_argument("--webcam", action="store_true", help="Usar webcam.")
    p.add_argument("--out", type=str, required=True, help="Carpeta de salida.")
    p.add_argument("--score", type=float, default=0.5, help="Umbral de confianza.")
    p.add_argument("--fps", type=float, default=1.0, help="FPS para video/webcam.")
    p.add_argument("--max-side", type=int, default=None, help="Redimensionar manteniendo aspecto al máximo lado.")
    p.add_argument("--cpu", action="store_true", help="Forzar CPU.")
    p.add_argument("--lite", action="store_true", help="Modelos livianos (YOLOS-Tiny + BLIP-base).")
    p.add_argument("--lang", type=str, choices=["en","es"], default="en", help="Idioma de salida (captions/labels).")
    args = p.parse_args()

    detector, captioner, translator = load_pipelines(use_gpu=not args.cpu, lite=args.lite, lang=args.lang)

    if args.input:
        if os.path.isdir(args.input):
            paths = list_images(args.input)
            if not paths: raise FileNotFoundError("No se encontraron imágenes en la carpeta.")
        else:
            if not os.path.exists(args.input): raise FileNotFoundError(f"No existe: {args.input}")
            paths = [args.input]
        ensure_dir(args.out)
        for path in tqdm(paths, desc="Procesando imágenes"):
            img0 = imread_rgb(path)
            img, _ = resize_max_side(img0, args.max_side)
            res = process_image(path, detector, captioner, args.score, args.max_side, translator=translator)
            save_results(img, res, args.out)
    elif args.video:
        process_video(args.video, detector, captioner, args.score, args.fps, args.max_side, args.out, translator=translator)
    else:
        process_webcam(detector, captioner, args.score, args.fps, args.max_side, args.out, translator=translator)

if __name__ == "__main__":
    main()
