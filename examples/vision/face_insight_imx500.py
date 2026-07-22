from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vision.face_insight_imx500 import Imx500ArcFacePipeline # pyright: ignore[reportMissingImports]
from src.vision.face_detector import Imx500Config
from src.vision.face_insight_pipeline import FaceInsightPipeline


# ==========================
# Exemple d'utilisation
# ==========================

def demo_phase1_cpu(video_src: int = 0) -> None:
    cap = cv2.VideoCapture(video_src)
    pipe = FaceInsightPipeline(det_size=(640, 640))

    # Exemple : enregistrer un visage depuis la première frame non vide
    print("Phase 1: enregistrement d'un visage sur la première frame...")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        try:
            pipe.register_face("user", frame)
            break
        except RuntimeError:
            continue

    print("Reconnaissance en continu...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = pipe.recognize(frame)

        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{f.identity or 'unknown'}"
            if f.similarity is not None:
                label += f" ({f.similarity:.2f})"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Phase 1 - InsightFace CPU", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def demo_phase2_imx500(video_src: int = 0) -> None:
    """
    Phase 2 : IMX500 pour la détection, ArcFace pour l'ID.
    On suppose :
    - un pipeline caméra AI Camera qui tourne,
    - et en parallèle on récupère la frame RGB via la caméra classique (ou une API IMX500 si dispo).
    """
    from src.utils.config import load_config
    sys_cfg = load_config()
    cfg = Imx500Config(cmd=sys_cfg.vision.camera.imx500_cmd)
    pipe = Imx500ArcFacePipeline(cfg)

    cap = cv2.VideoCapture(video_src)

    # Enregistrement manuel : ici on réutilise la même logique que Phase 1
    print("Phase 2: enregistrement d'un visage (crop manuel) ...")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        # Ici on peut utiliser temporairement InsightFace CPU pour découper un visage de référence
        # ou bien utiliser une image déjà recadrée.
        pipe.register_face("user", frame)
        break

    print("Reconnaissance (IMX500 + ArcFace)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = pipe.recognize_from_frame(frame)
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{f.identity or 'unknown'}"
            if f.similarity is not None:
                label += f" ({f.similarity:.2f})"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("Phase 2 - IMX500 + ArcFace", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    pipe.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Choisir la démo à lancer
    # demo_phase1_cpu()
    # demo_phase2_imx500()
    pass
