import cv2
import numpy as np
from PIL import Image, ExifTags
import argparse
import sys
from pathlib import Path



CORNER_RADIUS = 8 # rayon des cercles de coins (px, sur l'image affichée)
CORNER_COLOR = (0, 255, 0)
ACTIVE_COLOR = (0, 120, 255)
LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 2
DRAG_THRESHOLD = 20 # distance max (px) pour saisir un coin au clic



#  1. CORRECTION D'ORIENTATION EXIF


def correct_exif_orientation(img_bgr: np.ndarray, path: str) -> np.ndarray:
    try:
        tag_id = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
        pil = Image.open(path)
        exif = pil._getexif()
        if exif:
            rotations = {3: Image.ROTATE_180, 6: Image.ROTATE_270, 8: Image.ROTATE_90}
            rot = rotations.get(exif.get(tag_id, 1))
            if rot:
                pil = pil.transpose(rot)
                img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                print("[INFO] Orientation EXIF corrigée.")
    except Exception:
        pass
    return img_bgr



#  2. PRÉTRAITEMENT SELON LE MODE


# Document / feuille de papier
def preprocess_document(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Seuils automatiques (médiane ×0.5 / ×1.5)
    median = np.median(blurred)
    low = int(max(0,   0.5 * median))
    high = int(min(255, 1.5 * median))
    edges = cv2.Canny(blurred, low, high)
    # Dilatation pour fermer les petits trous
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


# Écran (moniteur, tablette, téléphone)
def preprocess_screen(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Égalisation locale pour atténuer les reflets
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blurred, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.dilate(edges, kernel, iterations=2)
    return edges



#  3. DÉTECTION AUTOMATIQUE DU QUADRILATÈRE


# Réordonne 4 points dans l'ordre : haut-gauche, haut-droit, bas-droit, bas-gauche (sens indirect).
def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.array([
        pts[np.argmin(s)], # haut-gauche  (x+y minimal)
        pts[np.argmin(diff)], # haut-droit   (y-x minimal)
        pts[np.argmax(s)], # bas-droit    (x+y maximal)
        pts[np.argmax(diff)], # bas-gauche   (y-x maximal)
    ], dtype=np.float32)
    return ordered


def detect_quad(edges: np.ndarray, img_shape: tuple) -> np.ndarray | None:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = img_shape[:2]
    img_area = w * h
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.05:   # trop petit -> ignoré
            continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area = area
                best = approx

    if best is None:
        return None

    return order_corners(best)


def fallback_corners(img_shape: tuple) -> np.ndarray:
    h, w = img_shape[:2]
    mx, my = int(w * 0.1), int(h * 0.1)
    return np.array([
        [mx, my ],
        [w - mx, my ],
        [w - mx, h - my],
        [mx, h - my],
    ], dtype=np.float32)



#  4. INTERFACE DE CORRECTION MANUELLE


# Fenêtre OpenCV interactive
class CornerEditor:

    def __init__(self, img_bgr: np.ndarray, corners: np.ndarray,
                 auto_corners: np.ndarray, max_display: int = 900):
        self.original = img_bgr.copy()
        self.auto_corners = auto_corners.copy()
        self.corners = corners.copy()
        self.active_idx = -1
        self.validated = False
        self.cancelled = False

        # Mise à l'échelle pour l'affichage (image haute-résolution -> fenêtre raisonnable)
        h, w = img_bgr.shape[:2]
        self.scale = min(1.0, max_display / max(h, w))
        self.disp_w = int(w * self.scale)
        self.disp_h = int(h * self.scale)

    def _to_display(self, pts: np.ndarray) -> np.ndarray:
        return (pts * self.scale).astype(int)

    def _from_display(self, x: int, y: int):
        return np.array([x / self.scale, y / self.scale], dtype=np.float32)

    def _draw(self) -> np.ndarray:
        disp = cv2.resize(self.original, (self.disp_w, self.disp_h))
        pts  = self._to_display(self.corners)

        # Quadrilatère
        cv2.polylines(disp, [pts.reshape(-1, 1, 2)], True, LINE_COLOR, LINE_THICKNESS)

        # Coins
        for i, (x, y) in enumerate(pts):
            color = ACTIVE_COLOR if i == self.active_idx else CORNER_COLOR
            cv2.circle(disp, (x, y), CORNER_RADIUS, color, -1)
            cv2.circle(disp, (x, y), CORNER_RADIUS, (255, 255, 255), 1)
            label = ["HG", "HD", "BD", "BG"][i]
            cv2.putText(disp, label, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Instructions
        instructions = [
            "Glisser : deplacer un coin",
            "Entree : valider",
            "R : reinitialiser",
            "Q/Echap : quitter",
        ]
        for i, txt in enumerate(instructions):
            cv2.putText(disp, txt, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return disp

    def _nearest_corner(self, x: int, y: int) -> int:
        pts  = self._to_display(self.corners)
        dists = np.linalg.norm(pts - np.array([x, y]), axis=1)
        idx  = int(np.argmin(dists))
        return idx if dists[idx] < DRAG_THRESHOLD else -1

    def _mouse_callback(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.active_idx = self._nearest_corner(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.active_idx >= 0:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self.corners[self.active_idx] = self._from_display(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.active_idx = -1

    def run(self) -> np.ndarray | None:
        win = "Recadrage 9:16 – correction des coins"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.disp_w, self.disp_h)
        cv2.setMouseCallback(win, self._mouse_callback)

        print("\n[Interface] Vérifiez les coins détectés.")
        print("  Glisser -> déplacer un coin")
        print("  Entrée -> valider et recadrer")
        print("  R -> réinitialiser")
        print("  Q / Échap -> quitter\n")

        while True:
            cv2.imshow(win, self._draw())
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 10):  # Entrée
                self.validated = True
                break
            elif key == ord('r'):  # Réinitialiser
                self.corners = self.auto_corners.copy()
                print("[INFO] Coins réinitialisés.")
            elif key in (ord('q'), 27):  # Q ou Échap
                self.cancelled = True
                break

        cv2.destroyAllWindows()
        return None if self.cancelled else self.corners.copy()



#  5. HOMOGRAPHIE & WARP


def build_destination_corners(target_w: int, target_h: int) -> np.ndarray:
    return np.array([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h],
    ], dtype=np.float32)


#  Calcul de l'homographie depuis 4 pts source vers 4 pts destination et application de la transformation de perspective.
def warp_image(img_bgr: np.ndarray, src_corners: np.ndarray,
               target_w: int, target_h: int) -> np.ndarray:
    dst_corners = build_destination_corners(target_w, target_h)
    H = cv2.getPerspectiveTransform(src_corners, dst_corners)
    print(f"[INFO] Matrice d'homographie :\n{np.round(H, 4)}")
    warped = cv2.warpPerspective(img_bgr, H, (target_w, target_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped



#  6. PIPELINE PRINCIPAL


# Fonction principale
def recadrer(source_path: str, output_path: str, mode: str,
             target_w: int, target_h: int, auto_only: bool):

    print("\n══════════════════════════════════════════")
    print(f"  Recadrage 9:16 par homographie – mode : {mode}")
    print("══════════════════════════════════════════\n")

    # Chargement
    img = cv2.imread(source_path)
    if img is None:
        sys.exit(f"[ERREUR] Impossible de charger : {source_path}")
    print(f"[INFO] Image chargée : {img.shape[1]}×{img.shape[0]} px")

    # Correction EXIF
    img = correct_exif_orientation(img, source_path)

    # Prétraitement
    if mode == "document":
        edges = preprocess_document(img)
    else:
        edges = preprocess_screen(img)

    # Détection automatique
    auto_corners = detect_quad(edges, img.shape)
    if auto_corners is None:
        print("[AVERT] Aucun quadrilatère détecté -> coins par défaut utilisés.")
        auto_corners = fallback_corners(img.shape)
    else:
        print("[INFO] Quadrilatère détecté automatiquement.")

    # Correction manuelle (sauf si --auto)
    if auto_only:
        final_corners = auto_corners
    else:
        editor = CornerEditor(img, auto_corners.copy(), auto_corners.copy())
        final_corners = editor.run()
        if final_corners is None:
            print("[INFO] Annulé par l'utilisateur.")
            sys.exit(0)

    print(f"[INFO] Coins finaux :\n{np.round(final_corners, 1)}")

    # Homographie + warp
    result = warp_image(img, final_corners, target_w, target_h)

    # Sauvegarde
    out = Path(output_path)
    cv2.imwrite(str(out), result)
    print(f"\n[OK] Image recadrée sauvegardée : {out.resolve()}")
    print(f"     Résolution : {target_w}×{target_h} px\n")




#  Interface en Ligne de Commande(CLI)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recadrage intelligent au format 9:16 par homographie.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source",
                        help="Chemin de l'image source.")
    parser.add_argument("-m", "--mode",
                        choices=["document", "ecran"],
                        default="document",
                        help="Profil de détection : 'document' (papier) ou 'ecran'.")
    parser.add_argument("-o", "--output",
                        default="recadre_9_16.jpg",
                        help="Chemin de l'image de sortie.")
    parser.add_argument("--width",  type=int, default=1080,
                        help="Largeur cible (px).")
    parser.add_argument("--height", type=int, default=1920,
                        help="Hauteur cible (px).")
    parser.add_argument("--auto",   action="store_true",
                        help="Mode entièrement automatique, sans interface de correction.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recadrer(
        source_path=args.source,
        output_path=args.output,
        mode=args.mode,
        target_w=args.width,
        target_h=args.height,
        auto_only=args.auto,
    )
