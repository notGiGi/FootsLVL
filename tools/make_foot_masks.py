# tools/make_foot_masks_v2.py
import os, argparse
import numpy as np
from PIL import Image, ImageDraw

# --- Bézier cúbica ---
def bezier_cubic(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n, endpoint=False)
    u = 1 - t
    pts = (u**3)[:,None]*p0 + (3*u**2*t)[:,None]*p1 + (3*u*t**2)[:,None]*p2 + (t**3)[:,None]*p3
    return pts

# --- Contorno de pie IZQUIERDO (y=0 dedos arriba, y=1 talón abajo) ---
def left_foot_outline(width_scale=1.00, arch_depth=0.22, toe_splay=0.18, heel_round=0.24):
    """
    Parámetros:
      width_scale  -> ancho total (1.00 normal; 1.1 ancho; 0.9 delgado)
      arch_depth   -> cuánta 'mordida' tiene el arco medial (0.18..0.30)
      toe_splay    -> apertura de la línea de los dedos (0.10..0.25)
      heel_round   -> redondez de talón (0.20..0.30)
    Devuelve una polilínea (M,2) en [0..1] cerrada (último = primero).
    """
    # Anclas anatómicas base (izquierdo)
    # x=0 medial, x=1 lateral ; y=0 dedos, y=1 talón
    H = 1.0
    W = 1.0 * width_scale

    # Puntos clave
    hallux_tip   = np.array([0.33*W, 0.05])                  # punta del hallux
    toes_lateral = np.array([0.78*W, 0.10 + 0.02*toe_splay]) # esquina dedos 5º
    lat_mid      = np.array([0.90*W, 0.45])                  # borde lateral medio
    lat_low      = np.array([0.86*W, 0.70])                  # lateral inferior
    heel_lat     = np.array([0.74*W, 0.95])                  # talón lateral
    heel_med     = np.array([0.48*W, 0.98])                  # talón medial
    arch_low     = np.array([0.34*W, 0.78])                  # medial bajo
    arch_peak    = np.array([0.30*W, 0.55 + 0.12*arch_depth])# pico de arco
    mth1         = np.array([0.32*W, 0.28])                  # zona MTH1
    pre_hallux   = np.array([0.31*W, 0.13 + 0.02*toe_splay]) # antes de la punta

    # Controles (afinados a mano para forma orgánica)
    # 1) Línea de dedos: hallux -> toes_lateral
    c1a = hallux_tip + [0.10*W, -0.04]      # suaviza punta
    c1b = toes_lateral + [-0.12*W, 0.00]    # redondea hacia lateral

    # 2) Lateral alto: toes_lateral -> lat_mid
    c2a = toes_lateral + [0.08*W, 0.05]
    c2b = lat_mid      + [0.02*W, 0.06]

    # 3) Lateral bajo: lat_mid -> lat_low
    c3a = lat_mid + [0.02*W, 0.10]
    c3b = lat_low + [0.00*W, 0.10]

    # 4) Talón: lat_low -> heel_lat -> heel_med (arco amplio)
    heel_ctrl = heel_round
    c4a = lat_low   + [-0.06*W, 0.12]
    c4b = heel_lat  + [ 0.00*W, 0.08]
    c5a = heel_lat  + [-0.08*W, 0.06]
    c5b = heel_med  + [ 0.00*W, -0.02]

    # 5) Medial bajo: heel_med -> arch_low
    c6a = heel_med + [-0.06*W, -0.02]
    c6b = arch_low + [ 0.02*W, -0.10]

    # 6) Arco: arch_low -> arch_peak
    c7a = arch_low  + [-0.04*W, -0.10]
    c7b = arch_peak + [ 0.02*W, -0.10]

    # 7) Hacia MTH1: arch_peak -> mth1
    c8a = arch_peak + [ 0.02*W, -0.12]
    c8b = mth1      + [ 0.02*W, -0.08]

    # 8) MTH1 a pre_hallux
    c9a = mth1      + [ 0.02*W, -0.06]
    c9b = pre_hallux+ [ 0.00*W, -0.02]

    # 9) cierre: pre_hallux -> hallux_tip
    c10a= pre_hallux+ [ 0.02*W, -0.04]
    c10b= hallux_tip+ [-0.02*W, -0.02]

    segs = []
    segs.append(bezier_cubic(hallux_tip, c1a,  c1b,  toes_lateral, n=70))
    segs.append(bezier_cubic(toes_lateral, c2a, c2b, lat_mid,      n=50))
    segs.append(bezier_cubic(lat_mid,      c3a, c3b, lat_low,      n=40))
    segs.append(bezier_cubic(lat_low,      c4a, c4b, heel_lat,     n=40))
    segs.append(bezier_cubic(heel_lat,     c5a, c5b, heel_med,     n=40))
    segs.append(bezier_cubic(heel_med,     c6a, c6b, arch_low,     n=40))
    segs.append(bezier_cubic(arch_low,     c7a, c7b, arch_peak,    n=40))
    segs.append(bezier_cubic(arch_peak,    c8a, c8b, mth1,         n=40))
    segs.append(bezier_cubic(mth1,         c9a, c9b, pre_hallux,   n=40))
    segs.append(bezier_cubic(pre_hallux,   c10a,c10b, hallux_tip,  n=40))

    poly = np.vstack(segs)
    # seguridad: recorta a [0,1]
    poly[:,0] = np.clip(poly[:,0], 0.0, 1.0)
    poly[:,1] = np.clip(poly[:,1], 0.0, 1.0)
    # cerrar
    poly = np.vstack([poly, poly[0:1]])
    return poly

def raster_mask(poly01, w, h):
    from PIL import Image, ImageDraw
    poly_px = poly01.copy()
    poly_px[:,0] *= w; poly_px[:,1] *= h
    img = Image.new("L", (w, h), color=0)         # negro = fondo
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(p) for p in poly_px], fill=255)  # blanco = pie
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=360, help="ancho PNG")
    ap.add_argument("--h", type=int, default=640, help="alto PNG")
    ap.add_argument("--dir", type=str, default="assets", help="carpeta destino")
    ap.add_argument("--width", type=float, default=1.00, help="ancho relativo (0.9..1.2)")
    ap.add_argument("--arch", type=float, default=0.22, help="profundidad arco (0.18..0.30)")
    ap.add_argument("--toe", type=float, default=0.18, help="apertura dedos (0.10..0.25)")
    ap.add_argument("--heel", type=float, default=0.24, help="redondez talón (0.20..0.30)")
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    polyL = left_foot_outline(width_scale=args.width, arch_depth=args.arch,
                              toe_splay=args.toe, heel_round=args.heel)
    # espejo para derecho: x -> 1-x
    polyR = polyL.copy(); polyR[:,0] = 1.0 - polyR[:,0]

    imgL = raster_mask(polyL, args.w, args.h)
    imgR = raster_mask(polyR, args.w, args.h)

    pathL = os.path.join(args.dir, "foot_left.png")
    pathR = os.path.join(args.dir, "foot_right.png")
    imgL.save(pathL); imgR.save(pathR)
    print(f"OK: {pathL}  ({args.w}x{args.h})")
    print(f"OK: {pathR}  ({args.w}x{args.h})")

if __name__ == "__main__":
    main()
