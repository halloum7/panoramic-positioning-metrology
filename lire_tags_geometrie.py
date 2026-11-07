"""
lire_tags_geometrie.py — A executer LOCALEMENT sur votre PC.
Lit UNIQUEMENT les tags geometriques/techniques des DICOM panoramiques
(aucun nom, aucune date, aucun identifiant n'est lu ni ecrit).

Usage :
    python lire_tags_geometrie.py "C:\\Users\\HP\\Desktop\\panoramique data" tags_geometrie.csv

Si pydicom est installe (pip install pydicom), il est utilise ;
sinon, un lecteur minimal (Explicit VR Little Endian) prend le relais.
"""
import csv
import os
import struct
import sys

# (groupe, element) -> nom lisible. Liste blanche stricte.
TAGS = {
    (0x0028, 0x0010): "Rows",
    (0x0028, 0x0011): "Columns",
    (0x0028, 0x0030): "PixelSpacing",
    (0x0018, 0x1164): "ImagerPixelSpacing",
    (0x0018, 0x1114): "EstimatedRadiographicMagnificationFactor",
    (0x0028, 0x0A02): "PixelSpacingCalibrationType",
    (0x0018, 0x1110): "DistanceSourceToDetector",
    (0x0018, 0x1111): "DistanceSourceToPatient",
    (0x0018, 0x0060): "KVP",
    (0x0018, 0x1151): "XRayTubeCurrent",
    (0x0018, 0x1150): "ExposureTime",
    (0x0018, 0x115E): "ImageAndFluoroscopyAreaDoseProduct",
    (0x0008, 0x0068): "PresentationIntentType",
    (0x0008, 0x0070): "Manufacturer",
    (0x0008, 0x1090): "ManufacturerModelName",
    (0x0028, 0x0100): "BitsAllocated",
    (0x0028, 0x0101): "BitsStored",
}
LONG_VR = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN", b"UC", b"UR", b"OD", b"OL", b"OV"}


def read_with_pydicom(path):
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    out = {}
    for (g, e), name in TAGS.items():
        if (g, e) in ds:
            out[name] = str(ds[(g, e)].value)
    return out


def read_minimal(path):
    """Parcours sequentiel ; s'arrete aux pixels (7FE0,0010)."""
    out = {}
    with open(path, "rb") as f:
        data = f.read()
    pos = 132 if data[128:132] == b"DICM" else 0
    while pos + 8 <= len(data):
        g, e = struct.unpack_from("<HH", data, pos)
        if (g, e) == (0x7FE0, 0x0010):
            break
        vr = data[pos + 4:pos + 6]
        if vr.isalpha() and vr.isupper():
            if vr in LONG_VR:
                length = struct.unpack_from("<I", data, pos + 8)[0]
                start = pos + 12
            else:
                length = struct.unpack_from("<H", data, pos + 6)[0]
                start = pos + 8
        else:  # implicit VR
            vr = None
            length = struct.unpack_from("<I", data, pos + 4)[0]
            start = pos + 8
        if length == 0xFFFFFFFF:  # longueur indefinie : on abandonne proprement
            break
        if (g, e) in TAGS:
            raw = data[start:start + length]
            if vr == b"US" and length >= 2:
                val = str(struct.unpack_from("<H", raw)[0])
            elif vr in (b"FL",) and length >= 4:
                val = str(struct.unpack_from("<f", raw)[0])
            elif vr in (b"FD",) and length >= 8:
                val = str(struct.unpack_from("<d", raw)[0])
            else:
                val = raw.decode("latin-1", "replace").strip("\x00 ")
            out[TAGS[(g, e)]] = val
        pos = start + length
    return out


def main(folder, out_csv):
    try:
        import pydicom  # noqa: F401
        reader = read_with_pydicom
    except ImportError:
        reader = read_minimal
    rows = []
    for root, _, files in os.walk(folder):
        if "Article6" in root:  # on ne lit que les DICOM d'origine
            continue
        for fn in sorted(files):
            if fn.upper().endswith((".DCM", ".DICOM")) or "." not in fn:
                p = os.path.join(root, fn)
                try:
                    tags = reader(p)
                except Exception as exc:  # fichier non DICOM ou atypique
                    tags = {"erreur": type(exc).__name__}
                tags["fichier_index"] = len(rows) + 1  # pas le nom de fichier
                rows.append(tags)
    cols = ["fichier_index"] + list(TAGS.values()) + ["erreur"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"{len(rows)} fichiers lus -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "tags_geometrie.csv")
