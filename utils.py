import re
import faiss
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyproj
from sklearn.decomposition import PCA

def get_ecef_coords(utm_east, utm_north, utm_zone, latitude, longitude):
    epsg = (32600 if latitude > 0 else 32700) + int(utm_zone)
    crs_utm = pyproj.CRS(f"EPSG:{epsg}")
    crs_ecef = pyproj.CRS("EPSG:4978")
    transformer = pyproj.Transformer.from_crs(crs_utm, crs_ecef, always_xy=True)
    x, y, z = transformer.transform(utm_east, utm_north, 0)
    return np.array([x, y, z])

# @ UTM_east @ UTM_north @ UTM_zone_number @ UTM_zone_letter @ latitude @ longitude @ pano_id @ tile_num @ heading @ pitch @ roll @ height @ timestamp @ note @ extension
def parse_image_name(image_name):
    try:
        _, easting, northing, zone_number, zone_letter, latitude, longitude = image_name.split("@")[:7]
        pano_id, tile_num, heading, pitch, roll, height, timestamp, note, extension = image_name.split("@")[7:]
        easting = float(easting)
        northing = float(northing)
        zone_number = int(zone_number)
        zone_letter = zone_letter
        latitude = float(latitude)
        longitude = float(longitude)
        
        pano_id = pano_id if pano_id != "" else None
        tile_num = int(tile_num) if tile_num != "" else None
        heading = float(heading) if heading != "" else None
        pitch = float(pitch) if pitch != "" else None
        roll = float(roll) if roll != "" else None
        height = float(height) if height != "" else None
        timestamp_str = timestamp
        note = note if note != "" else None
        scene, img_id = extension.split(")")[0].split("(") if '(' in extension else (None, None)
        img_id = int(img_id) if img_id is not None else None
        
        return {
            "easting": easting,
            "northing": northing,
            "zone_number": zone_number,
            "zone_letter": zone_letter,
            "latitude": latitude,
            "longitude": longitude,
            "pano_id": pano_id,
            "tile_num": tile_num,
            "heading": heading,
            "pitch": pitch,
            "roll": roll,
            "height": height,
            "timestamp": timestamp,
            "note": note,
            "extension": extension,
            "scene": scene,
            "img_id": img_id,
        }
    except Exception:
        return None
        return None

def compute_diff_matrix(db_descs, query_descs) -> np.ndarray:
    """
        Return:
            D: np.ndarray, n_db x n_query
            query_descs: np.ndarray, n_query x descriptor_dim
    """
    dots = np.dot(db_descs, query_descs.T)
    db_norms = np.linalg.norm(db_descs, axis=1)[:, None]
    q_norms = np.linalg.norm(query_descs, axis=1)[None, :]
    D_matrix = 1.0 - dots / (db_norms * q_norms + 1e-8)
    return D_matrix

def viz_diff_matrix(save_img_path, D_matrix):
    fig = plt.figure(figsize=(18, 9))
    ax = fig.add_subplot(111)
    im = ax.imshow(D_matrix, cmap='Greys', aspect='auto')
    fig.colorbar(im, ax=ax)
    im.set_clim(0.0, 1.0)
    ax.set_xlabel('Query Index')
    ax.set_ylabel('Database Index')
    ax.set_title("Difference Matrix")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_pca(args, model, pca_ds, full_features_dim):
    model = model.eval()
    dl = torch.utils.data.DataLoader(pca_ds, args.infer_batch_size, shuffle=True)
    pca_features = np.empty([min(len(pca_ds), 2**14), full_features_dim])
    with torch.no_grad():
        for i, images in enumerate(dl):
            if i*args.infer_batch_size >= len(pca_features):
                break
            features = model(images).cpu().numpy()
            pca_features[i*args.infer_batch_size : (i*args.infer_batch_size)+len(features)] = features
    pca = PCA(args.pca_dim)
    pca.fit(pca_features)
    return pca

# def zero_padding_tensor(descs, sequence_length):
#     """
#     Zero pad the tensor to the sequence length
#     Args:
#         descs: torch.Tensor, (seq_len, desc_dim)
#         sequence_length: int

#     Returns:
#         torch.Tensor, (sequence_length, desc_dim)
#     """
#     if descs.shape[0] < sequence_length:
#         pad_len = sequence_length - descs.shape[0]
#         if descs.ndim == 2:
#             pad = torch.zeros((pad_len, descs.shape[1]), dtype=descs.dtype, device=descs.device)
#         elif descs.ndim == 3:
#             pad = torch.zeros((pad_len, descs.shape[1], descs.shape[2]), dtype=descs.dtype, device=descs.device)
#         elif descs.ndim == 4:
#             pad = torch.zeros((pad_len, descs.shape[1], descs.shape[2], descs.shape[3]), dtype=descs.dtype, device=descs.device)
#         else:
#             raise ValueError(f"Unsupported tensor dimension: {descs.ndim}")
#         descs = torch.cat([descs, pad], dim=0)
#     return descs

if __name__ == "__main__":
    image_name = "@0767652.88@3542131.10@36@R@031.98366@0035.83263@ot6eSG8GM1PJ-_fCLHqSNA@@@@@@20181221@night_forward_amman@vewLfr8ISxmORaCqIc5AHg(223).jpg"
    print(parse_image_name(image_name))
