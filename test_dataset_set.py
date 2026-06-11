import os
import logging
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from test_dataset import TestDataset
import util

pos_threshold_min = 0.0
pos_threshold = 7.0
ori_threshold = 30.0

def parse_pos_ori(path):
    data = util.parse_image_name(path)
    utm_east = data['easting']
    utm_north = data['northing']
    utm_zone = data['zone_number']
    utm_letter = data['zone_letter']
    latitude = data['latitude']
    longitude = data['longitude']
    orientation = data['heading']
    ecef_x, ecef_y, ecef_z = util.get_ecef_coords(utm_east, utm_north, utm_zone, latitude, longitude)
    return ecef_x, ecef_y, ecef_z, orientation

class TestDatasetSet(TestDataset):
    def __init__(self, database_folder, queries_folder, positive_dist_threshold=25, image_size=None, use_labels=True, seq_len=1):
        super().__init__(database_folder, queries_folder, positive_dist_threshold, image_size, use_labels, seq_len)
        
        self.database_pos = np.array([parse_pos_ori(path)[:3] for path in self.database_paths])
        self.queries_pos = np.array([parse_pos_ori(path)[:3] for path in self.queries_paths])
        self.all_pos = np.concatenate([self.database_pos, self.queries_pos])

        self.database_orientations = np.array([parse_pos_ori(path)[3] for path in self.database_paths])
        self.queries_orientations = np.array([parse_pos_ori(path)[3] for path in self.queries_paths])
        self.all_orientations = np.concatenate([self.database_orientations, self.queries_orientations])

        self.knn = NearestNeighbors(n_jobs=1)
        self.knn.fit(self.all_pos)
        
    def __getitem__(self, index):
        is_database = index < self.num_database            
        current_pos = np.array([parse_pos_ori(self.images_paths[index])[:2]])
        current_orientation = parse_pos_ori(self.images_paths[index])[3]
        
        distances, indices = self.knn.radius_neighbors(current_pos, radius=pos_threshold)
        distances = distances[0]
        indices = indices[0]

        valid_indices = []
        for i, idx in enumerate(indices):
            if idx == index: continue
            if is_database and idx >= self.num_database: continue
            if not is_database and idx < self.num_database: continue
                
            neighbor_orientation = self.all_orientations[idx]
            orientation_diff = abs(current_orientation - neighbor_orientation)
            orientation_diff = min(orientation_diff, 360 - orientation_diff)            
            if distances[i] >= pos_threshold_min and \
               distances[i] <= pos_threshold and \
               orientation_diff <= ori_threshold:
                valid_indices.append((idx, distances[i]))
        
        valid_indices.sort(key=lambda x: x[1])
        selected_indices = [index]
        for idx, _ in valid_indices[:self.seq_len - 1]:
            selected_indices.append(idx)
        selected_indices = [selected_indices[-1]] * (self.seq_len - len(selected_indices)) + selected_indices
        
        image_paths = [self.images_paths[i] for i in selected_indices]
        imgs = []
        for path in image_paths:
            from PIL import Image
            pil_img = Image.open(path).convert("RGB")
            normalized_img = self.transform(pil_img)
            imgs.append(normalized_img)

        return torch.stack(imgs), torch.tensor(selected_indices)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Subset
    import torch
    import torchvision.transforms as transforms
    
    split = "test"
    dataset = TestDatasetSet(
        database_folder=f"/Rocket_ssd/dataset/data_vpr/botanicgarden/images/{split}/database",
        queries_folder=f"/Rocket_ssd/dataset/data_vpr/botanicgarden/images/{split}/queries",
        positive_dist_threshold=25,
        image_size=(224, 224),
        use_labels=True,
        seq_len=5
    )

    def visualize_loader(loader, seq_len, viz_img, max_batches=5):
        for batch_idx, (images, indices) in enumerate(tqdm(loader)):
            print(f"Images tensor shape: {images.shape} (Batch x Seq_Len x C x H x W) and indices: {indices}")
            
            if viz_img:
                set_to_plot = images[0]
                fig, axes = plt.subplots(1, seq_len, figsize=(seq_len * 3, 3))
                fig.suptitle(f"Example Image Set from Batch {batch_idx + 1} (Seq_Len: {seq_len})", fontsize=10)
                mean = torch.tensor(dataset.transform.transforms[-2].mean).view(3, 1, 1)
                std = torch.tensor(dataset.transform.transforms[-2].std).view(3, 1, 1)
                for i in range(set_to_plot.shape[0]):
                    img = set_to_plot[i].cpu() * std + mean
                    img = transforms.functional.to_pil_image(img.clamp(0, 1))
                    axes[i].imshow(img)
                    axes[i].axis('off')
                    axes[i].set_title(f"Frame {i+1}", fontsize=8)
                plt.tight_layout()
                plt.show()
            
            if batch_idx >= max_batches - 1:
                break

    viz_img = False
    for subset_range in [range(dataset.num_database), range(dataset.num_database, dataset.num_database + dataset.num_queries)]:
        subset_ds = Subset(dataset, subset_range)
        loader = DataLoader(subset_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=False)
        visualize_loader(loader, dataset.seq_len, viz_img, max_batches=3)