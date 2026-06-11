import os
import re
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import utils

base_transformations = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]

class PCADataset(data.Dataset):
    def __init__(self, args, datasets_folder="dataset", dataset_folder="pitts30k/images/train"):
        dataset_folder_full_path = os.join(datasets_folder, dataset_folder)
        if not os.path.exists(dataset_folder_full_path) :
            raise FileNotFoundError(f"Folder {dataset_folder_full_path} does not exist")
        self.images_paths = sorted(glob(os.join(dataset_folder_full_path, "**", "*.jpg"), recursive=True))
        self.resize = args.resize
    def __getitem__(self, index):
        base_transform = transforms.Compose(base_transformations)
        img = base_transform(Image.open(self.images_paths[index]).convert("RGB"))
        img = transforms.functional.resize(img, self.resize)
        return img
    def __len__(self):
        return len(self.images_paths)

def read_images_paths(dataset_folder):
    """Find images within 'dataset_folder'. If the file
    'dataset_folder'_images_paths.txt exists, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over very large folders might be slow.

    Parameters
    ----------
    dataset_folder : str, folder containing images

    Returns
    -------
    images_paths : list[str], paths of images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        print(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [dataset_folder + "/" + path for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(
                f"Image with path {images_paths[0]} "
                f"does not exist within {dataset_folder}. It is likely "
                f"that the content of {file_with_paths} is wrong."
            )
    else:
        print(f"Searching test images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*", recursive=True))
        images_paths = [p for p in images_paths if os.path.isfile(p) and os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png"]]
        if len(images_paths) == 0:
            images_paths = sorted(glob(f"{dataset_folder}/**/*.png", recursive=True))
            if len(images_paths) == 0:
                raise FileNotFoundError(f"Directory {dataset_folder} does not contain any JPEG or PNG images")
    return images_paths

class TestDataset(data.Dataset):
    def __init__(self, database_folder, queries_folder, positive_dist_threshold=25, image_size=None, use_labels=True, seq_len=1):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()

        self.seq_len = seq_len

        # Image paths in random order
        raw_database_paths = read_images_paths(database_folder)
        raw_queries_paths = read_images_paths(queries_folder)

        # Image paths in sequential order
        self.database_paths, self.queries_paths, self.data_groups = self.parse_and_sort_paths(raw_database_paths, raw_queries_paths)
        self.images_paths = list(self.database_paths) + list(self.queries_paths)

        self.num_database = len(self.database_paths)
        self.num_queries = len(self.queries_paths)

        if use_labels:
            try:
                image_path = self.database_paths[0]
                utm_east = float(image_path.split("@")[1])
                utm_north = float(image_path.split("@")[2])
                utm_zone = int(image_path.split("@")[3])
                utm_letter = image_path.split("@")[4]
                latitude = float(image_path.split("@")[5])
                longitude = float(image_path.split("@")[6])
            except:
                raise ValueError(
                    "The path of images should be path/to/file/@utm_east@utm_north@...@.jpg "
                    f"but it is {image_path}, which does not contain the UTM coordinates."
                )

            self.database_pos = np.array(
                [utils.get_ecef_coords(
                    path.split("@")[1], path.split("@")[2], int(path.split("@")[3]), 
                    float(path.split("@")[5]), float(path.split("@")[6])
                ) for path in self.database_paths]
            )
            self.queries_pos = np.array(
                [utils.get_ecef_coords(
                    path.split("@")[1], path.split("@")[2], int(path.split("@")[3]), 
                    float(path.split("@")[5]), float(path.split("@")[6])
                ) for path in self.queries_paths]
            )

            # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_pos)
            self.positives_per_query = knn.radius_neighbors(
                self.queries_pos, radius=positive_dist_threshold, return_distance=False
            )

        transformations = base_transformations
        if image_size:
            transformations.append(transforms.Resize(size=image_size, antialias=True))

        self.transform = transforms.Compose(transformations)

    def parse_and_sort_paths(self, raw_database_paths, raw_queries_paths):
        num_database, num_queries = len(raw_database_paths), len(raw_queries_paths)
        parse_data = []
    
        def group_and_sort(items):
            scene_to_items = defaultdict(list)
            for item in items:
                scene_to_items[item['scene']].append(item)
            return [item for scene in sorted(scene_to_items)
                    for item in sorted(scene_to_items[scene], key=lambda x: x['img_id'])]

        parse_data = []
        for img_name in raw_database_paths + raw_queries_paths:
            parsed = utils.parse_image_name(img_name)
            if parsed is None or parsed.get('scene') is None:
                img_id_str = os.path.splitext(os.path.basename(img_name))[0]
                scene = 'default'
                img_id = 0
                for match in re.finditer(r'\d+', img_id_str):
                    img_id = int(match.group())
                parsed = {'scene': scene, 'img_id': img_id}
            parse_data.append({
                'path': img_name,
                'scene': parsed['scene'],
                'img_id': parsed['img_id']
            })

        data_groups_database = group_and_sort(parse_data[:num_database])
        data_groups_queries  = group_and_sort(parse_data[num_database:])
        data_groups = data_groups_database + data_groups_queries

        database_paths = [item['path'] for item in data_groups_database]
        queries_paths = [item['path'] for item in data_groups_queries]

        return database_paths, queries_paths, data_groups

    def __getitem__(self, index):
        scene = self.data_groups[index]['scene']

        image_paths, indices = [], []
        start_index = index - self.seq_len + 1
        for idx in range(start_index, index + 1):
            if idx < 0 or \
                (not index < self.num_database and idx < self.num_database) or \
                (self.data_groups[idx]['scene'] != scene):
                continue
            image_paths.append(self.images_paths[idx])
            indices.append(idx)

        if len(image_paths) < self.seq_len:
            image_paths = [image_paths[-1]] * (self.seq_len - len(image_paths)) + image_paths
            indices = [indices[-1]] * (self.seq_len - len(indices)) + indices

        imgs = []
        for idx, path in enumerate(image_paths):
            try:
                pil_img = Image.open(path).convert("RGB")
                normalized_img = self.transform(pil_img)
                imgs.append(normalized_img)
            except Exception as e:
                image_paths[idx] = image_paths[-1]
                indices[idx] = indices[-1]
                print(f"Error opening image {path}: {e}, using last image instead")
                continue

        imgs_tensor = torch.stack(imgs)
        indices_tensor = torch.tensor(indices)
        if self.seq_len == 1:
            return imgs_tensor.squeeze(0), indices_tensor.squeeze(0)
        return imgs_tensor, indices_tensor

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< #queries: {self.num_queries}; #database: {self.num_database} >"

    def get_positives(self):
        return self.positives_per_query

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from torch.utils.data import DataLoader, Subset
    import torch
    
    split = "test"
    dataset = TestDataset(
        database_folder=f"/Rocket_ssd/dataset/data_vpr/pitts250k/images/{split}/database",
        queries_folder=f"/Rocket_ssd/dataset/data_vpr/pitts250k/images/{split}/queries",
        positive_dist_threshold=25,
        image_size=(224, 224),
        use_labels=True,
        seq_len=5
    )

    def visualize_loader(loader, set_size, viz_img, max_batches=5):
        for batch_idx, (images, indices) in enumerate(tqdm(loader)):
            print(f"Images tensor shape: {images.shape} (Batch x Set_Size x C x H x W) and indices: {indices}")
            
            if viz_img:
                set_to_plot = images[0]
                fig, axes = plt.subplots(1, set_size, figsize=(set_size * 3, 3))
                fig.suptitle(f"Example Image Set from Batch {batch_idx + 1} (Set Size: {set_to_plot.shape[0]})", fontsize=10)
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
    for subset_range in [
        range(dataset.num_database), 
        range(dataset.num_database, dataset.num_database + dataset.num_queries)
    ]:
        subset_ds = Subset(dataset, subset_range)
        loader = DataLoader(subset_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        visualize_loader(loader, dataset.seq_len, viz_img, max_batches=3)

    # Plot database and query positions on a scatter plot
    db_pos = dataset.database_pos[:, :2]  # shape: (num_database, 2)
    q_pos = dataset.queries_pos[:, :2]    # shape: (num_queries, 2)

    plt.figure(figsize=(8, 6))
    plt.scatter(db_pos[:, 0], db_pos[:, 1], c='blue', label='Database', s=10, alpha=0.7)
    plt.scatter(q_pos[:, 0], q_pos[:, 1], c='red', label='Queries', s=10, alpha=0.7)
    plt.xlabel('ECEF X')
    plt.ylabel('ECEF Y')
    plt.title('Database and Query Positions')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()