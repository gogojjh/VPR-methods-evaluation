import os

sub_datasets = [
    "st_lucia", 
    "pitts250k", 
    "msls",
    "botanicgarden", 
    "wildscenes", 
    "opennavmap_ucl_campus", 
    "opennavmap_hkust"
]
subfolders = {
    "train": ["queries", "database"],
    "val": ["queries", "database"],
    "test": ["queries", "database"],
}

def create_symbolic_links(source_base_dir, dest_base_dir):
    for dataset_name in sub_datasets:
        for split in subfolders.keys():
            for folder in subfolders[split]:
                source_dir = os.path.join(source_base_dir, dataset_name, 'images', split, folder)
                dest_dir = os.path.join(dest_base_dir, 'images', split, folder)
                if os.path.isdir(source_dir):
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    for filename in os.listdir(source_dir):
                        source_file = os.path.join(source_dir, filename)
                        dest_file = os.path.join(dest_dir, filename)
                        
                        if not os.path.lexists(dest_file):
                            try:
                                os.symlink(source_file, dest_file)
                                # print(f"Created link: {dest_file} -> {source_file}")
                            except OSError as e:
                                print(f"Error creating link {dest_file}: {e}")
                        else:
                            print(f"Link already exists: {dest_file}")
                else:
                    print(f"Source directory not found, skipping: {source_dir}")


if __name__ == "__main__":
    source_datasets_path = "/Rocket_ssd/dataset/data_vpr"
    altasvpr_path = os.path.join(source_datasets_path, "altasvpr")

    create_symbolic_links(source_datasets_path, altasvpr_path)
    print("\nSymbolic link creation process finished.")
