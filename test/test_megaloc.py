import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party/MegaLoc'))
from megaloc_model import MegaLocModel
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import compute_diff_matrix, viz_diff_matrix
from test_dataset import TestDataset
import time

total_extract_time = 0
total_faiss_time = 0
total_matching_time = 0

def load_megaloc_model(device='cuda'):
    model = MegaLocModel()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://github.com/gmberton/MegaLoc/releases/download/v1.0/megaloc.torch", 
            map_location=torch.device(device)
        )
    )
    return model.eval().to(device)

def extract_descriptors(args, test_ds, seq_len):
    global total_extract_time
    dataset_folder = Path(args.database_folder).parent
    desc_folder = Path(os.path.join(dataset_folder, "../../descriptors/test"))
    os.makedirs(desc_folder, exist_ok=True)
    q_desc_path = desc_folder / f"megaloc_{seq_len}_{args.mm_method}_queries_descriptors.npy"
    db_desc_path = desc_folder / f"megaloc_{seq_len}_{args.mm_method}_database_descriptors.npy"
    
    try:
        raise FileNotFoundError
        queries_descriptors = np.load(q_desc_path)
        database_descriptors = np.load(db_desc_path)
    except FileNotFoundError:
        model = load_megaloc_model(args.device)
        with torch.inference_mode():
            all_descriptors = np.empty((len(test_ds), 8448), dtype="float32")
            full_dataloader = DataLoader(dataset=test_ds, num_workers=args.num_workers, batch_size=args.batch_size)
            for images, indices in tqdm(full_dataloader, desc="Extracting MegaLoc descriptors"):
                B, S, C, H, W = images.shape                
                start_time = time.perf_counter()
                if args.mm_method == "vlad_pooling":
                    descriptors = model(images.to(args.device))
                else:
                    descriptors = model(images.to(args.device))
                    descriptors = descriptors.view(B, S, -1).mean(dim=1)
                if args.device == "cuda": torch.cuda.synchronize()
                total_extract_time += time.perf_counter() - start_time
                
                all_descriptors[indices.numpy()[:, -1], :] = descriptors.cpu().numpy()
        
        database_descriptors = all_descriptors[:test_ds.num_database]
        queries_descriptors = all_descriptors[test_ds.num_database:]
        # np.save(db_desc_path, database_descriptors)
        # np.save(q_desc_path, queries_descriptors)
    
    L = min(2000, database_descriptors.shape[0])
    D = compute_diff_matrix(database_descriptors[:L, :], queries_descriptors[:L, :])
    viz_diff_matrix(desc_folder / f"megaloc_{args.sequence_length}_{args.mm_method}_diff_matrix.png", D)
    return database_descriptors, queries_descriptors

def evaluate_and_save(args, db_descs, q_descs, test_ds, log_dir):
    global total_extract_time, total_faiss_time, total_matching_time
    start_time = time.perf_counter()
    faiss_index = faiss.IndexFlatL2(db_descs.shape[1])
    faiss_index.add(db_descs)
    total_faiss_time += time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    _, predictions = faiss_index.search(q_descs, max(args.recall_values))
    total_matching_time += time.perf_counter() - start_time

    if args.use_labels:
        # Only keep queries with positive database images
        positives_per_query = test_ds.get_positives()
        queries_with_positives = np.where(np.array([len(positives) > 0 for positives in positives_per_query]))[0]
        num_queries_with_positives = queries_with_positives.shape[0]
        
        recalls = np.zeros(len(args.recall_values))
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        recalls = recalls / num_queries_with_positives * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
        logger.info(recalls_str)

        with open(Path('logs') / f"{args.log_dir}_{args.sequence_length}_{args.mm_method}_recalls.txt", "w") as f:
            f.write(recalls_str + "\n")

        predictions = predictions[queries_with_positives, :]

        with open(Path('logs') / f"{args.log_dir}_{args.sequence_length}_{args.mm_method}_time.txt", "w") as f:
            msg =  f"Total extract time: {1000 * total_extract_time:.6f}ms, Mean extract time: {1000 * total_extract_time / len(test_ds):.6f}ms\n"
            msg += f"Total faiss time: {1000 * total_faiss_time:.6f}ms, Mean faiss time: {1000 * total_faiss_time / test_ds.num_database:.6f}ms\n"
            msg += f"Total matching time: {1000 * total_matching_time:.6f}ms, Mean matching time: {1000 * total_matching_time / test_ds.num_queries:.6f}ms\n"
            f.write(msg)

    if args.num_preds_to_save > 0:
        logger.info("Saving final predictions")
        import visualizations
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels
        )        

def main():
    parser = argparse.ArgumentParser(description='Test MegaLoc model')
    parser.add_argument('--database_folder', type=str, required=True, help='Path to database folder')
    parser.add_argument('--queries_folder', type=str, required=True, help='Path to queries folder')
    parser.add_argument('--positive_dist_threshold', type=int, default=25, help='Positive distance threshold in meters')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Image size (height, width)')
    parser.add_argument('--use_labels', action='store_true', default=True, help='Use ground truth labels for evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--recall_values', type=int, nargs='+', default=[1, 5, 10, 20, 50, 100], help='Recall values to compute')
    parser.add_argument('--num_preds_to_save', type=int, default=0, help='Number of predictions to save')
    parser.add_argument('--save_only_wrong_preds', action='store_true', default=False, help='Save only wrong predictions')
    parser.add_argument('--log_dir', type=str, default='megaloc_test', help='Log directory name')
    parser.add_argument('--sequence_length', type=int, default=1, help='Sequence length')
    parser.add_argument('--mm_method', type=str, default='max_pooling', help='Matching method')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.remove()
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.info(f"Arguments: {args}")

    # Single Image Matching
    test_ds = TestDataset(
        args.database_folder, args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size, use_labels=args.use_labels,
        seq_len=args.sequence_length
    )
    logger.info(f"Testing on {test_ds}")
    db_descs, q_descs = extract_descriptors(args, test_ds, args.sequence_length)
    
    evaluate_and_save(args, db_descs, q_descs, test_ds, log_dir)
    logger.info("Finished successfully")

if __name__ == "__main__":
    main()
