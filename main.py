
import sys
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import vpr_models
import parser
import commons
import visualizations
from test_dataset import TestDataset
import time

args = parser.parse_arguments()
start_time = datetime.now()
log_dir = Path("logs") / args.log_dir / start_time.strftime('%Y-%m-%d_%H-%M-%S')
commons.setup_logging(log_dir, stdout="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}")
logging.info(f"The outputs are being saved in {log_dir}")

model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
model = model.eval().to(args.device)

test_ds = TestDataset(args.database_folder, args.queries_folder,
                      positive_dist_threshold=args.positive_dist_threshold,
                      image_size=args.image_size, use_labels=args.use_labels)
logging.info(f"Testing on {test_ds}")

with torch.inference_mode():
    logging.debug("Extracting database descriptors for evaluation/testing")
    database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
    database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                      batch_size=args.batch_size)
    all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
    for images, indices in tqdm(database_dataloader):
        descriptors = model(images.to(args.device))
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors
        
    logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
    queries_subset_ds = Subset(test_ds,
                               list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries)))
    queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                    batch_size=1)
    start_time = time.time()
    for images, indices in tqdm(queries_dataloader):
        descriptors = model(images.to(args.device))
        print('Extract desc costs: {:3f}s'.format(time.time() - start_time))
        descriptors = descriptors.cpu().numpy()
        all_descriptors[indices.numpy(), :] = descriptors

queries_descriptors = all_descriptors[test_ds.num_database:]
database_descriptors = all_descriptors[:test_ds.num_database]

if args.save_descriptors:
    logging.info(f"Saving the descriptors in {log_dir}")
    np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
    np.save(log_dir / "database_descriptors.npy", database_descriptors)

# Use a kNN to find predictions
start_time = time.time()
faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
faiss_index.add(database_descriptors)
del database_descriptors, all_descriptors

logging.debug("Calculating recalls")
_, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))
print('Matching desc costs: {}s'.format((time.time() - start_time) / len(queries_descriptors)))

# For each query, check if the predictions are correct
if args.use_labels:
    positives_per_query = test_ds.get_positives()
    recalls = np.zeros(len(args.recall_values))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    
    # Divide by num_queries and multiply by 100, so the recalls are in percentages
    recalls = recalls / test_ds.num_queries * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    logging.info(recalls_str)

# Save visualizations of predictions
if args.num_preds_to_save != 0:
    logging.info("Saving final predictions")
    # For each query save num_preds_to_save predictions
    visualizations.save_preds(predictions[:, :args.num_preds_to_save], test_ds,
                              log_dir, args.save_only_wrong_preds, args.use_labels)

