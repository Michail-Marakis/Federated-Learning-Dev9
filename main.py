import argparse
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from hdbscan import HDBSCAN

# ===== SAFE HDBSCAN =====
def safe_hdbscan_fit(center_list):
    try:
        import numpy as np
        center_array = np.array(center_list)

        if center_array.ndim == 1:
            center_array = center_array.reshape(-1, 1)

        clusterer = HDBSCAN(min_cluster_size=2, allow_single_cluster=False)
        clusterer.fit(center_array)
        return clusterer

    except Exception:
        class MockCluster:
            def __init__(self, centers):
                self.labels_ = [0] * max(1, len(centers))
                self.centers = centers

            def weighted_cluster_centroid(self, cid):
                return self.centers[0]

        return MockCluster(center_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # federation
    parser.add_argument('--num_clients', type=int, default=738)
    parser.add_argument('-k', type=float, default=0.05)
    parser.add_argument('--rounds', type=int, default=40)
    parser.add_argument('--batch_or_epoch', type=str, default='epoch', choices=['epoch', 'batch'])
    parser.add_argument('--local_step', type=int, default=1)
    parser.add_argument('--equal_weight', default=False, action='store_true')

    # data
    parser.add_argument('--dataset', type=str, default='alpaca')
    parser.add_argument('--data_sample', type=float, default=1.0)
    parser.add_argument('--iid', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--zeroshot', default=True, action='store_true')
    parser.add_argument('--zerotask', default='7', type=str)
    parser.add_argument('--split', type=str, default='[0.98, 0.01, 0.01]')
    parser.add_argument('--train_eval_ratio', default='[0.99, 0.01]', type=str)
    parser.add_argument('--use_prompts', default=False, action='store_true')

    # Filtering
    parser.add_argument('--filtering', action='store_true', default=False)
    parser.add_argument('--feature_layer', default='-1', type=str)
    parser.add_argument('--compound_dim', default=2, type=int)
    parser.add_argument('--feature_token', default='avg', type=str, choices=['avg', 'last'])
    parser.add_argument('--clustering_score', default='ch', type=str)
    parser.add_argument('--clustering', type=str, default='kmeans')
    parser.add_argument('--n_cluster', type=int, default=7)
    parser.add_argument('--kernel_ratio', type=float, default=1.0)
    parser.add_argument('--filtering_model', type=str, default='same')
    parser.add_argument('--dp_noise', type=float, default=0.0)
    parser.add_argument('--min_cluster', type=int, default=5)

    # model
    parser.add_argument('--model', type=str, default='datajuicer/LLaMA-1B-dj-refine-150B')
    parser.add_argument('--peft', action='store_true', default=False)
    parser.add_argument('--peft_method', default='lora', type=str)

    # training
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=-100.0)

    # env
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log', default=False, action='store_true')
    parser.add_argument('--log_root', default='')
    parser.add_argument('--seed', default=42, type=int)

    # eval
    parser.add_argument('--eval_metrics', default='none', type=str)
    parser.add_argument('--generate_eval', default='rouge', type=str)
    parser.add_argument('--eval_subsampling', default=False, action='store_true')
    parser.add_argument('--full_evaluation', default=False, action='store_true')
    parser.add_argument('--start_eval_epoch', default=30, type=int)
    parser.add_argument('--eval_interval', default=20, type=int)
    parser.add_argument('--loss', default=False, action='store_true')

    parser.add_argument('--save', default=False, action='store_true')

    # INIT
    time_stamp = str(time.time())
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    import random
    import numpy as np
    import torch
    from server import Server
    from client import Client
    from utils_data.load_data import get_loaders, get_loaders_for_filtering
    import yaml
    from copy import deepcopy
    import json

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    train_avg_acc = []
    eval_avg_acc = []
    metric_history = {}
    memory_record_dic = {}
    train_time_history = []
    extract_time_history = []

    setup_seed(args.seed)

    list_train_loader, eval_loader, _ = get_loaders(args)

    list_train_loader_for_filtering = None
    if args.filtering_model != 'same':
        list_train_loader_for_filtering, _, _ = get_loaders_for_filtering(args)

    log_dir = os.path.join('exp', time_stamp)
    os.makedirs(log_dir, exist_ok=True)

    server = Server(args, eval_loader=eval_loader, log_dir=log_dir)

    if list_train_loader_for_filtering is None:
        list_train_loader_for_filtering = [None for _ in range(args.num_clients)]

    client_list = [
        Client(i, args, list_train_loader[i], list_train_loader_for_filtering[i])
        for i in range(args.num_clients)
    ]

    client_indices_rounds = [
        np.random.choice(np.arange(args.num_clients),
                         size=max(1, int(args.num_clients * args.k)),
                         replace=False)
        for _ in range(args.rounds)
    ]

    # ================= LOOP =================
    for r in range(1, args.rounds + 1):
        print(f'ROUND {r}')

        selected_client = [client_list[i] for i in client_indices_rounds[r - 1]]

        server.prepare_aggregate()

        staged = []

        for client in selected_client:
            client.pull(deepcopy(server.model))

            for cid, centroid in client.calculated_cluster_center():
                staged.append((cid, client.idx, centroid))

            client.clear_model()

        # ===== SAFE CLUSTERING =====
        center_list = [x[2] for x in staged]
        clusterer = safe_hdbscan_fit(center_list)
        cluster_labels = clusterer.labels_

        center_list = np.array(center_list) if len(center_list) > 0 else np.array([])

        selected_ids = []

        if len(cluster_labels) > 0 and np.max(cluster_labels) > -1:
            for cid in range(np.max(cluster_labels) + 1):
                tmp = clusterer.weighted_cluster_centroid(cid)

                idxs = np.where(cluster_labels == cid)[0]
                feats = center_list[idxs]

                dist = np.linalg.norm(feats - tmp, axis=1)
                selected_ids.append(idxs[np.argmin(dist)])

        for idx in selected_ids:
            cid, client_id = staged[idx][0], staged[idx][1]
            client_list[client_id].selected_clusters.append(cid)

        for client in selected_client:
            client.build_training_set_with_precalculated_clusters()

        selected_client = [c for c in selected_client if c.train_iterator is not None]

        for client in selected_client:
            client.pull(deepcopy(server.model))

            start = time.time()
            client.local_train(cur_round=r)
            end = time.time()

            train_time_history.append(end - start)

            server.online_aggregate(client, selected_client)
            client.clear_model()

        server.finish_aggregate()

    result = server.eval(cur_round=args.rounds, eval_avg_acc=eval_avg_acc)
    print("FINAL:", result)

# ===== FINAL EVAL =====
result = server.eval(cur_round=args.rounds, eval_avg_acc=eval_avg_acc)
print("FINAL:", result)

# ===== SAFE IMPORT =====
import os
import json

# ===== EXTRACT ROUGE =====
final_eval_rouge = None

if isinstance(result, dict):
    if "rouge" in result:
        final_eval_rouge = result["rouge"]
    elif "final_eval_rouge" in result:
        final_eval_rouge = result["final_eval_rouge"]
    else:
        final_eval_rouge = result
else:
    final_eval_rouge = result


# ===== 1) SAVE ACCURACY JSON =====
acc_output = {
    "eval_avg_acc": eval_avg_acc
}

with open(os.path.join(log_dir, "eval_avg_acc.json"), "w") as f:
    json.dump(acc_output, f, indent=2)


# ===== 2) SAVE ROUGE JSON =====
rouge_output = {
    "final_eval_rouge": final_eval_rouge
}

with open(os.path.join(log_dir, "eval_rouge.json"), "w") as f:
    json.dump(rouge_output, f, indent=2)


# ===== LOG =====
print("[LOG] saved eval_avg_acc.json")
print("[LOG] saved eval_rouge.json")

print(json.dumps(acc_output, indent=2))
print(json.dumps(rouge_output, indent=2))
