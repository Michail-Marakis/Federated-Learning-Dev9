import numpy as np
import torch
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from m_utils import *
from collections import Counter
from math import ceil
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA, PCA


class Client(object):
    def __init__(self, idx, args, train_loader, train_loader_for_filtering=None):
        self.idx = idx
        self.args = args
        self.model = None
        self.filtering_model = None
        self.train_loader_for_filtering = train_loader_for_filtering

        self.train_loader = train_loader
        self.original_train_loader = train_loader

        self.train_iterator = iter(self.train_loader) if not args.filtering else None
        self.device = torch.device(f'cuda:{args.device}')

        self.cluster_and_center = []

    def calculated_cluster_center(self):
        # ✅ FIX: safe model selection
        model = self.model if self.args.filtering_model == 'same' else self.filtering_model

        if hasattr(model, "config"):
            model.config.output_hidden_states = True

        model.to(self.device)

        loader = self.original_train_loader if self.args.filtering_model == 'same' else self.train_loader_for_filtering
        model = self.model if self.args.filtering_model == 'same' else self.filtering_model

        model.config.output_hidden_states = True
        model.config.return_dict = True

        model.to(self.device)
        flatten_hidden_state_list = get_flatten_features(model, loader, args=self.args)

        # ✅ FIX: fallback if too few samples
        if len(flatten_hidden_state_list) < 2:
            flatten_hidden_state_list = np.array(flatten_hidden_state_list)
            self.cluster_labels = np.zeros(len(flatten_hidden_state_list), dtype=int)
            self.centroids = flatten_hidden_state_list
            self.reduced_feature_list = flatten_hidden_state_list
            self.selected_clusters = []
            return zip(self.cluster_labels, self.centroids)

        # dimensionality reduction
        if self.args.feature_layer == 'tsne':
            tsne = TSNE(n_components=self.args.compound_dim,
                        perplexity=min(30, len(flatten_hidden_state_list) - 1))
            reduced_feature_list = tsne.fit_transform(np.array(flatten_hidden_state_list))

        elif self.args.feature_layer == 'pca':
            reduced_feature_list = PCA(n_components=self.args.compound_dim)\
                .fit_transform(flatten_hidden_state_list)

        elif self.args.feature_layer == 'kpca':
            reduced_feature_list = KernelPCA(n_components=self.args.compound_dim)\
                .fit_transform(flatten_hidden_state_list)

        else:
            reduced_feature_list = np.array(flatten_hidden_state_list)

        cluster_labels, centroids, _ = clustering(reduced_feature_list, args=self.args)

        self.reduced_feature_list = reduced_feature_list
        self.cluster_labels = cluster_labels
        self.centroids = centroids
        self.selected_clusters = []

        return zip(cluster_labels, centroids)

    def build_training_set_with_precalculated_clusters(self):
        select_sample_id_list = []

        for cluster_id in self.selected_clusters:
            sample_id_in_cluster = np.argwhere(self.cluster_labels == cluster_id).flatten()

            if len(sample_id_in_cluster) == 0:
                continue

            features_in_cluster = self.reduced_feature_list[sample_id_in_cluster]
            distances = np.linalg.norm(
                features_in_cluster - self.centroids[cluster_id], axis=1
            )

            select_sample_id_list.append(
                sample_id_in_cluster[np.argmin(distances)]
            )

        if len(select_sample_id_list) > 0:
            subset_train = [
                self.original_train_loader.dataset[idx]
                for idx in select_sample_id_list
            ]

            self.train_loader = DataLoader(
                subset_train,
                shuffle=True,
                batch_size=self.args.batch_size,
                collate_fn=self.original_train_loader.collate_fn
            )
            self.train_iterator = iter(self.train_loader)
        else:
            self.train_iterator = None

    def local_train(self, cur_round, memory_record_dic=None):
        self.model.to(self.device)

        lr = self.args.lr * math.pow(self.args.lr_decay, cur_round - 1)

        iter_steps = (
            self.args.local_step * len(self.train_loader)
            if self.args.batch_or_epoch == 'epoch'
            else self.args.local_step
        )

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        progress_bar = tqdm(range(iter_steps))

        for _ in progress_bar:
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                batch = next(self.train_iterator)

            batch = {
                'input_ids': batch['input_ids'].to(self.device),
                'labels': batch['labels'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            }

            outputs = self.model(**batch)
            loss = outputs.loss

            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()

        self.model = self.model.cpu()

    def pull(self, forked_global_model):
        self.model = forked_global_model

    def pull_filtering_model(self, filtering_model):
        self.filtering_model = filtering_model

    def clear_model(self):
        self.model = None
