import numpy as np
import torch
import torch.nn.functional as F
from faiss import Kmeans as faiss_Kmeans
import networkx as nx


def single_kmeans(num_centroids, data, init_centroids=None, frozen_centroids=False, seed=0, use_gpu=False):
    data = data.cpu().detach().numpy()
    feat_dim = data.shape[-1]
    if init_centroids is not None:
        init_centroids = init_centroids.cpu().detach().numpy()
    km = faiss_Kmeans(
        feat_dim,
        num_centroids,
        niter=40,
        verbose=False,
        spherical=True,
        min_points_per_centroid=int(data.shape[0] / num_centroids * 0.9),
        gpu=use_gpu,
        seed=seed,
        frozen_centroids=frozen_centroids
    )
    km.train(data, init_centroids=init_centroids)
    _, assignments = km.index.search(data, 1)
    centroids = torch.from_numpy(km.centroids).cuda()
    assignments = torch.from_numpy(assignments).long().cuda().squeeze(1)

    return centroids, assignments


def multi_kmeans(K_list, data, seed=0, init_centroids=None, frozen_centroids=False):
    if init_centroids is None:
        init_centroids = [None] * len(K_list)
    centroids_list = []
    assignments_list = []
    for idx, K in enumerate(K_list):
        centroids, assignments = single_kmeans(K,
                                               data,
                                               init_centroids=init_centroids[idx],
                                               frozen_centroids=frozen_centroids,
                                               seed=seed+idx)
        centroids_list.append(centroids)
        assignments_list.append(assignments)

    return centroids_list, assignments_list


def update_semantic_prototypes(feats_lb, labels_lb, num_classes):
    feats = feats_lb
    labels = labels_lb
    feat_dim = feats.shape[1]
    prototypes = torch.zeros((num_classes, feat_dim)).cuda() + 1e-7
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes[c] = F.normalize(feats[mask].mean(0), dim=0)
    return prototypes


def update_fused_semantic_prototypes(feats_lb, labels_lb, feats_ulb, plabels_ulb, num_classes):
    feats = torch.cat((feats_lb, feats_ulb), dim=0)
    labels = torch.cat((labels_lb, plabels_ulb))
    feat_dim = feats.shape[1]
    prototypes = torch.zeros((num_classes, feat_dim)).cuda() + 1e-7
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            prototypes[c] = F.normalize(feats[mask].mean(0), dim=0)
    return prototypes


def get_cluster_labels(semantic_prototypes, structural_centroids, min_cluster=0.75):
    num_classes = semantic_prototypes.shape[0]
    num_clusters = structural_centroids.shape[0]
    cluster_scale = int(min_cluster * num_clusters / num_classes)
    dists = torch.cdist(structural_centroids, semantic_prototypes)
    G = nx.DiGraph()
    # Here with the networkx package, the demand value is actually "-b(v)" presented in the paper
    # Therefore, positive demand value means it is a demand node, and negative for a supply node
    for u in range(num_clusters):
        G.add_node(f"st_{u}", demand=-1)
        for v in range(num_classes):
            if u == 0:
                G.add_node(f"se_{v}", demand=cluster_scale)
            G.add_edge(f"st_{u}", f"se_{v}", capacity=1, weight=int(dists[u, v] * 1000))
            if v == 0:
                G.add_node(f"sink", demand=num_clusters - num_classes * cluster_scale)
            if u == 0:
                G.add_edge(f"se_{v}", "sink")
    flow = nx.min_cost_flow(G)
    cluster_labels = torch.zeros(num_clusters).long().cuda()
    for u in range(num_clusters):
        for v in range(num_classes):
            if flow[f"st_{u}"][f"se_{v}"] > 0:
                cluster_labels[u] = v
    return cluster_labels


def get_cluster_labels_simple(semantic_prototypes, structural_centroids, T):
    sim = torch.exp(torch.mm(structural_centroids, semantic_prototypes.t()) / T)
    sim_probs = sim / sim.sum(1, keepdim=True)
    _, cluster_labels = sim_probs.max(1)
    return cluster_labels


def get_shadow_centroids(structural_assignments, num_centroids, feats):
    feat_dim = feats.shape[1]
    shadow_centroids = torch.zeros((num_centroids, feat_dim)).cuda()
    for c in range(num_centroids):
        mask = (structural_assignments == c)
        shadow_centroids[c] = feats[mask].mean(0)
    return shadow_centroids
