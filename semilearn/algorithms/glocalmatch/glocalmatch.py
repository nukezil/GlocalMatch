import torch
import torch.nn as nn
import torch.nn.functional as F
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import DistAlignQueueHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, concat_all_gather
from semilearn.algorithms.glocalmatch.utils import single_kmeans, update_semantic_prototypes, \
     get_cluster_labels, get_shadow_centroids


class GlocalMatchNet(nn.Module):
    def __init__(self, base, proj_size=128):
        super(GlocalMatchNet, self).__init__()
        self.backbone = base
        self.num_features = base.num_features

        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_features, proj_size)
        ])

    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out

    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = self.l2norm(self.mlp_proj(feat))
        return {'logits': logits, 'feat': feat, 'feat_proj': feat_proj}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


@ALGORITHMS.register('glocalmatch')
class GlocalMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # GlocalMatch specified arguments
        self.lambda_c = args.cluster_loss_ratio
        self.T = args.T
        self.p_cutoff = args.p_cutoff
        self.smoothing_alpha = args.smoothing_alpha
        self.use_da = args.use_da
        self.da_len = args.da_len
        self.lambda_p = 1.0

        self.len_lb = len(self.dataset_dict['train_lb'])
        self.len_ulb = len(self.dataset_dict['train_ulb'])
        self.cluster_interval = self.len_ulb // (args.all_batch_size * args.uratio) + 1

        self.cluster_scale = args.cluster_scale
        self.min_cluster = args.min_cluster
        self.num_centroids = self.num_classes * self.cluster_scale
        self.structural_centroids = None
        self.shadow_centroids = None
        self.structural_assignments = None

        self.feats_lb = torch.zeros(self.len_lb, self.model.num_features).cuda(self.gpu)
        self.feats_proj_lb = F.normalize(torch.zeros(self.len_lb, self.args.proj_size).cuda(self.gpu), 1)
        self.labels_lb = torch.zeros(self.len_lb, dtype=torch.long).cuda(self.gpu)
        self.plabels_ulb = torch.zeros(self.len_ulb, dtype=torch.long).cuda(self.gpu) - 1
        self.semantic_prototypes = None

        self.feats_ulb = torch.zeros(self.len_ulb, self.model.num_features).cuda(self.gpu)
        self.feats_proj_ulb = F.normalize(torch.zeros(self.len_ulb, self.args.proj_size).cuda(self.gpu), 1)
        self.cluster_labels = None

        self.use_ema_bank = args.use_ema_bank
        self.ema_bank_m = args.ema_bank_m

    def set_hooks(self):
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),
            "DistAlignHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        model = GlocalMatchNet(model, proj_size=self.args.proj_size)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = GlocalMatchNet(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model

    @torch.no_grad()
    def update_bank(self, feats, feats_proj, labels, indices, labeled=True):
        if self.distributed and self.world_size > 1:
            feats = concat_all_gather(feats)
            feats_proj = concat_all_gather(feats_proj)
            labels = concat_all_gather(labels)
            indices = concat_all_gather(indices)

        if labeled:
            self.feats_lb[indices] = feats
            self.feats_proj_lb[indices] = feats_proj
            self.labels_lb[indices] = labels
        else:
            self.feats_ulb[indices] = feats
            self.feats_proj_ulb[indices] = feats_proj
            self.plabels_ulb[indices] = labels

    def train_step(self, idx_lb, idx_ulb, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        num_ulb = idx_ulb.shape[0]

        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits, feats, feats_proj = outputs['logits'], outputs['feat'], outputs['feat_proj']
                logits_x_lb, feats_x_lb, feats_proj_x_lb = logits[:num_lb], feats[:num_lb], feats_proj[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
                feats_x_ulb_w, _ = feats[num_lb:].chunk(2)
                feats_proj_x_ulb_w, feats_proj_x_ulb_s = feats_proj[num_lb:].chunk(2)
            else:
                raise ValueError("Must set use_cat as True!")

            if self.it > 0 and self.it % (self.len_lb // self.args.all_batch_size + 1) == 0:
            # if self.it > 0 and self.it % self.cluster_interval == 0:
                self.semantic_prototypes = update_semantic_prototypes(self.feats_proj_lb, self.labels_lb,
                                                                      self.num_classes)
                # self.semantic_prototypes = update_fused_semantic_prototypes(self.feats_proj_lb, self.labels_lb,
                #                                                             self.feats_proj_ulb, self.plabels_ulb,
                #                                                             self.num_classes)

            if self.it > 0 and self.it % self.cluster_interval == 0:
                gpu_kmeans = False
                if self.args.dataset in ['domainnet', 'domainnet_balanced']:
                    gpu_kmeans = True
                self.structural_centroids, self.structural_assignments = single_kmeans(self.num_centroids,
                                                                                       self.feats_proj_ulb,
                                                                                       seed=self.args.seed,
                                                                                       use_gpu=gpu_kmeans)
                self.shadow_centroids = get_shadow_centroids(self.structural_assignments, self.num_centroids,
                                                             self.feats_ulb)
                self.cluster_labels = get_cluster_labels(self.semantic_prototypes, self.structural_centroids,
                                                         min_cluster=self.min_cluster)

            # supervised loss
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            with torch.no_grad():
                logits_x_ulb_w = logits_x_ulb_w.detach()
                feats_proj_x_ulb_w = feats_proj_x_ulb_w.detach()

                probs_sem = self.compute_prob(logits_x_ulb_w)
                if self.use_da:
                    probs_sem = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_sem.detach())

            if self.epoch > 0 and self.smoothing_alpha < 1 and self.structural_centroids is not None:
                sim = torch.exp(torch.mm(feats_proj_x_ulb_w, self.structural_centroids.t()) / self.T)
                sim_probs = sim / sim.sum(1, keepdim=True)
                probs_clus = torch.zeros((num_ulb, self.num_classes)).cuda(self.gpu)
                for c in range(self.num_classes):
                    mask_c = (self.cluster_labels == c)
                    probs_clus[:, c] = sim_probs[:, mask_c].sum(1)
                probs_smoothed = probs_sem * self.smoothing_alpha + probs_clus * (1 - self.smoothing_alpha)
                probs = probs_smoothed
            else:
                probs = probs_sem

            if self.epoch > 0 and self.structural_centroids is not None:
                with torch.no_grad():
                    ctr_prt_sim = torch.exp(torch.mm(self.structural_centroids, self.semantic_prototypes.t()) / self.T)
                    p_centroids = ctr_prt_sim / ctr_prt_sim.sum(1, keepdim=True)
                    W_local = F.one_hot(self.structural_assignments[idx_ulb], num_classes=self.num_centroids)
                    ins_ctr_sim = torch.mm(probs_sem, p_centroids.t())
                    W_global = ins_ctr_sim / ins_ctr_sim.sum(1, keepdim=True)
                    W_glocal = W_global + W_local
                    W_glocal = W_glocal / W_glocal.sum(1, keepdim=True)
                cluster_loss = self.cluster_loss(feats_proj_x_ulb_s, W_glocal, self.T)
            else:
                cluster_loss = torch.zeros(1).cuda(self.gpu)

            pl_ulb = probs.max(1)[1]
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs, softmax_x_ulb=False)
            pl_ulb[~(mask.bool())] = -1
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               probs,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_c * cluster_loss

            self.update_bank(feats_x_lb, feats_proj_x_lb, y_lb, idx_lb)
            self.update_bank(feats_x_ulb_w, feats_proj_x_ulb_w, pl_ulb, idx_ulb, labeled=False)

        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         cluster_loss=cluster_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def cluster_loss(self, feats_proj_x_ulb_s, centroids_sim, T=0.2):
        sim = torch.exp(torch.mm(feats_proj_x_ulb_s, self.structural_centroids.t()) / T)
        sim_probs = sim / sim.sum(1, keepdim=True)
        loss = -(torch.log(sim_probs + 1e-7) * centroids_sim).sum(1)
        loss = loss.mean()
        return loss

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_model_ptr'] = self.hooks_dict['DistAlignHook'].p_model_ptr.cpu()
        save_dict['feats_lb'] = self.feats_lb.cpu()
        save_dict['feats_proj_lb'] = self.feats_proj_lb.cpu()
        save_dict['feats_ulb'] = self.feats_ulb.cpu()
        save_dict['feats_proj_ulb'] = self.feats_proj_ulb.cpu()
        save_dict['labels_lb'] = self.labels_lb.cpu()
        # save_dict['semantic_prototypes'] = self.semantic_prototypes.cpu()
        # save_dict['structural_centroids'] = self.structural_centroids.cpu()
        # save_dict['shadow_centroids'] = self.shadow_centroids.cpu()
        # save_dict['structural_assignments'] = self.structural_assignments.cpu()
        # save_dict['cluster_labels'] = self.cluster_labels.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_model_ptr = checkpoint['p_model_ptr'].cuda(self.args.gpu)
        self.feats_lb = checkpoint['feats_lb'].cuda(self.args.gpu)
        self.feats_proj_lb = checkpoint['feats_proj_lb'].cuda(self.args.gpu)
        self.feats_ulb = checkpoint['feats_ulb'].cuda(self.args.gpu)
        self.feats_proj_ulb = checkpoint['feats_proj_ulb'].cuda(self.args.gpu)
        self.labels_lb = checkpoint['labels_lb'].cuda(self.args.gpu)
        # self.semantic_prototypes = checkpoint['semantic_prototypes'].cuda(self.args.gpu)
        # self.structural_centroids = checkpoint['structural_centroids'].cuda(self.args.gpu)
        # self.shadow_centroids = checkpoint['shadow_centroids'].cuda(self.args.gpu)
        # self.structural_assignments = checkpoint['structural_assignments'].cuda(self.args.gpu)
        # self.cluster_labels = checkpoint['cluster_labels'].cuda(self.args.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--T', float, 0.2),
            SSL_Argument('--cluster_loss_ratio', float, 1.0),
            SSL_Argument('--proj_size', int, 128),
            SSL_Argument('--smoothing_alpha', float, 0.9),
            SSL_Argument('--use_da', str2bool, True),
            SSL_Argument('--da_len', int, 256),
            SSL_Argument('--cluster_scale', int, 100),
            SSL_Argument('--min_cluster', float, 0.9),
            SSL_Argument('--use_ema_bank', str2bool, False),
            SSL_Argument('--ema_bank_m', float, 0.7),
        ]
