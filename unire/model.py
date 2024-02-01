import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from clip import clip
from sentence_transformers import util


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


class unire(nn.Module):
    def __init__(self, args, config: dict):
        super().__init__()
        self.args = args
        self.config = config
        self.loss_config = config['loss_config']
        self.device = torch.device(args.gpu)

        # set model backbone
        self.clip_model, self.preprocess = clip.load(config['clip_model'], device=self.device, jit=False)
        self.embed_dim = self.clip_model.embed_dim
        # projection layer for image, one is for cross-modal retrieval, the other is for uni-modal retrieval
        # for corss-modal retrieval
        if self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel"):
            self.ln_cross_image_projection = nn.LayerNorm(self.embed_dim)
            self.ln_cross_text_projection = nn.LayerNorm(self.embed_dim)
            self.cross_image_projection = nn.Linear(self.embed_dim, self.embed_dim)
            self.cross_text_projection = nn.Linear(self.embed_dim, self.embed_dim)
        # for uni-modal retrieval
        if self.is_mode_on("uni_softlabel"):
            self.ln_uni_image_projection = nn.LayerNorm(self.embed_dim)
            self.ln_uni_text_projection = nn.LayerNorm(self.embed_dim)
            self.uni_image_projection = nn.Linear(self.embed_dim, self.embed_dim)
            self.uni_text_projection = nn.Linear(self.embed_dim, self.embed_dim)

        # set tau
        if self.is_mode_on("contrastive"):
            self.__init_tau = self.loss_config['contrastive']['tau']
            self.tau = nn.Parameter(torch.tensor(self.__init_tau, device=self.device))

        if self.is_mode_on("cross_softlabel"):
            self.__init_cross_image_tau = self.loss_config['cross_softlabel']['image_tau']
            self.__init_cross_text_tau = self.loss_config['cross_softlabel']['text_tau']
            self.__init_cross_tau = (self.__init_cross_image_tau + self.__init_cross_text_tau) / 2.0
            self.__init_cross_the_softlabel_image_tau = self.loss_config['cross_softlabel']['the_softlabel_image_tau']
            self.__init_cross_the_softlabel_text_tau = self.loss_config['cross_softlabel']['the_softlabel_text_tau']
            self.__init_cross_the_softlabel_tau = (self.__init_cross_the_softlabel_image_tau + self.__init_cross_the_softlabel_text_tau) / 2.0
            if self.is_each_cross_soft_mode():
                if self.loss_config['cross_softlabel']['use_same_tau']:
                    self.cross_tau = nn.Parameter(torch.tensor(self.__init_cross_tau, device=self.device))
                else:
                    self.cross_tau_image = nn.Parameter(torch.tensor(self.__init_cross_image_tau, device=self.device))
                    self.cross_tau_text = nn.Parameter(torch.tensor(self.__init_cross_text_tau, device=self.device))
                if self.loss_config['cross_softlabel']['use_same_softlabel_tau']:
                    self.cross_the_softlabel_tau = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_tau, device=self.device))
                else:
                    self.cross_the_softlabel_tau_image = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_image_tau, device=self.device))
                    self.cross_the_softlabel_tau_text = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_text_tau, device=self.device))
            else:
                self.cross_tau = nn.Parameter(torch.tensor(self.__init_cross_tau, device=self.device))
                self.cross_the_softlabel_tau = nn.Parameter(torch.tensor(self.__init_cross_the_softlabel_tau, device=self.device))

        if self.is_mode_on("uni_softlabel"):
            self.__init_uni_image_tau = self.loss_config['uni_softlabel']['image_tau']
            self.__init_uni_text_tau = self.loss_config['uni_softlabel']['text_tau']
            self.__init_uni_tau = (self.__init_uni_image_tau + self.__init_uni_text_tau) / 2.0
            self.__init_uni_the_softlabel_image_tau = self.loss_config['uni_softlabel']['the_softlabel_image_tau']
            self.__init_uni_the_softlabel_text_tau = self.loss_config['uni_softlabel']['the_softlabel_text_tau']
            self.__init_uni_the_softlabel_tau = (self.__init_uni_the_softlabel_image_tau + self.__init_uni_the_softlabel_text_tau) / 2.0
            
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if self.loss_config['uni_softlabel']['use_same_tau']:
                    self.uni_tau = nn.Parameter(torch.tensor(self.__init_uni_tau, device=self.device))
                else:
                    self.uni_tau_image = nn.Parameter(torch.tensor(self.__init_uni_image_tau, device=self.device))
                    self.uni_tau_text = nn.Parameter(torch.tensor(self.__init_uni_text_tau, device=self.device))

            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau']:
                if self.loss_config['uni_softlabel']['use_same_softlabel_tau']:
                    self.uni_the_softlabel_tau = nn.Parameter(torch.tensor(self.__init_uni_the_softlabel_tau, device=self.device))
                else:
                    self.uni_the_softlabel_tau_image = nn.Parameter(torch.tensor(self.__init_uni_the_softlabel_image_tau, device=self.device))
                    self.uni_the_softlabel_tau_text = nn.Parameter(torch.tensor(self.__init_uni_the_softlabel_text_tau, device=self.device))

        self.initialize_parameters()

    def is_all_gather(self):
        """check if all_gather"""
        return "is_all_gather" in self.config and self.config['is_all_gather']

    def is_mode_on(self, modeName: str) -> bool:
        return self.loss_config[modeName]['is_on']

    def is_add_cross_soft_mode(self):
        """check if add softlabel"""
        return self.is_mode_on("cross_softlabel") and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "add"

    def is_dot_cross_soft_mode(self):
        """check if dot softlabel"""
        return self.is_mode_on("cross_softlabel") and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "dot"

    def is_each_cross_soft_mode(self):
        """check if each softlabel"""
        return self.is_mode_on("cross_softlabel") and self.loss_config['cross_softlabel']['cross_softlabel_mode'] == "each"

    def is_mean_contrastive_loss_mode(self, lossName):
        return self.loss_config[lossName]['contrastive_loss_mode'] == "mean"

    def is_sum_contrastive_loss_mode(self, lossName):
        return self.loss_config[lossName]['contrastive_loss_mode'] == "sum"

    def encode_image(self, image, cross_modal=True):
        """Returns the image embedding "z" of shape [batch_size, projection_dim]."""
        image_features = self.clip_model.encode_image(image)
        return self._encode_image_features(image_features, cross_modal=cross_modal)

    def encode_text(self, text, cross_modal=True):
        """Returns the text embedding "z" of shape [batch_size, projection_dim]."""
        text_features = self.clip_model.encode_text(text)
        return self._encode_text_features(text_features, cross_modal=cross_modal)

    def _encode_image_features(self, image_features, cross_modal=True):
        """encode from clip model"""
        if cross_modal and (self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel")):
            image_features = self.ln_cross_image_projection(image_features)
            image_features = self.cross_image_projection(image_features)
        elif (not cross_modal) and self.is_mode_on("uni_softlabel"):
            image_features = self.ln_uni_image_projection(image_features)
            image_features = self.uni_image_projection(image_features)
        return image_features

    def _encode_text_features(self, text_features, cross_modal=True):
        """encode from clip model"""
        if cross_modal and (self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel")):
            text_features = self.ln_cross_text_projection(text_features)
            text_features = self.cross_text_projection(text_features)
        elif (not cross_modal) and self.is_mode_on("uni_softlabel"):
            text_features = self.ln_uni_text_projection(text_features)
            text_features = self.uni_text_projection(text_features)
        return text_features

    def get_similarity(self, image_features, text_features, cross_modal=True):
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if cross_modal:
            """if cross-modal retrieval, return the similarity between image and text"""
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            return logits_per_image, logits_per_text
        else:
            """if uni-modal retrieval, return the similarity between image and image, text and text"""
            logits_image_image = image_features @ image_features.t()
            logits_text_text = text_features @ text_features.t()
            return logits_image_image, logits_text_text

    def initialize_parameters(self):
        """Initialize the model parameters."""
        if self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel"):
            nn.init.normal_(self.cross_image_projection.weight, std=0.02)
            nn.init.normal_(self.cross_text_projection.weight, std=0.02)
        if self.is_mode_on("uni_softlabel"):
            nn.init.normal_(self.uni_image_projection.weight, std=0.02)
            nn.init.normal_(self.uni_text_projection.weight, std=0.02)

        if self.is_mode_on("contrastive"):
            if self.loss_config['contrastive']['is_block_tau']:
                self.tau.requires_grad_(False)

        if self.is_mode_on("cross_softlabel"):
            if self.loss_config['cross_softlabel']['is_block_tau']:
                if hasattr(self, "cross_tau"):
                    self.cross_tau.requires_grad_(False)
                else:
                    self.cross_tau_image.requires_grad_(False)
                    self.cross_tau_text.requires_grad_(False)
            if self.loss_config['cross_softlabel']['is_block_softlabel_tau']:
                if hasattr(self, "cross_the_softlabel_tau"):
                    self.cross_the_softlabel_tau.requires_grad_(False)
                else:
                    self.cross_the_softlabel_tau_image.requires_grad_(False)
                    self.cross_the_softlabel_tau_text.requires_grad_(False)

        if self.is_mode_on("uni_softlabel"):
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if self.loss_config['uni_softlabel']['is_block_tau']:
                    if hasattr(self, "uni_tau"):
                        self.uni_tau.requires_grad_(False)
                    else:
                        self.uni_tau_image.requires_grad_(False)
                        self.uni_tau_text.requires_grad_(False)
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau']:
                if self.loss_config['uni_softlabel']['is_block_softlabel_tau']:
                    if hasattr(self, "uni_the_softlabel_tau"):
                        self.uni_the_softlabel_tau.requires_grad_(False)
                    else:
                        self.uni_the_softlabel_tau_image.requires_grad_(False)
                        self.uni_the_softlabel_tau_text.requires_grad_(False)

    def load_state_dict(self, state_dict, strict=True):
        """load state dict"""
        if state_dict is None:
            return "state_dict is None"
        msg = super().load_state_dict(state_dict, strict)
        return msg

    def ContrastiveLoss(self, logits_per_image, logits_per_text, idx=None):
        # contrastive loss
        if idx is None:
            sim_targets = torch.eye(logits_per_image.shape[0], device=self.device)
        else:
            idx = idx.view(-1, 1)
            pos_idx = torch.eq(idx, idx.t()).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        if self.is_mean_contrastive_loss_mode("contrastive"):
            loss_i2t = -torch.mean(F.log_softmax(logits_per_image / self.tau, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.mean(F.log_softmax(logits_per_text / self.tau, dim=1) * sim_targets, dim=1).mean()
        elif self.is_sum_contrastive_loss_mode("contrastive"):
            loss_i2t = -torch.sum(F.log_softmax(logits_per_image / self.tau, dim=1) * sim_targets, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits_per_text / self.tau, dim=1) * sim_targets, dim=1).mean()
        else:
            raise ValueError("contrastive loss mode error")
        contrastive_loss = loss_i2t + loss_t2i

        return contrastive_loss

    def KLContrastiveSimLoss(self, logits, softlabel, tau, softlabel_tau, lossName, use_loss="kl"):
        """
        KL divergence loss
        make logits and softlabel have the same distribution
        logits to softlabel
        """
        # softmax for softlabel
        sim_targets = F.softmax(softlabel / softlabel_tau, dim=1)

        # log softmax
        logit_inputs = F.log_softmax(logits / tau, dim=1)

        if use_loss == "kl":
            # KL divergence
            loss = F.kl_div(logit_inputs, sim_targets, reduction='batchmean')
        elif use_loss == "contrastive":
            # Switch to the same loss as ContrastiveLoss, but sim_targets is soft
            if self.is_mean_contrastive_loss_mode(lossName):
                loss = -torch.mean(logit_inputs * sim_targets, dim=1).mean()
            elif self.is_sum_contrastive_loss_mode(lossName):
                loss = -torch.sum(logit_inputs * sim_targets, dim=1).mean()
            else:
                raise ValueError("contrastive loss mode error")
        else:
            raise ValueError("loss mode error")

        return loss

    @torch.no_grad()
    def clamp_tau(self):
        # clip tau to prevent overflow
        if self.is_mode_on("contrastive"):
            self.tau.clamp_(min=self.loss_config['contrastive']['tau_min'], max=self.loss_config['contrastive']['tau_max'])

        if self.is_mode_on("cross_softlabel"):
            if hasattr(self, "cross_tau"):
                self.cross_tau.clamp_(min=(self.loss_config['cross_softlabel']['image_tau_min']+self.loss_config['cross_softlabel']['text_tau_min'])/2.0,
                                      max=(self.loss_config['cross_softlabel']['image_tau_max']+self.loss_config['cross_softlabel']['text_tau_max'])/2.0)
            else:
                self.cross_tau_image.clamp_(min=self.loss_config['cross_softlabel']['image_tau_min'],
                                            max=self.loss_config['cross_softlabel']['image_tau_max'])
                self.cross_tau_text.clamp_(min=self.loss_config['cross_softlabel']['text_tau_min'],
                                           max=self.loss_config['cross_softlabel']['text_tau_max'])
            if hasattr(self, "cross_the_softlabel_tau"):
                self.cross_the_softlabel_tau.clamp_(min=(self.loss_config['cross_softlabel']['the_softlabel_image_tau_min']+self.loss_config['cross_softlabel']['the_softlabel_text_tau_min'])/2.0,
                                                    max=(self.loss_config['cross_softlabel']['the_softlabel_image_tau_max']+self.loss_config['cross_softlabel']['the_softlabel_text_tau_max'])/2.0)
            else:
                self.cross_the_softlabel_tau_image.clamp_(min=self.loss_config['cross_softlabel']['the_softlabel_image_tau_min'],
                                                          max=self.loss_config['cross_softlabel']['the_softlabel_image_tau_max'])
                self.cross_the_softlabel_tau_text.clamp_(min=self.loss_config['cross_softlabel']['the_softlabel_text_tau_min'],
                                                         max=self.loss_config['cross_softlabel']['the_softlabel_text_tau_max'])

        if self.is_mode_on("uni_softlabel"):
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if hasattr(self, "uni_tau"):
                    self.uni_tau.clamp_(min=(self.loss_config['uni_softlabel']['image_tau_min']+self.loss_config['uni_softlabel']['text_tau_min'])/2.0,
                                        max=(self.loss_config['uni_softlabel']['image_tau_max']+self.loss_config['uni_softlabel']['text_tau_max'])/2.0)
                else:
                    self.uni_tau_image.clamp_(min=self.loss_config['uni_softlabel']['image_tau_min'],
                                            max=self.loss_config['uni_softlabel']['image_tau_max'])
                    self.uni_tau_text.clamp_(min=self.loss_config['uni_softlabel']['text_tau_min'],
                                            max=self.loss_config['uni_softlabel']['text_tau_max'])
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau']:
                if hasattr(self, "uni_the_softlabel_tau"):
                    self.uni_the_softlabel_tau.clamp_(min=(self.loss_config['uni_softlabel']['the_softlabel_image_tau_min']+self.loss_config['uni_softlabel']['the_softlabel_text_tau_min'])/2.0,
                                                      max=(self.loss_config['uni_softlabel']['the_softlabel_image_tau_max']+self.loss_config['uni_softlabel']['the_softlabel_text_tau_max'])/2.0)
                else:
                    self.uni_the_softlabel_tau_image.clamp_(min=self.loss_config['uni_softlabel']['the_softlabel_image_tau_min'],
                                                            max=self.loss_config['uni_softlabel']['the_softlabel_image_tau_max'])
                    self.uni_the_softlabel_tau_text.clamp_(min=self.loss_config['uni_softlabel']['the_softlabel_text_tau_min'],
                                                           max=self.loss_config['uni_softlabel']['the_softlabel_text_tau_max'])

    def forward(self, image, text, softlabel_image_features=None, softlabel_text_features=None, epoch=None, idx=None):
        rankNum = torch.distributed.get_rank()
        worldSize = torch.distributed.get_world_size()
        # clip tau to prevent overflow
        self.clamp_tau()

        # use clip model to extract features
        # can be used for both cross-modal and uni-modal retrieval
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(text)

        if self.is_all_gather() and idx is not None:
            idx_all = allgather(idx, rankNum, worldSize)
        else:
            idx_all = idx

        # use clip model to extract features and similarity
        # for cross-modal retrieval
        if self.is_mode_on("contrastive") or self.is_mode_on("cross_softlabel"):
            cross_image_features, cross_text_features = self._encode_image_features(
                image_features, cross_modal=True), self._encode_text_features(text_features, cross_modal=True)
            if self.is_all_gather():
                cross_image_features, cross_text_features = allgather(
                    cross_image_features, rankNum, worldSize), allgather(cross_text_features, rankNum, worldSize)
            logits_per_image, logits_per_text = self.get_similarity(cross_image_features, cross_text_features, cross_modal=True)

        # for uni-modal retrieval
        if self.is_mode_on("uni_softlabel"):
            uni_image_features, uni_text_features = self._encode_image_features(
                image_features, cross_modal=False), self._encode_text_features(text_features, cross_modal=False)
            if self.is_all_gather():
                uni_image_features, uni_text_features = allgather(uni_image_features, rankNum, worldSize), allgather(
                    uni_text_features, rankNum, worldSize)
            logits_image_image, logits_text_text = self.get_similarity(uni_image_features, uni_text_features, cross_modal=False)

        # use external softlabel to get similarity
        # only image-image and text-text similarity
        if self.is_mode_on("cross_softlabel") or self.is_mode_on("uni_softlabel"):
            with torch.no_grad():
                if self.is_all_gather():
                    softlabel_image_features, softlabel_text_features = allgather(
                        softlabel_image_features, rankNum, worldSize), allgather(softlabel_text_features, rankNum, worldSize)
                softlabel_image_sim = util.cos_sim(softlabel_image_features, softlabel_image_features)
                softlabel_text_sim = util.cos_sim(softlabel_text_features, softlabel_text_features)

                if self.is_mode_on("cross_softlabel"):
                    if self.is_add_cross_soft_mode():
                        # Average two similarities
                        softlabel_all_sim = (softlabel_image_sim + softlabel_text_sim) / 2.0
                    elif self.is_dot_cross_soft_mode():
                        # Dot two similarities
                        softlabel_image_sim_copy = softlabel_image_sim.clone()
                        softlabel_text_sim_copy = softlabel_text_sim.clone()
                        softlabel_image_sim_copy[softlabel_image_sim_copy < 0.0] = 0.0
                        softlabel_text_sim_copy[softlabel_text_sim_copy < 0.0] = 0.0
                        softlabel_all_sim = softlabel_image_sim_copy * softlabel_text_sim_copy
                        softlabel_all_sim = torch.sqrt(softlabel_all_sim)
                    elif self.is_each_cross_soft_mode():
                        pass
                    else:
                        raise ValueError("softlabel mode error")

        cross_modal_loss, uni_modal_loss, contrastive_loss = torch.tensor(0.0, device=self.device), torch.tensor(
            0.0, device=self.device), torch.tensor(0.0, device=self.device)

        if self.is_mode_on("cross_softlabel"):
            # for cross-modal alignment (similarity)
            # image-text and image-image softlabel
            # text-image and text-text softlabel
            softlabel_image_sim_loss = softlabel_image_sim
            softlabel_text_sim_loss = softlabel_text_sim
            if not self.is_each_cross_soft_mode():
                softlabel_image_sim_loss = softlabel_all_sim
                softlabel_text_sim_loss = softlabel_all_sim

            if hasattr(self, "cross_tau"):
                cross_tau_loss_image = self.cross_tau
                cross_tau_loss_text = self.cross_tau
            else:
                cross_tau_loss_image = self.cross_tau_image
                cross_tau_loss_text = self.cross_tau_text

            if hasattr(self, "cross_the_softlabel_tau"):
                cross_the_softlabel_tau_loss_image = self.cross_the_softlabel_tau
                cross_the_softlabel_tau_loss_text = self.cross_the_softlabel_tau
            else:
                cross_the_softlabel_tau_loss_image = self.cross_the_softlabel_tau_image
                cross_the_softlabel_tau_loss_text = self.cross_the_softlabel_tau_text

            cross_modal_loss = self.KLContrastiveSimLoss(logits_per_image, softlabel_image_sim_loss, cross_tau_loss_image,
                                                         cross_the_softlabel_tau_loss_image, "cross_softlabel", use_loss=self.loss_config['cross_softlabel']['use_loss'])
            cross_modal_loss += self.KLContrastiveSimLoss(logits_per_text, softlabel_text_sim_loss, cross_tau_loss_text,
                                                          cross_the_softlabel_tau_loss_text, "cross_softlabel", use_loss=self.loss_config['cross_softlabel']['use_loss'])
            cross_modal_loss /= 2.0
            cross_modal_loss = cross_modal_loss * self.loss_config['cross_softlabel']['rate']

        if self.is_mode_on("uni_softlabel"):
            # fro uni-modal alignment (similarity)
            # image-image and image-image softlabel
            # text-text and text-text softlabel
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_tau']:
                if hasattr(self, "uni_tau"):
                    uni_tau_image_loss = self.uni_tau
                    uni_tau_text_loss = self.uni_tau
                else:
                    uni_tau_image_loss = self.uni_tau_image
                    uni_tau_text_loss = self.uni_tau_text
            else:
                if hasattr(self, "cross_tau"):
                    uni_tau_image_loss = self.cross_tau
                    uni_tau_text_loss = self.cross_tau
                else:
                    uni_tau_image_loss = self.cross_tau_image
                    uni_tau_text_loss = self.cross_tau_text
            if not self.loss_config['uni_softlabel']['use_cross_softlabel_same_softlabel_tau']:
                if hasattr(self, "uni_the_softlabel_tau"):
                    uni_the_softlabel_tau_image_loss = self.uni_the_softlabel_tau
                    uni_the_softlabel_tau_text_loss = self.uni_the_softlabel_tau
                else:
                    uni_the_softlabel_tau_image_loss = self.uni_the_softlabel_tau_image
                    uni_the_softlabel_tau_text_loss = self.uni_the_softlabel_tau_text
            else:
                if hasattr(self, "cross_the_softlabel_tau"):
                    uni_the_softlabel_tau_image_loss = self.cross_the_softlabel_tau
                    uni_the_softlabel_tau_text_loss = self.cross_the_softlabel_tau
                else:
                    uni_the_softlabel_tau_image_loss = self.cross_the_softlabel_tau_image
                    uni_the_softlabel_tau_text_loss = self.cross_the_softlabel_tau_text

            uni_modal_loss = self.KLContrastiveSimLoss(logits_image_image, softlabel_image_sim, uni_tau_image_loss, uni_the_softlabel_tau_image_loss,
                                                       "uni_softlabel", use_loss=self.loss_config['uni_softlabel']['use_loss'])
            uni_modal_loss += self.KLContrastiveSimLoss(logits_text_text, softlabel_text_sim, uni_tau_text_loss, uni_the_softlabel_tau_text_loss,
                                                        "uni_softlabel", use_loss=self.loss_config['uni_softlabel']['use_loss'])
            uni_modal_loss /= 2.0
            uni_modal_loss = uni_modal_loss * self.loss_config['uni_softlabel']['rate']

        if self.is_mode_on("contrastive"):
            # the simplest contrastive loss
            # image-text and text-image
            contrastive_loss = self.ContrastiveLoss(logits_per_image, logits_per_text, idx_all)
            contrastive_loss /= 2.0
            contrastive_loss = contrastive_loss * self.loss_config['contrastive']['rate']

        return cross_modal_loss, uni_modal_loss, contrastive_loss
