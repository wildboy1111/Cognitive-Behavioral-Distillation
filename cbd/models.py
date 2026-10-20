"""CBD teacher and image-only student.

Parameter names and the effective original student loss equations are retained.
The frozen teacher is passed explicitly and is never stored in student weights.
"""

import torch
from torch import nn

from .encoders import EEGEncoder, EyeTrackingEncoder, ImageEncoder
from .losses import MutualInfoLoss, OrthogonalityLoss


def _gate(input_dim, hidden_dim, dropout):
    """Shared gate layout with independent weights in each semantic branch."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.Tanh(),
    )


class Teacher(nn.Module):
    """Bidirectional image/EEG/ET gates with four supervised classification heads."""

    def __init__(self, config, pretrained=True):
        super().__init__()
        n_classes = config.num_classes
        d = 256
        head_num = 1
        self.n_classes = n_classes
        self.hidden_dim = d
        self.head_num = head_num
        self.dropout = 0.4
        self.eeg_model = EEGEncoder(
            n_classes=n_classes,
            channels=config.eeg_channels,
            samples=config.eeg_samples,
        )
        self.img_model = ImageEncoder(n_classes=n_classes, pretrained=pretrained)
        self.et_model = EyeTrackingEncoder(
            n_classes=n_classes, input_size=(config.et_samples, config.et_features)
        )
        self.img2eeg_attn_g = _gate(self.hidden_dim * 2, self.hidden_dim, self.dropout)
        self.eeg2img_attn_g = _gate(self.hidden_dim * 2, self.hidden_dim, self.dropout)
        self.img2et_attn_l = _gate(self.hidden_dim * 2, self.hidden_dim, self.dropout)
        self.et2img_attn_l = _gate(self.hidden_dim * 2, self.hidden_dim, self.dropout)
        self.eeg_attn_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_classes)
        )
        self.img_g_attn_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_classes)
        )
        self.img_l_attn_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_classes)
        )
        self.et_attn_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.n_classes)
        )
        self.softmax = nn.Softmax(dim=1)
        self.loss_MI = MutualInfoLoss()

    def forward(self, x_img, x_eeg, x_et, labels=None, return_features=False):
        eeg_g = self.eeg_model(x_eeg, labels)["global_feat"]
        img_g, img_l = self.img_model(x_img, labels)["global_feat, local_feat"]
        et_l = self.et_model(x_et, labels)["local_feat"]
        eeg2img_g_attn = self.eeg2img_attn_g(torch.concat([eeg_g, img_g], dim=1))
        img2eeg_g_attn = self.img2eeg_attn_g(torch.concat([eeg_g, img_g], dim=1))
        et2img_l_attn = self.et2img_attn_l(torch.concat([et_l, img_l], dim=1))
        img2et_l_attn = self.img2et_attn_l(torch.concat([et_l, img_l], dim=1))
        eeg_g_attn = eeg2img_g_attn * eeg_g
        et_l_attn = et2img_l_attn * et_l
        img_g_attn = img2eeg_g_attn * img_g
        img_l_attn = img2et_l_attn * img_l
        loss_MI = self.loss_MI(eeg_g_attn, img_g_attn) + self.loss_MI(
            et_l_attn, img_l_attn
        )
        eeg_attn_out = self.eeg_attn_classifier(eeg_g_attn)
        img_g_attn_out = self.img_g_attn_classifier(img_g_attn)
        img_l_attn_out = self.img_l_attn_classifier(img_l_attn)
        et_attn_out = self.et_attn_classifier(et_l_attn)
        out = {
            "logits": {
                "eeg": eeg_attn_out,
                "img_l": img_l_attn_out,
                "img_g": img_g_attn_out,
                "et": et_attn_out,
            },
            "loss": {"loss_MI": loss_MI},
        }
        if return_features:
            out["features"] = {
                "eeg2img_g_attn": eeg2img_g_attn,
                "img2eeg_g_attn": img2eeg_g_attn,
                "et2img_l_attn": et2img_l_attn,
                "img2et_l_attn": img2et_l_attn,
                "eeg_g_attn": eeg_g_attn,
                "et_l_attn": et_l_attn,
                "img_g_attn": img_g_attn,
                "img_l_attn": img_l_attn,
            }
        return out


class Student(nn.Module):
    """Image-only semantic/private branches and four-token attention fusion."""

    def __init__(self, config, pretrained=True):
        super().__init__()
        n_classes = config.num_classes
        d = 256
        self.n_classes = n_classes
        self.hidden_dim = d
        self.dropout1 = config.student_dropout
        self.dropout2 = config.private_dropout
        self.img_model = ImageEncoder(n_classes=n_classes, pretrained=pretrained)
        self.img2eeg_fc_g = _gate(self.hidden_dim, self.hidden_dim, self.dropout1)
        self.img2et_fc_l = _gate(self.hidden_dim, self.hidden_dim, self.dropout1)
        self.img_g_p = _gate(self.hidden_dim, self.hidden_dim, self.dropout2)
        self.img_l_p = _gate(self.hidden_dim, self.hidden_dim, self.dropout2)
        self.attn_fusion = nn.MultiheadAttention(
            embed_dim=d, num_heads=1, batch_first=True
        )
        self.img_classifier = nn.Sequential(nn.Linear(self.hidden_dim, self.n_classes))
        self.softmax = nn.Softmax(dim=1)
        self.loss_distill_weight = nn.MSELoss()
        self.loss_distill_feature = nn.MSELoss()
        self.loss_dec = OrthogonalityLoss()

    def forward(self, x_img, x_eeg=None, x_et=None, labels=None, teacher=None):
        (
            loss_distill_weights_eeg,
            loss_distill_weights_et,
            loss_distill_feature_eeg,
            loss_distill_feature_et,
            loss_distill,
            loss_dec,
        ) = [x_img.new_zeros(()) for _ in range(6)]
        img_g, img_l = self.img_model(x_img, labels)["global_feat, local_feat"]
        img2eeg_g_weight = self.img2eeg_fc_g(img_g)
        img2et_l_weight = self.img2et_fc_l(img_l)
        img_g_d_eeg = img2eeg_g_weight * img_g
        img_l_d_et = img2et_l_weight * img_l
        img_g_p = self.img_g_p(img_g) * img_g
        img_l_p = self.img_l_p(img_l) * img_l
        loss_dec = self.loss_dec(img_g_p, img_g_d_eeg) + self.loss_dec(
            img_l_p, img_l_d_et
        )
        if teacher is not None:
            if x_eeg is None or x_et is None:
                raise ValueError("Distillation requires EEG and ET inputs.")
            teacher.eval()
            with torch.no_grad():
                tea_out = teacher(x_img, x_eeg, x_et, labels, return_features=True)
            loss_distill_weights_eeg = self.loss_distill_weight(
                img2eeg_g_weight, tea_out["features"]["img2eeg_g_attn"]
            )
            loss_distill_weights_et = self.loss_distill_weight(
                img2et_l_weight, tea_out["features"]["img2et_l_attn"]
            )
            loss_distill_feature_eeg = self.loss_distill_feature(
                img_g_d_eeg, tea_out["features"]["img_g_attn"]
            )
            loss_distill_feature_et = self.loss_distill_feature(
                img_l_d_et, tea_out["features"]["img_l_attn"]
            )
            teacher_confidence_eeg = tea_out["logits"]["img_g"].max(dim=1)[0].mean()
            teacher_confidence_et = tea_out["logits"]["img_l"].max(dim=1)[0].mean()
            loss_distill_weights_eeg_w = (
                teacher_confidence_eeg
                * loss_distill_weights_eeg
                / (loss_distill_weights_eeg + loss_distill_feature_eeg + 1e-06)
            )
            loss_distill_weights_et_w = (
                teacher_confidence_et
                * loss_distill_weights_et
                / (loss_distill_weights_et + loss_distill_feature_et + 1e-06)
            )
            loss_distill_feature_eeg_w = (
                teacher_confidence_eeg
                * loss_distill_feature_eeg
                / (loss_distill_weights_eeg + loss_distill_feature_eeg + 1e-06)
            )
            loss_distill_feature_et_w = (
                teacher_confidence_et
                * loss_distill_feature_et
                / (loss_distill_weights_et + loss_distill_feature_et + 1e-06)
            )
            loss_distill_weights_eeg = (
                loss_distill_weights_eeg * loss_distill_weights_eeg_w
            )
            loss_distill_weights_et = (
                loss_distill_weights_et * loss_distill_weights_et_w
            )
            loss_distill_feature_eeg = (
                loss_distill_feature_eeg * loss_distill_feature_eeg_w
            )
            loss_distill_feature_et = (
                loss_distill_feature_et * loss_distill_feature_et_w
            )
            loss_distill = (
                loss_distill_weights_eeg
                + loss_distill_feature_eeg
                + loss_distill_weights_et
                + loss_distill_feature_et
            )
        img_v = torch.stack([img_g_d_eeg, img_l_d_et, img_g_p, img_l_p], dim=1)
        img_attn, _ = self.attn_fusion(img_v, img_v, img_v)
        img_attn = img_attn.mean(dim=1)
        img_out = self.img_classifier(img_attn)
        return {
            "logits": {"img": img_out},
            "loss": {
                "loss_distill": loss_distill,
                "loss_distill_weights_eeg": loss_distill_weights_eeg,
                "loss_distill_feature_eeg": loss_distill_feature_eeg,
                "loss_distill_weights_et": loss_distill_weights_et,
                "loss_distill_feature_et": loss_distill_feature_et,
                "loss_dec": loss_dec,
            },
            "features": {
                "img_g_d_eeg": img_g_d_eeg,
                "img_l_d_et": img_l_d_et,
                "img_g_p": img_g_p,
                "img_l_p": img_l_p,
            },
        }
