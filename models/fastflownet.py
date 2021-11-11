from copy import deepcopy
from pprint import pprint

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.correlation import Correlation
from models.decoder import Decoder, deconv, convrelu


class FastFlowNet(pl.LightningModule):
    def __init__(self, groups=3, device="cpu", learning_rate=1e-4, q=0.4, robust_loss=True, is_finetuning=False):
        super(FastFlowNet, self).__init__()

        assert learning_rate > 0
        self.learning_rate = learning_rate
        # self.criterion = nn.MSELoss()
        assert isinstance(robust_loss, bool)
        self.robust_loss = robust_loss
        assert isinstance(is_finetuning, bool)
        self.is_finetuning = is_finetuning
        assert isinstance(q, float) or isinstance(q, int)
        self.q = float(q)

        self.groups = groups
        self.pconv1_1 = convrelu(3, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)

        self.corr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1, corr_multiply=1)
        self.index = torch.tensor([0, 2, 4, 6, 8,
                                   10, 12, 14, 16,
                                   18, 20, 21, 22, 23, 24, 26,
                                   28, 29, 30, 31, 32, 33, 34,
                                   36, 38, 39, 40, 41, 42, 44,
                                   46, 47, 48, 49, 50, 51, 52,
                                   54, 56, 57, 58, 59, 60, 62,
                                   64, 66, 68, 70,
                                   72, 74, 76, 78, 80])

        self.rconv2 = convrelu(32, 32, 3, 1)
        self.rconv3 = convrelu(64, 32, 3, 1)
        self.rconv4 = convrelu(64, 32, 3, 1)
        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, 3, 1)

        self.up3 = deconv(2, 2)
        self.up4 = deconv(2, 2)
        self.up5 = deconv(2, 2)
        self.up6 = deconv(2, 2)

        self.decoder2 = Decoder(87, groups)
        self.decoder2_f = Decoder(2, groups)
        self.decoder3 = Decoder(87, groups)
        self.decoder4 = Decoder(87, groups)
        self.decoder5 = Decoder(87, groups)
        self.decoder6 = Decoder(87, groups)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # self.a = SoftFullAttention()
        # self.b = SoftFullAttention()
        # self.c = SoftFullAttention()
        # self.d = SoftFullAttention()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def warp(self, x, flo):
        # print("start warp")
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).to(x)
        vgrid = grid + flo
        vgrid = torch.stack([(2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0),
                             (2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0)], 1)
        vgrid = vgrid.permute(0, 2, 3, 1)
        #todo check align_corners behavior
        output = F.grid_sample(x, vgrid, mode='bilinear', align_corners=True)
        # print("end warp")
        return output

    def forward(self, x):
        img1 = x[:, :3, :, :]
        img2 = x[:, 3:6, :, :]
        f11 = self.pconv1_2(self.pconv1_1(img1))
        f21 = self.pconv1_2(self.pconv1_1(img2))
        f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
        f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
        f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
        f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
        f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
        f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
        f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
        f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
        f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
        f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))

        flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
        cv6 = torch.index_select(self.corr(f16, f26), dim=1, index=self.index.to(f16).long())
        r16 = self.rconv6(f16)
        cat6 = torch.cat([cv6, r16, flow7_up], 1)
        flow6 = self.decoder6(cat6)

        flow6_up = self.up6(flow6)
        f25_w = self.warp(f25, flow6_up * 0.625)
        cv5 = torch.index_select(self.corr(f15, f25_w), dim=1, index=self.index.to(f15).long())
        r15 = self.rconv5(f15)
        cat5 = torch.cat([cv5, r15, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up

        flow5_up = self.up5(flow5)
        f24_w = self.warp(f24, flow5_up * 1.25)
        cv4 = torch.index_select(self.corr(f14, f24_w), dim=1, index=self.index.to(f14).long())
        r14 = self.rconv4(f14)
        cat4 = torch.cat([cv4, r14, flow5_up], 1)
        flow4 = self.decoder4(cat4) + flow5_up

        flow4_up = self.up4(flow4)
        f23_w = self.warp(f23, flow4_up * 2.5)
        cv3 = torch.index_select(self.corr(f13, f23_w), dim=1, index=self.index.to(f13).long())
        r13 = self.rconv3(f13)
        cat3 = torch.cat([cv3, r13, flow4_up], 1)
        flow3 = self.decoder3(cat3) + flow4_up

        flow3_up = self.up3(flow3)
        f22_w = self.warp(f22, flow3_up * 5.0)
        cv2 = torch.index_select(self.corr(f12, f22_w), dim=1, index=self.index.to(f12).long())
        r12 = self.rconv2(f12)
        cat2 = torch.cat([cv2, r12, flow3_up], 1)
        flow2 = self.decoder2(cat2) + flow3_up

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        first, last, gt = train_batch
        first, last, gt = first.to(self.device), last.to(self.device), gt.to(self.device)
        # gt = torch.sign(gt) * torch.log(torch.abs(gt)+2)
        gt = gt / 20
        # gt = torch.sign(gt) * (((torch.abs(gt)/100)+1)**2)

        outputs = self(torch.cat([first, last], 1))
        if not self.robust_loss:
            loss = nn.MSELoss()(outputs[0], gt)
        else:
            loss = None
            alphas = [0.32, 0.08, 0.02, 0.01, 0.005]
            alphas_reversed = [0.005, 0.01, 0.02, 0.08, 0.32]
            for alpha, flow_pred in zip(alphas, outputs):
                flow_gt = deepcopy(gt)
                # resizes the ground truth flow
                if flow_pred.shape != flow_gt.shape:
                    flow_gt = F.interpolate(flow_gt, size=(flow_pred.shape[2:]), mode='nearest')
                # check if the training has to use the L2 or L1 loss
                if not self.is_finetuning:
                    loss_partial = alpha * torch.norm(flow_gt - flow_pred, p=2,
                                                      dim=1).mean()  # nn.MSELoss()(flow_pred, flow_gt)
                else:
                    loss_partial = alpha * (nn.L1Loss()(flow_pred, flow_gt) + 1e-2) ** self.q
                # updates the cumulative loss
                if loss is None:
                    loss = loss_partial
                else:
                    loss += loss_partial
        return loss

    def validation_step(self, val_batch, batch_idx):
        first, last, flow_gt = val_batch
        first, last, flow_gt = first.to(self.device), last.to(self.device), flow_gt.to(self.device)

        flow_pred = self(torch.cat([first, last], 1))
        if flow_pred.shape != flow_gt.shape:
            flow_gt = F.interpolate(flow_gt, size=(flow_pred.shape[2:]), mode='bilinear', align_corners=True)
        flow_pred = flow_pred * 20
        outputs = {
            "loss": nn.MSELoss()(flow_pred, flow_gt),
            "epes_flownet2": torch.norm(flow_gt - flow_pred, p=2, dim=1).mean(),
            "epes_sqrt": torch.sqrt(nn.MSELoss()(flow_pred, flow_gt))
        }
        return outputs

    def validation_epoch_end(self, outputs):
        losses = torch.as_tensor([o["loss"].item() for o in outputs])
        mse = losses.sum() / len(losses)

        epes_flownet2 = torch.as_tensor([o["epes_flownet2"].item() for o in outputs])
        epe_avg_flownet2 = epes_flownet2.sum() / len(epes_flownet2)

        epes_sqrt = torch.as_tensor([o["epes_sqrt"].item() for o in outputs])
        epe_avg_sqrt = epes_sqrt.sum() / len(epes_sqrt)

        results = {
            "mse": mse,
            "epe_flownet2": epe_avg_flownet2,
            "epe_sqrt": epe_avg_sqrt
        }
        print(f"Validation results:")
        pprint(results)

    def test_step(self, test_batch, batch_idx):
        first, last, flow_gt = test_batch
        first, last, flow_gt = first.to(self.device), last.to(self.device), flow_gt.to(self.device)

        flow_pred = self(torch.cat([first, last], 1))
        if flow_pred.shape != flow_gt.shape:
            flow_gt = F.interpolate(flow_gt, size=(flow_pred.shape[2:]), mode='bilinear', align_corners=True)
        flow_pred = flow_pred * 20
        outputs = {
            "loss": nn.MSELoss()(flow_pred, flow_gt),
            "epes_flownet2": torch.norm(flow_gt - flow_pred, p=2, dim=1, keepdim=True).mean(),
            "epes_sqrt": torch.sqrt(nn.MSELoss()(flow_pred, flow_gt))
        }
        return outputs

    def test_epoch_end(self, outputs):
        losses = torch.as_tensor([o["loss"].item() for o in outputs])
        mse = losses.sum() / len(losses)

        epes_flownet2 = torch.as_tensor([o["epes_flownet2"].item() for o in outputs])
        epe_avg_flownet2 = epes_flownet2.sum() / len(epes_flownet2)

        epes_sqrt = torch.as_tensor([o["epes_sqrt"].item() for o in outputs])
        epe_avg_sqrt = epes_sqrt.sum() / len(epes_sqrt)

        results = {
            "mse": mse,
            "epe_flownet2": epe_avg_flownet2,
            "epe_sqrt": epe_avg_sqrt
        }
        print(f"Test results:")
        pprint(results)