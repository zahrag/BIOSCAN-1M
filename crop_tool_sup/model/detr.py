import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch
import io

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, train_dataloader=None, val_dataloader=None):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            num_labels=1,
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.train_dataloader_ = train_dataloader
        self.val_dataloader_ = val_dataloader

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return self.val_dataloader_


def load_model_from_ckpt(args):

    original = False
    if original:
        model = Detr.load_from_checkpoint(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
                                          checkpoint_path=args.checkpoint_path)
    else:
        # Load the checkpoint into a buffer
        with open(args.checkpoint_path, 'rb') as f:
            buffer = io.BytesIO(f.read())

        # Load the model from the buffer
        model = Detr.load_from_checkpoint(
            lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4,
            checkpoint_path=buffer,  # Pass the buffer instead of the file path
            map_location=torch.device('cpu'),
        )

    model.eval()
    return model
