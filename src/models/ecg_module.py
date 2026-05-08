import torch
import torch.nn as nn
import lightning as L
import torchmetrics

class ECGLitModule(L.LightningModule):
    def __init__(self, net: nn.Module, num_classes: int = 2, learning_rate: float = 1e-3, compile: bool = False):
        super().__init__()
        # Ignore net to avoid saving entire model architecture in hyperparameters
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.learning_rate = learning_rate
        self.net = net
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Exactly match the original metrics implementation
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        
        # Thêm F1 score (Tính riêng cho nhãn MI - đảo ngược class 0 thành 1)
        self.val_f1 = torchmetrics.classification.BinaryF1Score()
        self.test_f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        # Log exact name used in original code
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)   
        # Đảo ngược 0 (MI) thành 1 để tính Binary F1 cho MI
        self.val_f1.update(1 - preds, 1 - y)
        # Log exact names and params used in original code
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        # Đảo ngược 0 (MI) thành 1 để tính Binary F1 cho MI
        self.test_f1.update(1 - preds, 1 - y)
        # Log exact names and params used in original code
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        # Match original exactly
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs, 
            eta_min=self.learning_rate/100
        )
        return [optimizer], [scheduler]
