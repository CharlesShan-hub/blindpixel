import click
from tqdm import tqdm
import kornia
from kornia.losses import ssim_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transform import transform
from dataset import BlindPointINO
from model import FastBlindNet
from config import TrainOptions
from clib.train import BaseTrainer
# from clib.utils import glance

class FastBlindNetTrainer(BaseTrainer):
    def __init__(self, opts):
        super().__init__(opts)

        self.model = FastBlindNet()

        self.criterion = nn.MSELoss()

        self.opts.optimizer = "SGD"
        self.optimizer = optim.SGD(
            params=self.model.parameters(), 
            lr=opts.lr
        )

        self.opts.lr_scheduler = "ReduceLROnPlateau"
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, 
            mode='min', 
            factor=opts.factor, 
            patience=2
        )

        self.transform = transform(self.opts.width,self.opts.height)

        dataset = BlindPointINO(root=opts.dataset_path, download=True, transform=self.transform)
        val_size = int(opts.val * len(dataset))
        test_size = int(opts.val * len(dataset))
        train_size = len(dataset) - val_size - test_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(opts.seed),
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )

    def holdout_train(self, epoch):
        assert self.optimizer is not None
        assert self.model is not None
        assert self.criterion is not None
        assert self.train_loader is not None
        pbar = tqdm(self.train_loader, total=len(self.train_loader))
        running_loss = torch.tensor(0.0).to(self.opts.device)
        total_samples = 0
        self.model.train()
        for noisy, clean, _ in pbar:
            noisy = noisy.to(self.opts.device)
            clean = clean.to(self.opts.device)
            self.optimizer.zero_grad()
            outputs = self.model(noisy)
            loss = self.criterion(outputs, clean)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * clean.size(0)
            total_samples += clean.size(0)
            pbar.set_description(
                f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]"
            )
            pbar.set_postfix(loss=(running_loss.item() / total_samples))

        train_loss = running_loss / total_samples

        self.loss = loss
        self.writer.add_scalar("lr", self.get_lr(), epoch)
        self.writer.add_scalar("Loss/train", train_loss, epoch)

        print(f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]", \
              f"Train Loss: {train_loss:.4f}")
        
        return train_loss

    def holdout_validate(self,epoch):
        assert self.model is not None
        assert self.criterion is not None
        assert self.val_loader is not None
        pbar = tqdm(self.val_loader, total=len(self.val_loader))
        running_loss = torch.tensor(0.0).to(self.opts.device)
        total_samples = 0
        self.model.eval()
        for noisy, clean, _ in pbar:
            noisy = noisy.to(self.opts.device)
            clean = clean.to(self.opts.device)
            outputs = self.model(noisy)
            loss = self.criterion(outputs, clean)
            running_loss += loss.item() * clean.size(0)
            total_samples += clean.size(0)
            pbar.set_description(
                f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]"
            )
            pbar.set_postfix(loss=(running_loss.item() / total_samples))

        val_loss = running_loss / total_samples

        assert isinstance(self.scheduler, ReduceLROnPlateau)
        self.scheduler.step(metrics=val_loss)

        self.writer.add_scalar("lr", self.get_lr(), epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]", \
              f"Train Loss: {val_loss:.4f}")
        
        return val_loss
    
    def test(self):
        assert self.model is not None
        assert self.criterion is not None
        assert self.test_loader is not None
        pbar = tqdm(self.test_loader, total=len(self.test_loader))
        running_loss = torch.tensor(0.0).to(self.opts.device)
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for batch_index, (noisy, clean, _) in enumerate(pbar, start=1):
                noisy = noisy.to(self.opts.device)
                clean = clean.to(self.opts.device)
                outputs = self.model(noisy)
                loss = self.criterion(outputs, clean)
                running_loss += loss.item() * clean.size(0)
                total_samples += clean.size(0)
                pbar.set_description(f"Test Epoch {batch_index}")
                pbar.set_postfix(loss=(running_loss.item() / total_samples))

        test_loss = running_loss / total_samples

        print(f"Loss of the model on the {total_samples} test images: {test_loss:.4f}")

        return test_loss


class FastBlindNetMaskTrainer(BaseTrainer):
    def __init__(self, opts):
        super().__init__(opts)

        self.model = FastBlindNet(use_mask=self.opts.use_mask)

        self.criterion = nn.MSELoss()

        self.opts.optimizer = "SGD"
        self.optimizer = optim.SGD(
            params=self.model.parameters(), 
            lr=opts.lr
        )

        self.opts.lr_scheduler = "ReduceLROnPlateau"
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, 
            mode='min', 
            factor=opts.factor, 
            patience=2
        )

        self.transform = transform(self.opts.width,self.opts.height)

        dataset = BlindPointINO(root=opts.dataset_path, download=True, transform=self.transform)
        val_size = int(opts.val * len(dataset))
        test_size = int(opts.val * len(dataset))
        train_size = len(dataset) - val_size - test_size
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(opts.seed),
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )

    def holdout_train(self, epoch):
        assert self.optimizer is not None
        assert self.model is not None
        assert self.criterion is not None
        assert self.train_loader is not None
        pbar = tqdm(self.train_loader, total=len(self.train_loader))
        running_loss = torch.tensor(0.0).to(self.opts.device)
        total_samples = 0
        self.model.train()
        for noisy, clean, mask in pbar:
            # glance(mask)
            noisy = noisy.to(self.opts.device)
            clean = clean.to(self.opts.device)
            mask = mask.to(self.opts.device)
            grad_clean_x = kornia.filters.filter2d(clean,torch.tensor([[[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]]]),border_type='replicate')
            grad_clean_y = kornia.filters.filter2d(clean,torch.tensor([[[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]]]),border_type='replicate')
            self.optimizer.zero_grad()
            if self.opts.use_mask:
                outputs = self.model(torch.cat((noisy, mask), dim=1))
            else:
                outputs = self.model(noisy)
            grad_out_x = kornia.filters.filter2d(outputs,torch.tensor([[[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]]]),border_type='replicate')
            grad_out_y = kornia.filters.filter2d(outputs,torch.tensor([[[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]]]),border_type='replicate')
            loss_overview = ssim_loss(outputs,clean,7)
            loss_blind = self.criterion(outputs[mask==1.0], clean[mask==1.0]) / (mask.sum()) * (self.opts.width * self.opts.height)
            loss_grad = self.criterion((grad_clean_x**2 + grad_clean_y**2)[mask==1.0], (grad_out_x**2 + grad_out_y**2)[mask==1.0])
            loss = loss_blind + loss_overview + loss_grad
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * clean.size(0)
            total_samples += clean.size(0)
            pbar.set_description(
                f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]"
            )
            pbar.set_postfix(loss=(running_loss.item() / total_samples))

        train_loss = running_loss / total_samples

        self.loss = loss
        self.writer.add_scalar("lr", self.get_lr(), epoch)
        self.writer.add_scalar("Loss/train", train_loss, epoch)

        print(f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]", \
              f"Train Loss: {train_loss:.4f}")
        
        return train_loss

    def holdout_validate(self,epoch):
        assert self.model is not None
        assert self.criterion is not None
        assert self.val_loader is not None
        pbar = tqdm(self.val_loader, total=len(self.val_loader))
        running_loss = torch.tensor(0.0).to(self.opts.device)
        total_samples = 0
        self.model.eval()
        for noisy, clean, mask in pbar:
            noisy = noisy.to(self.opts.device)
            clean = clean.to(self.opts.device)
            mask = mask.to(self.opts.device)
            grad_clean_x = kornia.filters.filter2d(clean,torch.tensor([[[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]]]),border_type='replicate')
            grad_clean_y = kornia.filters.filter2d(clean,torch.tensor([[[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]]]),border_type='replicate')
            if self.opts.use_mask:
                outputs = self.model(torch.cat((noisy, mask), dim=1))
            else:
                outputs = self.model(noisy)
            grad_out_x = kornia.filters.filter2d(outputs,torch.tensor([[[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]]]),border_type='replicate')
            grad_out_y = kornia.filters.filter2d(outputs,torch.tensor([[[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]]]),border_type='replicate')
            loss_overview = ssim_loss(outputs,clean,7)
            loss_blind = self.criterion(outputs[mask==1.0], clean[mask==1.0]) / (mask.sum()) * (self.opts.width * self.opts.height)
            loss_grad = self.criterion((grad_clean_x**2 + grad_clean_y**2)[mask==1.0], (grad_out_x**2 + grad_out_y**2)[mask==1.0])
            loss = loss_blind + loss_overview + loss_grad
            running_loss += loss.item() * clean.size(0)
            total_samples += clean.size(0)
            pbar.set_description(
                f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]"
            )
            pbar.set_postfix(loss=(running_loss.item() / total_samples))

        val_loss = running_loss / total_samples

        assert isinstance(self.scheduler, ReduceLROnPlateau)
        self.scheduler.step(metrics=val_loss)

        self.writer.add_scalar("lr", self.get_lr(), epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"Epoch [{epoch}/{self.opts.max_epoch if self.opts.max_epoch != 0 else '∞'}]", \
              f"Train Loss: {val_loss:.4f}")
        
        return val_loss
    
    def test(self):
        assert self.model is not None
        assert self.criterion is not None
        assert self.test_loader is not None
        pbar = tqdm(self.test_loader, total=len(self.test_loader))
        running_loss = torch.tensor(0.0).to(self.opts.device)
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for batch_index, (noisy, clean, mask) in enumerate(pbar, start=1):
                noisy = noisy.to(self.opts.device)
                clean = clean.to(self.opts.device)
                mask = mask.to(self.opts.device)
                grad_clean_x = kornia.filters.filter2d(clean,torch.tensor([[[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]]]),border_type='replicate')
                grad_clean_y = kornia.filters.filter2d(clean,torch.tensor([[[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]]]),border_type='replicate')
                outputs = self.model(torch.cat((noisy, mask), dim=1))
                grad_out_x = kornia.filters.filter2d(outputs,torch.tensor([[[ 1,  2,  1],[ 0,  0,  0],[-1, -2, -1]]]),border_type='replicate')
                grad_out_y = kornia.filters.filter2d(outputs,torch.tensor([[[ 1,  0, -1],[ 2,  0, -2],[ 1,  0, -1]]]),border_type='replicate')
                if self.opts.use_mask:
                    outputs = self.model(torch.cat((noisy, mask), dim=1))
                else:
                    outputs = self.model(noisy)
                loss_overview = ssim_loss(outputs,clean,7)
                loss_blind = self.criterion(outputs[mask==1.0], clean[mask==1.0]) / (mask.sum()) * (self.opts.width * self.opts.height)
                loss_grad = self.criterion((grad_clean_x**2 + grad_clean_y**2)[mask==1.0], (grad_out_x**2 + grad_out_y**2)[mask==1.0])
                loss = loss_blind + loss_overview + loss_grad
                running_loss += loss.item() * clean.size(0)
                total_samples += clean.size(0)
                pbar.set_description(f"Test Epoch {batch_index}")
                pbar.set_postfix(loss=(running_loss.item() / total_samples))

        test_loss = running_loss / total_samples

        print(f"Loss of the model on the {total_samples} test images: {test_loss:.4f}")

        return test_loss



@click.command()
@click.option("--comment", type=str, default="", show_default=False)
@click.option("--model_base_path", type=click.Path(exists=True), required=True)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--height", type=int, default=254, show_default=True)
@click.option("--width", type=int, default=328, show_default=True)
@click.option("--use_mask", type=bool, default=True, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True, required=False)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--lr", type=float, default=0.03, show_default=True, required=False)
@click.option("--max_epoch", type=int, default=100, show_default=True, required=False)
@click.option("--max_reduce", type=int, default=6, show_default=True, required=False)
@click.option("--factor", type=float, default=0.1, show_default=True, required=False)
@click.option("--train_mode", type=str, default="Holdout", show_default=False)
@click.option("--val", type=float, default=0.1, show_default=True, required=False)
@click.option("--test", type=float, default=0.1, show_default=True, required=False)
def train(**kwargs):
    opts = TrainOptions().parse(kwargs)
    # trainer = FastBlindNetTrainer(opts)
    trainer = FastBlindNetMaskTrainer(opts)
    trainer.train()


if __name__ == "__main__":
    train()
