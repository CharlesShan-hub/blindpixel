import click
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from config import TestOptions
from dataset import BlindPointINO
from transform import transform
from model import FastBlindNet
from clib.inference import BaseInferencer
from clib.utils import glance


class FastBlindNetTester(BaseInferencer):
    def __init__(self, opts):
        super().__init__(opts)

        self.model = FastBlindNet().to(opts.device)
        self.load_checkpoint()

        self.criterion = torch.nn.MSELoss()

        self.transform = transform(self.opts.width,self.opts.height)

        dataset = BlindPointINO(root=opts.dataset_path, download=True, transform=self.transform)
        val_size = int(opts.val * len(dataset))
        test_size = int(opts.val * len(dataset))
        train_size = len(dataset) - val_size - test_size
        _, _, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(opts.seed),
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            worker_init_fn=self.seed_worker,
            generator=self.g,
        )
    
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
                # glance([noisy[0,:,:,:],clean[0,:,:,:],outputs[0,:,:,:]],title=['noisy','clean','output'])
                loss = self.criterion(outputs, clean)
                running_loss += loss.item() * clean.size(0)
                total_samples += clean.size(0)
                pbar.set_description(f"Test Epoch {batch_index}")
                pbar.set_postfix(loss=(running_loss.item() / total_samples))

        test_loss = running_loss / total_samples

        print(f"Loss of the model on the {total_samples} test images: {test_loss:.4f}")


@click.command()
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--dataset_path", type=click.Path(exists=True), required=True)
@click.option("--height", type=int, default=254, show_default=True)
@click.option("--width", type=int, default=328, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True, required=False)
@click.option("--batch_size", type=int, default=8, show_default=True, required=False)
@click.option("--val", type=float, default=0.1, show_default=True, required=False)
@click.option("--test", type=float, default=0.1, show_default=True, required=False)
@click.option("--comment", type=str, default="", show_default=False)
def test(**kwargs):
    opts = TestOptions().parse(kwargs,present=True)
    tester = FastBlindNetTester(opts)
    tester.test()


if __name__ == "__main__":
    test()
