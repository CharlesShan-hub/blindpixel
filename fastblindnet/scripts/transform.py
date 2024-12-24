from torchvision.transforms import Compose, Resize, ToTensor

def transform(width,height):
    return Compose([
        Resize((width,height)), 
        ToTensor()
    ])
