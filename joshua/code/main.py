from pack_images import pack_data 
from output_images import output_images
import torchvision


baseTransform = torchvision.transforms.ToTensor()

transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomRotation(5, expand=False),
                torchvision.transforms.RandomResizedCrop(512, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomErasing(p=0.5, scale=(0.01, 0.03), ratio=(0.3, 3.3), value=0, inplace=True)
            ])

pack_data(baseTransform)
#pack_data(transforms)
output_images()