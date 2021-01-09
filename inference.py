import torch
from PIL import Image, ImageDraw
from dataset import PennFudanDataset, get_transform
from models.MaskRCNN import get_model_instance_segmentation

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2

# Loading Model for Inference
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load("dict.pth"))
model.to(device)
model.eval()

dataset = PennFudanDataset('PennFudanPed', get_transform(train=False))

img, _ = dataset[0]

with torch.no_grad():
    prediction = model([img.to(device)])

im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
draw = ImageDraw.Draw(im)
draw.rectangle(prediction[0]['boxes'][0].cpu().numpy())
im.save("sample.bmp")
