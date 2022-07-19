import os
import argparse
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.bernoulli import Bernoulli

from Utils.utils import set_logging_settings, add_common_args
from dataset.ImageNet import ImageNet_data
from dataset.ImageNetLabels import ImageNet_labels
from Models.models import RealTimeSaliencyRBF

parser = argparse.ArgumentParser()
parser.add_argument('--random_seed', default=6)
parser.add_argument('--verbose', default=1)
parser.add_argument('--num_im_show', default=6)

# Data Parameters
parser.add_argument('--train_fraction', default=0.1)
parser.add_argument('--val_fraction', default=0.02)
parser.add_argument('--batch_size', default=1)
parser.add_argument('--debug', default=False)
parser.add_argument('--min_crop_scale', default=0.5, help='controls the min area of the original image to crop. '
                                                          'Default value for Pytorch is 0.08, but its too small. '
                                                          'It can remove the object which corresponds to the label.')

# Model Parameters
parser.add_argument('--model_type', default='resnet-50', help='resnet-34, resnet-50, resnet-101, MobileNetV2')
parser.add_argument('--load_saliency_encoder_from_pretrained', default=True)
parser.add_argument('--initial_sigma', default=False, help='effective when "selector_type" is gaussian.')
parser.add_argument('--load_saliency_encoder_from_checkpoint', default=False)
parser.add_argument('--fix_encoder', default=True)

parser = add_common_args(parser)
args = parser.parse_args()
args = set_logging_settings(args, os.path.basename(__file__).split('.')[0])

# ############################# Loading Data #############################
data = ImageNet_data(args)
data.prepare_data()
data.setup()
train_dataloader, test_dataloader = data.train_dataloader(), data.test_dataloader()

# ############################# Defining and Loading the Model #############################
if args.model_type == 'resnet-34':
    args.pretrained_model_dir = os.path.join('./checkpoints/resnet-34/selector', 'class_checkpoint_iter_1200.pth')
elif args.model_type == 'resnet-50':
    args.pretrained_model_dir = os.path.join('./checkpoints/resnet-50/selector', 'checkpoint_iter_9200.pth')
elif args.model_type == 'resnet-101':
    args.pretrained_model_dir = os.path.join('./checkpoints/resnet-101/selector', 'checkpoint_iter_2300.pth')
elif args.model_type == 'MobileNetV2':
    args.pretrained_model_dir = os.path.join('./checkpoints/mobilenetv2/selector', 'class_checkpoint_iter_2600.pth')
else:
    raise ValueError

model = RealTimeSaliencyRBF(args)
model.load_checkpoint(args.pretrained_model_dir)
model.eval()

# ############################# Visualization #############################
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)

classes = ImageNet_labels
n = args.num_im_show
test_images = []

for i, batch in enumerate(test_dataloader):
    if i == n:
        break
    sample_list = []
    x_test, y_test = batch[0], batch[1]
    print(classes[y_test.item()])
    sample_list.append(x_test[0])
    with torch.no_grad():
        mask, params = model(x_test, y_test)
    binary_mask = torch.clone(mask)
    sampler = Bernoulli(probs=binary_mask)
    binary_mask = sampler.sample()
    sample_list.append(mask[0])
    sample_list.append((x_test * mask)[0])
    sample_list.append(binary_mask[0])
    sample_list.append((x_test * binary_mask)[0])
    test_images.append(sample_list)

fig = plt.figure(figsize=(25, n * 5))
for j in range(n):
    for k in range(5):
        axes1 = fig.add_subplot(n, 5, 5 * j + k + 1)
        if (k != 1) and (k != 3):
            axes1.imshow(test_images[j][k].permute(1, 2, 0) / 5. + 0.5)
        else:
            axes1.imshow(test_images[j][k].squeeze(), cmap='gray')
        axes1.set_axis_off()
fig.subplots_adjust(wspace=0.025, hspace=0.05)
plt.savefig(os.path.join(args.stdout, 'test_images.pdf'), bbox_inches="tight")
plt.show()
plt.close()
