import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn

from data_loader.msrs_data import MSRS_data
from models.Common import YCrCb2RGB, clamp
from models.Fusion import MBHFuse
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()
def init_seeds(seed=0):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 修改的部分

    parser = argparse.ArgumentParser(description='PyTorch MBHFuse')
    parser.add_argument('--dataset', default='LR_RoadScene',
                        help='dataset')
    parser.add_argument('--dataset_path', metavar='DIR', default=r'/shares/image_fusion/IVIF_datasets/test',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='/home/zenghui/experiment/MBHFuse/results/ablation/module/wo_lfa/99/')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--fusion_pretrained', default='/home/zenghui/experiment/MBHFuse/trained_model/ablation/2ci/wo_lfa/model_epoch_99.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=3407, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)

    dataset_path = args.dataset_path + f"/test_{args.dataset}"

    test_dataset = MSRS_data(dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    save_path = args.save_path + args.dataset

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.arch == 'fusion_model':
        model = MBHFuse()
        x = torch.randn(1, 1, 40, 80).cuda()
        y = torch.randn(1, 1, 40, 80).cuda()
        # model = MBHFuse()
        model = model.cuda()
        total_params = sum(p.numel() for p in model.parameters())
        print("Params(M): %.3f" % (params_count(model) / (1000 ** 2)))

        flops = FlopCountAnalysis(model, (x, y))
        print("FLOPs(G): %.3f" % (flops.total()/1e9))


        if args.cuda and torch.cuda.device_count() > 1:
             model = nn.DataParallel(model)

        model.load_state_dict(torch.load(args.fusion_pretrained))
        model.eval()

        # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        test_tqdm = tqdm(test_loader, total=len(test_loader))
        with torch.no_grad():
            for vis_image, vis_y_image, cb, cr, inf_image, name in test_tqdm:
                vis_y_image = vis_y_image.cuda()
                vis_image = vis_image.cuda()
                cb = cb.cuda()
                cr = cr.cuda()
                inf_image = inf_image.cuda()

                if args.dataset == "TNO":
                    fused_image = model(vis_image, inf_image)
                    rgb_fused_image = clamp(fused_image) 
                    rgb_fused_image = rgb_fused_image.squeeze(1)   
                else:
                    fused_image = model(vis_y_image, inf_image)
                    fused_image = clamp(fused_image)
                    rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])

                rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
                rgb_fused_image.save(f'{save_path}/{name[0]}')
