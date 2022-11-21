# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/18 15:21
 @name: 
 @desc:
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.data import DataLoader
from model.ViT_LRP import vit_base_patch16_224
from model.lrp_manager import ignite_relprop, generate_visualization
from data_load.dataset import train_set, val_set

# torch.cuda.set_device(6)
batch_size = 64
n_epoch = 20
total_x = train_set.__len__()
id_experiment = '_1000e03l-delta2'
t_experiment = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=6, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=6, shuffle=True)
vit = vit_base_patch16_224(pretrained=True).cuda()
# ff.load_state_dict(get_state_dict('log/checkpoint/2022-11-04-15-59-42_1000e03l-pre.pkl',
#                                   map_location='cuda:0', exclude=['arc_margin.weight']))
optimizer = torch.optim.AdamW(vit.parameters(), lr=0.002, betas=(0.9, 0.98), eps=1e-9)
# optimizer = NoamOpt(model_size=40, factor=1, warmup=8000,
#                     optimizer=torch.optim.Adam(ff.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# ----- Testing code start ----- Use following to test code without load data -----
from PIL import Image
from data_load.dataset import transform
image = Image.open('data_load/samples/catdog.png')
dog_cat_image = transform(image)

# _x = torch.ones(3, 3, 224, 224).cuda()  # [batch_size, 1, time_step, channels]
_y = torch.tensor([243], dtype=torch.long).cuda()
_x = dog_cat_image.unsqueeze(0).cuda()

optimizer.zero_grad()
_logits = vit(_x)  # [bs, 1000]
print(_logits.shape)
_loss = F.cross_entropy(_logits, _y)
_loss.backward()
optimizer.step()
_cam = ignite_relprop(model=vit, x=_x[0].unsqueeze(0), index=_y[0].unsqueeze(0))  # [1, 3, 224, 224]
print(_cam.shape)
generate_visualization(image, dog_cat_image, _cam.squeeze())
del _x, _y, _logits, _loss
# ----- Testing code end-----------------------------------------------------------

# summary = SummaryWriter(log_dir='./log/'+t_experiment+id_experiment+'/')
# if __name__ == '__main__':
#     step = 0
#     global_step = 0
#     for epoch in range(n_epoch + 1):
#         for x, label in loader:
#             #  [b, 1, 512, 96], [b]
#             if x is None and label is None:
#                 step += 1
#                 global_step += 1
#                 continue
#
#             x = x.cuda()
#             label = label.cuda()
#             ff.train()
#             # if step % 2 == 0:
#             optimizer.zero_grad()  # clean grad per 2 step, to double the batch_size
#
#             y = ff(x, label=None)  # [bs, 40]
#             loss = F.cross_entropy(y, label)
#             loss.backward()
#             optimizer.step()
#             lr = optimizer.param_groups[0]['lr']
#
#             step += 1
#             global_step += 1
#             if step % 1 == 0:
#                 corrects = (torch.argmax(y, dim=1).data == label.data)
#                 accuracy = corrects.cpu().int().sum().numpy() / batch_size
#                 print('epoch:{}/{} step:{}/{} global_step:{} '
#                       'loss={:.5f} acc={:.3f} lr={}'.format(epoch, n_epoch, step, int(total_x / batch_size), global_step,
#                                                             loss, accuracy, lr))
#                 summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
#                 summary.add_scalar(tag='TrainAcc', scalar_value=accuracy, global_step=global_step)
#
#             # if step % 10 == 0:
#             #     cam = ignite_relprop(model=ff, x=x[0].unsqueeze(0), index=label[0])  # [1, 1, 512, 96]
#             #     generate_visualization(x[0].squeeze(), cam.squeeze(),
#             #                            save_name='S' + str(global_step) + '_C' + str(label[0].cpu().numpy()))
#
#         step = 0
#     # torch.save(ff.state_dict(), 'log/checkpoint/' + t_experiment + id_experiment + '.pkl')
#     summary.flush()
#     summary.close()
#     print('done')
