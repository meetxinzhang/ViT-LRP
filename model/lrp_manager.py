"""Forked from https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/helpers.py"""

from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention


def ignite_relprop(model, x, index=None, method="transformer_attribution", start_layer=0):
    model.eval()
    logits = model(x)  # [b, c, h, w] -> [b, classes]
    kwargs = {"alpha": 1}
    if index is None:  # classificatory index
        index = np.argmax(logits.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, logits.size()[-1]), dtype=np.float32)  # [1, classes]
    one_hot[0, index] = 1  # [1, classes]
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits)  # classificatory mask, if b=1 then [1, classes]
    # TODO auto-validate per batch

    model.zero_grad()
    one_hot.backward(retain_graph=True)  # generate partial-gradients

    # the input of model.relprop() is one_hot
    return model.relprop(cam=torch.tensor(one_hot_vector).to(x.device), method=method,
                         start_layer=start_layer, **kwargs).detach()


def generate_visualization(img, x, cam, save_name=None):
    # image = Image.fromarray(x)
    # image = transform(image)
    cam = cam.reshape(1, 1, 14, 14)
    # torch.nn.functional.interpolate up-sampling, to re
    cam = torch.nn.functional.interpolate(cam, scale_factor=16, mode='bilinear')
    cam = cam.reshape(224, 224).data.cpu().numpy()
    # permute: trans dimension at original image.permute(1, 2, 0)
    x = x.permute(1, 2, 0).data.cpu().numpy()

    # image + attribution
    heatmap, vis = add_cam_on_image(x, cam)

    if save_name is not None:
        path = './log/image/' + save_name

        vis = Image.fromarray(vis)
        vis.save(path + '_cam.jpg')

        x = Image.fromarray(x)
        x.save(path + '.jpg')

        heatmap = Image.fromarray(heatmap)
        heatmap.save(path + '_heatmap.jpg')

        print('saved ' + save_name)
    else:
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(img)
        axs[0].axis('off')
        axs[1].imshow(heatmap)
        axs[1].axis('off')
        axs[2].imshow(vis)
        axs[2].axis('off')
        plt.show()


def add_cam_on_image(img, cam):
    print(np.shape(img), np.shape(cam))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    img = (img - img.min()) / (img.max() - img.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # img = cv2.cvtColor(np.array(np.uint8(255 * x)), cv2.COLOR_RGB2BGR)

    vis = np.float32(heatmap) / 255 + np.float32(img)
    vis = vis / np.max(vis)
    vis = cv2.cvtColor(np.array(np.uint8(255 * vis)), cv2.COLOR_RGB2BGR)

    del cam
    return heatmap, vis


