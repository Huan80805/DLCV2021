import torch
import argparse
import os
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from PIL import Image
from model import VIT_B32, VIT_B16
from torchvision import transforms

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


parser = argparse.ArgumentParser()
parser.add_argument('--img_folder', default='img/', type=str)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--model_name', type=str, default='VIT_B16',choices=['VIT_B16','VIT_B32'])
parser.add_argument('--pos_img', type=str, default='../hw3_data/p1_data/train/0_0.jpg',help='test image for pos embedding')
parser.add_argument('--num_classes', type=int, default=37)
parser.add_argument('--img_size', default=224, type=int, choices=[224,384])
parser.add_argument('--num_heads', default=12, type=int)
parser.add_argument('--dropout_rate', default=0.1, type=float)
parser.add_argument('--attention_dropout_rate', default=0., type=float)
parser.add_argument('--pretrained', default=False, type=bool)
config = parser.parse_args()
os.makedirs(config.img_folder, exist_ok=True)

pos_img = Image.open(config.pos_img).convert("RGB")
pos_out = os.path.join(config.img_folder, "pos embed.png")
att_img_o = [os.path.join('../hw3_data/p1_data/val',img) for img in ['26_5064.jpg','29_4718.jpg','31_4838.jpg']]
att_img_o = [Image.open(img) for img in att_img_o]
att_out = [os.path.join(config.img_folder, "attention{}.png".format(i)) for i in range(3)]
pos_img.save(os.path.join(config.img_folder,'pos img.jpg'))
for i in range(len(att_img_o)):
    transforms.Resize((224,224))(att_img_o[i]).save(os.path.join(config.img_folder, '{}.jpg'.format(i)))
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
])
pos_img = transform(pos_img)
att_img = [transform(img) for img in att_img_o]
#tracking transformer: https://github.com/lukemelas/PyTorch-Pretrained-ViT/blob/master/pytorch_pretrained_vit/transformer.py
if config.model_name == 'VIT_B16': 
    model = VIT_B16(config)
elif config.model_name == 'VIT_B32':
    model = VIT_B32(config)
print(model)
state = torch.load(config.ckpt_path, map_location=torch.device("cpu"))
model.load_state_dict(state['state_dict'])
model.eval()
params = dict()
for name, param in model.model.named_parameters():
    #print(name)
    params[name] = param
pos_emb = params["positional_embedding.pos_embedding"].data
class_token = params["class_token"].data
# print(pos_emb.shape) #1,196,768
# print(class_token.shape) #1,1,768
plt.title("Position Embedding")
fig = plt.figure(figsize=(14,14))
row=1
col=1
for i in range(1, pos_emb.shape[1]):
    sim = F.cosine_similarity(pos_emb[0, i:i+1], pos_emb[0,1:], dim=1)
    sim = sim.reshape((14,14)).numpy()
    ax = fig.add_subplot(14,14,i)
    im = ax.imshow(sim, cmap='viridis')
    if (i%14==1):
        ax.set_ylabel(str(row))
        row += 1
    if ((i-1)//14==13):
        ax.set_xlabel(str(col))
        col += 1
    ax.set_xticks([])
    ax.set_yticks([])
fig.supxlabel('Input patch row', fontsize=20)
fig.supylabel('Input patch row', fontsize=20)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# cb = fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(-1,1), cmap='viridis'),cax=cbar_ax)
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label('Cosine Similarity',fontsize=20)

plt.savefig(pos_out)
plt.close()

vit = model.model
b = 1
for i in range(3):
    img = att_img[i]
    c, fh, fw = img.shape
    img = img.unsqueeze(0)
    img = vit.patch_embedding(img)
    img = img.flatten(2).transpose(1,2)
    img = torch.cat((class_token.expand(b, -1, -1), img), dim=1)  # b,gh*gw+1,d
    img = vit.positional_embedding(img)  # b,gh*gw+1,d 
    # img = vit.transformer(img)  # b,gh*gw+1,d
    for j in range(11):
        img = vit.transformer.blocks[j](img, mask=None)
    img = vit.transformer.blocks[11].norm1(img)
    attn = vit.transformer.blocks[11].attn
    # print(img.shape) (1,197,768)
    # query: style token (1,1,,768)  (1,1*d)
    # note: H = 12(num_heads)
    # split D(dim) into (H(n_heads), W(width of head)) ; D = H * W  
    # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
    q = class_token
    k, v = attn.proj_k(img), attn.proj_v(img)
    q, k, v = (split_last(x, (attn.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
    # print(q.shape, k.shape, v.shape) #(1,12,1,64), (1,12,197,64)
    # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
    scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
    scores = attn.drop(F.softmax(scores, dim=-1))
    # print(scores.shape) # (1,12,1,197)
    scores = scores[:,:,:,1:]
    # average over all heads
    scores = torch.mean(scores,dim=1).squeeze().reshape(14,14).data.numpy()
    attn_map = np.ones((224,224))
    # multiply on patch
    plt.title("Attention map")
    fig=plt.figure(figsize=(14,14))
    img = att_img_o[i].resize((224,224))
    np_img = np.array(img)
    mask = cv2.resize(scores, (224,224))
    mask = mask-np.min(mask)
    mask = np.uint8((mask/np.max(mask))*255)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    # cv2.imwrite('heatmap.png',heatmap)
    # cv2.imwrite('input.png',np_img)
    cam = heatmap*0.7 + np.float32(img)
    cv2.imwrite(att_out[i], cam)







