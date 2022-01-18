import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from PIL import Image
import argparse
from models import caption
from datasets import coco, utils
from configuration import Config
import os
import numpy as np
import cv2
import glob

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--img_dir', type=str, default='../hw3_data/p2_data/images')
parser.add_argument('--output_dir', type=str, default='img/')
args = parser.parse_args()
image_paths = glob.glob(os.path.join(args.img_dir,'*.jpg'))
version = args.v

config = Config()
if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)

@torch.no_grad()
def evaluate(image):
    model.eval()
    # Visualize using attention map for each word
    # after the decoder genrates [EOS]
    # [EOS] appear as [PAD], (padding to create same length)
    for i in range(config.max_position_embeddings - 1):
        #modify caption output
        predictions, attn_weight, src_seqshape = model(image, caption, cap_mask)
        #print(attn_weight.shape) #1,128,361 = N,L,S(batch, tgt seqlen, src seqlen))
        #print(predictions.shape) #1,128,30522 
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        if predicted_id[0] == 102:
            return caption, attn_weight, src_seqshape

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption, attn_weight, src_seqshape

for image_path in image_paths:
    image_o = Image.open(image_path)
    fn = os.path.split(image_path)[1].split('.')[0]
    image = coco.val_transform(image_o)
    image = image.unsqueeze(0)
    output, attn_weight,src_seqshape = evaluate(image)
    attn_weight = attn_weight.numpy()
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
    result = ['<start>', *result.capitalize().split(), '<end>']
    fig=plt.figure(figsize=(14,14))
    for i, text in enumerate(result):
        plt.subplot(len(result)//5+1, 5, i+1)
        plt.text(x=0, y=1, s=str(text), fontsize=20)
        if i==0:
            plt.imshow(image_o)
            plt.axis('off')
        else:
            fh, fw = image_o.size
            # multiply on patch
            # plt.title("Attention map")
            mask = cv2.resize(attn_weight[:,i-1,:].reshape(src_seqshape[0],src_seqshape[1]), (fh, fw))
            mask = mask-np.min(mask)
            mask = np.uint8((mask/np.max(mask))*255)
            heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            # cv2.imwrite('heatmap.png',heatmap)
            # cv2.imwrite('input.png',np_img)
            # cam = heatmap*0.7 + np.float32(image_o)
            plt.imshow(image_o)
            plt.imshow(heatmap,interpolation='none', alpha=0.7)
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, '{}.png'.format(fn)), bbox_inches = 'tight', pad_inches = 0)