wget https://www.dropbox.com/s/uyaymxfc1psmt60/wgan_g.pth?dl=1 -O model/wgan_g.pth
python3 part2/inference.py --output_path $1 --ckpt_path model/wgan_g.pth
