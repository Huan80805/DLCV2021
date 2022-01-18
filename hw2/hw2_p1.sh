#wget "https://www.dropbox.com/s/quqhr21oj218zqr/vgg16_final.pth?dl=1" -O part1/final.pth
python3 part1/inference.py --output_path $1 --ckpt_path model/dcgan_g.pth
