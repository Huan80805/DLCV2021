wget "https://www.dropbox.com/s/quqhr21oj218zqr/vgg16_final.pth?dl=1" -O part1/final.pth
python3 part1/inference.py --img_dir $1 --output_path $2 --ckpt_path part1/final.pth
