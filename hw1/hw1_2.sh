wget 'https://www.dropbox.com/s/0h253n6d8u29dmi/FCN8_final.pth?dl=1' -O part2/FCN8_final.pth
python3 part2/inference.py --img_dir $1 --output_dir $2 --ckpt_path part2/FCN8_final.pth
