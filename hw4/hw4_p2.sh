# TODO: create shell script for running your data hallucination model

# Example
python3 p2/test.py --test_csv $1 --test_data_dir $2 --output_csv $3 \
--ckpt_path c_epoch57.pth --num2raw p2/num2raw.npy
