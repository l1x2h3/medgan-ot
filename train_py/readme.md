# 默认会自动寻找 data/archive/Testing
python train_py/train_dcgan_from_nb.py --mode train
python train_py/train_dcgan_from_nb.py --mode test

# 如果你的测试集在其他位置，可以指定
python your_script.py --mode test --test_data_path /path/to/your/Testing