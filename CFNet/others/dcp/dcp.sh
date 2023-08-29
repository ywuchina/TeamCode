python main.py --exp_name 'um' --batch_size 10

python main.py --exp_name 'uc'  --unseen True --batch_size 10

python main.py --exp_name 'um_noise'  --noise True --batch_size 10

python main_outdoor.py --exp_name 'kitti_tracking_um' --type 'tracking' --batch_size 10

python main_outdoor.py --exp_name 'kitti_object_um'  --type 'object' --batch_size 10
