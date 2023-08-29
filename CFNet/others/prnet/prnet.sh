python main.py --exp_name 'um' --gpu 1

python main.py --exp_name 'uc'  --unseen True --gpu 1

python main.py --exp_name 'um_noise'  --noise True --gpu 1

python main_outdoor.py --exp_name 'kitti_tracking_um' --type 'tracking' --gpu 1

python main_outdoor.py --exp_name 'kitti_object_um'  --type 'object' --gpu 1
