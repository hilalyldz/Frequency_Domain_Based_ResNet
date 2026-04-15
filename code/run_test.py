import time
import os
import subprocess
import shlex
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch GAN Image Detection')

####################################################################
# Parse command line
####################################################################
parser.add_argument('--dataset', type=str, default='CycleGAN', help='Training dataset select from: CycleGAN and AutoGAN')
parser.add_argument('--feature', default='image', help='Feature used for training, choose from image and fft')

args = parser.parse_args()

gpu_set = ['0']

#Compare image and spectrum
if args.feature == 'image':
    parameter_set = [
            ' --feature=image ',
            ]
elif args.feature == 'fft':
    parameter_set = [
            ' --feature=fft '
            ]
elif args.feature == 'wavelet':
    parameter_set = [
            ' --feature=wavelet '
            ]
else:
    print('Not a valid feature!')
    exit(-1)



number_gpu = len(gpu_set)

# Create timestamped log directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f"./logs_test/run_test_{timestamp}/"
os.makedirs(log_dir, exist_ok=True)

if args.dataset == 'CycleGAN':
    datasets = ['satellite']
elif args.dataset == 'AutoGAN':
    datasets = ['horse_auto', 'zebra_auto', 'summer_auto', 'winter_auto', 'apple_auto', 'orange_auto', 'facades_auto', 'cityscapes_auto', 'satellite_auto', 'ukiyoe_auto', 'vangogh_auto', 'cezanne_auto', 'monet_auto', 'photo_auto']
else:
    print('Not a valid dataset!')
    exit(-1)

process_set = []

def run_command(command, log_file_path):
    """Run a command and both print output and save to a log file."""
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end='')     # print to terminal
            log_file.write(line)    # write to log file
        process.wait()

for dataset in datasets:
    for idx, parameter in enumerate(parameter_set):
        print('Test Parameter: {}'.format(parameter))
        command = 'python ./code/GAN_Detection_Test.py --training-set {} --model=resnet --test-set=transposed_conv --data_augment \
                --batch-size=16 --test-batch-size=16 --epochs 20 {}  --gpu-id {} --model-dir ./model_resnet/ '\
                .format(dataset, parameter, gpu_set[idx%number_gpu]) 
        print(command)

        log_file_path = os.path.join(log_dir, f"{dataset}_{args.feature}_gpu{gpu_set[idx % number_gpu]}.log")

        # Run command and tee output
        run_command(command, log_file_path)


        if (idx+1)%number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()
        
            process_set = []
    
        time.sleep(10)
    
    for sub_process in process_set:
        sub_process.wait()

