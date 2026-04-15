import time
import subprocess
import shlex
import argparse

parser = argparse.ArgumentParser(description='PyTorch GAN Image Detection')

####################################################################
# Parse command line
####################################################################
parser.add_argument('--dataset', type=str, default='CycleGAN', help='Training dataset select from: CycleGAN and AutoGAN')
parser.add_argument('--feature', default='fft', help='Feature used for training, choose from image and fft')
parser.add_argument('--gpu-id', default='0', help='Feature used for training, choose from image and fft')

args = parser.parse_args()

gpu_set = args.gpu_id.split(',')


#Compare image and spectrum
if args.feature == 'image':
    parameter_set = [
            ' --feature=image ',
            ]
elif args.feature == 'fft':
    parameter_set = [
            ' --feature=fft ',
            ]
elif args.feature == 'wavelet':
    parameter_set = ['--feature=wavelet ']
else:
    print('Not a valid feature!')
    exit(-1)


number_gpu = len(gpu_set)
datasets = ['satellite']
process_set = []
index = 0
for idx, parameter in enumerate(parameter_set):
    for dataset in datasets:
        print('Test Parameter: {}'.format(parameter))
        command = 'python ./code/GAN_Detection_Train.py --training-set {} --model=resnet --test-set=transposed_conv --data_augment\
                --batch-size=16 --test-batch-size=16 {} --gpu-id {} --model-dir ./model_resnet/  --log-dir ./resnet_log/ --enable-logging=False --epochs 20 '\
                .format(dataset, parameter, gpu_set[index%number_gpu])# 
    
        print(command)
        p = subprocess.Popen(shlex.split(command))
        process_set.append(p)
         
        if (index+1)%number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()
        
            process_set = []
        
        index+=1
        time.sleep(60)
    
for sub_process in process_set:
    sub_process.wait()

