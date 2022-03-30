stat_each_layer = '0'
beta = '6'

pretrained_file_exit = 'models_new/vgg19_train_exits_update_1.pt'
pretrained_file_normal = 'models_new/vgg19_train_normal_update_1.pt'
evaluate_mode_exit = 'exit'
evaluate_mode_normal = 'normal'

def generate_file_name( baseline_en, scene, core_num=None, cpu_freq_level=None, gpu_freq_level=None, exit_en=None ):
    if baseline_en == 1:
        return f'{scene}_baseline.txt'
    else:
        exit_en = 'Y' if exit_en == 'y' else 'N'
        return f'{scene}_{str(core_num)}_{str(cpu_freq_level)}_{str(gpu_freq_level)}_{exit_en}.txt'

def generate_script( baseline_time=30, sleep_time=30 ):
    '''
    the control parameters:
    1. exit or not
    2. core numbers
    3. cpu frequency
    4. gpu frequency
    5. sleep time
    6. baseline time
    '''
    for scene in ['continuous', 'periodical']:
        print( f"sudo python3 main_new.py --pretrained_file {pretrained_file_normal} --stat_each_layer {stat_each_layer} --evaluate_mode {evaluate_mode_normal} --beta {beta} --baseline 1 --core_num 4 --scene {scene} --baseline_time {str(baseline_time)} --sleep_time {str(sleep_time)} > testcase_result/{generate_file_name( 'y', scene )}" )
        for exit_en in ['y', 'n']:
            for core_num in [2, 4]:
                for cpu_freq_level in [4, 8, 12]:
                    for gpu_freq_level in [2, 5, 8]:
                        pretrained_file = pretrained_file_exit if exit_en=='y' else pretrained_file_normal
                        evaluate_mode = evaluate_mode_exit if exit_en=='y' else evaluate_mode_normal
                        print( f"sudo python3 main_new.py --pretrained_file {pretrained_file} --stat_each_layer {stat_each_layer} --evaluate_mode {evaluate_mode} --beta {beta} --baseline 0 --core_num {str(core_num)} --cpu_freq_level {str(cpu_freq_level)} --gpu_freq_level {str(gpu_freq_level)} --scene {scene} --baseline_time {str(baseline_time)} --sleep_time {str(sleep_time)} > testcase_result/{generate_file_name( 'n', scene, core_num, cpu_freq_level, gpu_freq_level, exit_en )}" )


if __name__ == '__main__':
    for i in range( 100 ):
        print('Accuracy of Type', i, ' ', "{:.2f}".format(0.13333))