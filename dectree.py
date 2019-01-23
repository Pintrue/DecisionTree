import os
import numpy

def load_data(dataset):
    file_name = ''
    if dataset == 'clean':
        file_name = 'clean_dataset.txt'
    elif dataset == 'noisy':
        file_name = 'noisy_dataset.txt'
    file_path = os.path.join('wifi_db', file_name)
    

    # print(file_path)

# load_data('noisy')