import yaml
import torch

CONF_YML_PATH = './util/conf.yml'

def get_yaml_data(yaml_file):
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    
    data = yaml.load(file_data, Loader=yaml.FullLoader)
    return data

CONF = get_yaml_data(CONF_YML_PATH)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model parameter setting
INPUT_CHANNELS = CONF['model']['input_channels']
HIDDEN_CHANNELS = CONF['model']['hidden_channels']
OUTPUT_CHANNELS = CONF['model']['output_channels']
INPUT_INDICES = CONF['model']['input_indices']
MODEL = CONF['model']['name']

# training parameter setting
BATCH_SIZE = CONF['training']['batch_size']
EPOCH = CONF['training']['epoch']
TEST_SIZE = CONF['training']['test_size']
CLIP = CONF['training']['clip']
INF = CONF['training']['inf']
BETA = CONF['training']['beta']
SEED = CONF['training']['seed']

# optimizer parameter setting
INIT_LR = CONF['optimizer']['init_lr']
WEIGHT_DECAY = CONF['optimizer']['weight_decay']
ADAM_EPS = CONF['optimizer']['adam_eps']

# LR-Scheduler parameter setting
FACTOR = CONF['lr_scheduler']['factor']
PATIENCE = CONF['lr_scheduler']['patience']
WARMUP = CONF['lr_scheduler']['warmup']

# dateset setting
DIRECTORY = CONF['dataset']['directory']
DATASET = CONF['dataset']['name']

# date setting
NUMBER = CONF['No.']

