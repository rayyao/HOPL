class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/HOPL/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/HOPL/pretrained/OSTrack_ep0300.pth.tar'
        self.lasher_dir = '/HOPL/data/lasher/train/trainingset'
        self.viptHSItrain_dir = '/HSI2023/train/ALL/'

