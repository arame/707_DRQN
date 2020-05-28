import torch
from enum import Enum

class DeviceType(Enum):
    cuda = 1
    cpu = 2

class Device:
    type = DeviceType.cpu
    device = None

    @staticmethod
    def get():
        if torch.cuda.is_available():
            Device.type = DeviceType.cuda
            Device.device = torch.device('cuda')
        else:
            Device.type = DeviceType.cpu
            Device.device = torch.device('cpu')
        return Device.device, Device.type
