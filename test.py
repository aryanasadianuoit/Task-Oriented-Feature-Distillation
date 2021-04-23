import torch
from torchsummary import summary
from models.resnet import *


teacher = resnet18()

teacher.load_state_dict(torch.load("/home/aasadian/tofd/teacher/resnet18.pth"))


summary(teacher,input_size=(3,32,32),device="cpu")


