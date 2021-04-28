import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F
from cutout import Cutout
from models.resnet_cifar import *
from models.preactresnet import *
from models.resnet import *
from models.senet import *
from utils import *
import time


print("RESNET 110")
from dataloader import get_test_loader_cifar,get_train_valid_loader_cifars

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Task-Oriented Feature Distillation. ')
parser.add_argument('--model', default="res8", help="choose the student model", type=str)
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar10/cifar100")
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--beta', default=0.03, type=float)
parser.add_argument('--l2', default=5e-4, type=float)
parser.add_argument('--teacher', default="res110", type=str)
parser.add_argument('--t', default=5.0, type=float, help="temperature for logit distillation ")
args = parser.parse_args()
print(args)

BATCH_SIZE = 128
LR = 0.1
SEED = 30


def reproducible_state(seed =3,device ="cuda"):
   #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = device
    # fix the seed and make cudnn deterministic
    #seed = 3
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("REPRODUCIBILITY is Active!Random seed = ",seed,"\n")



reproducible_state(seed=SEED,device=device)

data_loader_dict, dataset_sizes = get_train_valid_loader_cifars(batch_size=BATCH_SIZE,cifar10_100="cifar100")

trainloader = data_loader_dict["train"]

testloader = get_test_loader_cifar(batch_size=BATCH_SIZE,dataset="cifar100")

#   get the student model
if args.model == "res8":
    net = resnet8_cifar(num_classes=100)
if args.model == "res110":
    net = resnet110_cifar(num_classes=100)
if args.model == "res34":
    net = resnet34()
if args.model == "resnet152":
    net = resnet152(num_classes=100)

    LR = 0.02
    # reduce init lr for stable training
if args.model == "preactresnet50":
    net = preactresnet50()
    LR = 0.02
    # reduce init lr for stable training
if args.model == "senet18":
    net = seresnet18()
if args.model == "senet50":
    net = seresnet50()

#   get the teacher model
if args.teacher == 'res34':
    teacher = resnet34()
elif args.teacher == 'res110':
    teacher = resnet110_cifar(num_classes=100)
elif args.teacher == 'res20':
    teacher = resnet20_cifar(num_classes=100)
elif args.teacher == 'resnet101':
    teacher = resnet101()
elif args.teacher == 'resnet152':
    teacher = resnet152()



teacher_path = "/home/aasadian/tofd/teacher/teacher_res110.pth"
saved_state_dict = torch.load(teacher_path)
testing_state_dict = {}
for (key, value), (key_saved, value_saved) in zip(teacher.state_dict().items(), saved_state_dict.items()):
    testing_state_dict[key] = value_saved
teacher.load_state_dict(testing_state_dict)
teacher.eval()


#teacher.load_state_dict(torch.load("./teacher/" + args.teacher + ".pth"))
#teacher.load_state_dict(torch.load("/home/aasadian/saved/ce/cifar100/res110_cifar100.pth"))
#teacher.cuda()
teacher.to(device)
net.to(device)
orthogonal_penalty = args.beta
init = False
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=args.l2, momentum=0.9)

if __name__ == "__main__":
    best_acc = 0
    print("Start Training")
    since = time.time()
    for epoch in range(250):
        print("Epoch :",epoch)
        if epoch in [80, 160, 240]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, student_feature = net(inputs)

            #   get teacher results
            with torch.no_grad():
                teacher_logits, teacher_feature = teacher(inputs)

            #   init the feature resizing layer depending on the feature size of students and teachers
            #   a fully connected layer is used as feature resizing layer here
            if not init:

                print("True")

                teacher_feature_size = teacher_feature[0].size(1)
                student_feature_size = student_feature[0].size(1)
                num_auxiliary_classifier = len(teacher_logits)
                link = []
                for j in range(num_auxiliary_classifier):
                    link.append(nn.Linear(student_feature_size, teacher_feature_size, bias=False))
                net.link = nn.ModuleList(link)
                #net.cuda()
                net.to(device)
                #   we redefine optimizer here so it can optimize the net.link layers.
                optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)
                init = True

            #   compute loss
            loss = torch.FloatTensor([0.]).to(device)

            #   Distillation Loss + Task Loss
            for index in range(len(student_feature)):
                student_feature[index] = net.link[index](student_feature[index])
                #   task-oriented feature distillation loss
                loss += torch.dist(student_feature[index], teacher_feature[index], p=2) * args.alpha
                #   task loss (cross entropy loss for the classification task)
                loss += criterion(outputs[index], labels)
                #   logit distillation loss, CrossEntropy implemented in utils.py.
                loss += CrossEntropy(outputs[index], teacher_logits[index], 1 + (args.t / 250) * float(1 + epoch))

            # Orthogonal Loss
            for index in range(len(student_feature)):


                #check to apply this loss only for feature resizing layer. In original paper, it has been applied on all the layers
                if student_feature[index].shape != teacher_feature[index].shape:
                    weight = list(net.link[index].parameters())[0]
                    weight_trans = weight.permute(1, 0)
                    #ones = torch.eye(weight.size(0)).cuda()
                    ones = torch.eye(weight.size(0)).to(device)
                    #ones2 = torch.eye(weight.size(1)).cuda()
                    ones2 = torch.eye(weight.size(1)).to(device)
                    loss += torch.dist(torch.mm(weight, weight_trans), ones, p=2) * args.beta
                    loss += torch.dist(torch.mm(weight_trans, weight), ones2, p=2) * args.beta
                #else:
                    #print("No Orthogonal loss!")

            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(labels.size(0))
            _, predicted = torch.max(outputs[0].data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())

            if i % 20 == 0:
                print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.2f%% '
                    % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1),
                     100 * correct / total))

        print("Waiting Test!")
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, feature = net(images)
                _, predicted = torch.max(outputs[0].data, 1)
                correct += float(predicted.eq(labels.data).cpu().sum())
                total += float(labels.size(0))

            print('Test Set AccuracyAcc:  %.4f%% ' % (100 * correct / total))
            if correct / total > best_acc:
                best_acc = correct / total
                print("Best Accuracy Updated: ", best_acc * 100)
                #torch.save(net.state_dict(), "./checkpoint/" + args.model + ".pth")
                torch.save(net.state_dict(), "/home/aasadian/tofd/saved/res8_teacher_res110_"+str(SEED)+ ".pth")
    print("Training Finished, Best Accuracy is %.4f%%" % (best_acc * 100))
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))


