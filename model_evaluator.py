import torch
from models.resnet_cifar import resnet110_cifar
from dataloader import get_test_loader_cifar
from general_utils import test_data_evaluation


DATASET = "cifar100"
#seeds = [30,50,67]
seeds = [30]
BATCH_SIZE = 128
SPATIAL_SIZE = 32
SERVER = 1

if DATASET == "cifar10":
    NUM_CLASSES = 10
    test_loader = get_test_loader_cifar(batch_size=BATCH_SIZE,
                                        dataset=DATASET)

elif DATASET == "cifar100":
    NUM_CLASSES = 100
    test_loader = get_test_loader_cifar(batch_size=BATCH_SIZE,
                                        dataset=DATASET)

elif DATASET == "tiny":
    NUM_CLASSES = 200
    SPATIAL_SIZE = 64

    tiny = Tiny(batch_size=BATCH_SIZE,server=SERVER)
    data_loader_dict, dataset_sizes = tiny.data_loader_dict, tiny.dataset_sizes
    test_loader = tiny.data_loader_dict["val"]



for seed in seeds:
    model = resnet110_cifar(seed=seed,num_classes=NUM_CLASSES)
    #from torchsummary import summary
    #summary(model, input_size=(3,SPATIAL_SIZE,SPATIAL_SIZE),device="cpu")
    print("SEED ===> ",str(seed),"\t DATASET ===> ",DATASET)

    saved_path = "/home/aasadian/tofd/teacherteacher_res110.pth"


    full_modules_state_dict = {}
    saved_state_dict = torch.load(saved_path)
    testing_state_dict = {}
    for (key, value), (key_saved, value_saved) in zip(model.state_dict().items(), saved_state_dict.items()):
        testing_state_dict[key] = value_saved
        full_modules_state_dict[ key] = value_saved
    model.load_state_dict(testing_state_dict)
    model.eval()

    test_data_evaluation(model=model,
                         test_loader=test_loader,
                         state_dict=full_modules_state_dict,  # testing_state_dict,#testing_state_dict,
                         saved_load_state_path=None,  # path_1,
                         show_log=False,
                         device="cuda:2")

