import torch
from DE import load_all_NN,load_imagenet,accuracy
from time import time

def de_normalize(image):

    img=image.clone().squeeze()
    img[0] = (img[0] * 0.229 + 0.485)* 255
    img[1] = (img[1] * 0.224 + 0.456)* 255
    img[2] = (img[2] * 0.225 + 0.406)* 255

    return img


def show_img(image):

    import matplotlib.pyplot as plt
    plt.imshow(image.squeeze().permute(1,2,0).round().int())
    plt.show()


def norm_show(image):

    show_img(de_normalize(image))


def test():
    """test for cuda gpu programming"""
    cuda4 = torch.device('cuda:4')
    _, alexnet, _, _, _, _ = load_all_NN()
    alexnet.cuda(cuda4)

    data_cropped = load_imagenet(600)

    images, targets = data_cropped

    timeloop=time()
    for idx, data in enumerate(images):
        a=alexnet(data.unsqueeze(0).cuda(cuda4)).squeeze()


    print("Time for loop: ",time()-timeloop," seconds" )


    timecopy=time()

    tens=torch.zeros((len(images),3,227,227),device=cuda4)
    for idx,img in enumerate(images):
        tens[idx]=img

    print("Time for copy: ", time() - timecopy, " seconds")
    timebatch=time()

    a=alexnet(tens)

    print("Time for  batch: ", time() - timebatch, " seconds")
    print("batchsize: ",tens.size())


def cuda_to_cpu(file):
    output=[]
    for i in file:
        output.append(i.cpu())
    return output

def find_nice_samples(img_size=600):
    import os
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import numpy as np

    resnet152, alexnet, squeezenet, vgg16, densenet, inception = load_all_NN()


    print("loading imagenet_{}".format(img_size))

    # Data loading code
    if torch.cuda.is_available():
        print("using cuda")
        valdir = os.path.join("/net/hci-storage02/userfolders/amatskev/", "val/")
    else:
        valdir = os.path.join("../", "val/")

    # normally, we would use this transformation for our pretrained nets
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # use this transformation to have RGB values in [0, 255], transform as above later on
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    print("Loaded")

    while True:
        # create subset with only img_size images
        numbers = np.arange(len(val_dataset))
        indices = np.random.choice(numbers, img_size, replace=False)

        # no dataloader, we need shuffled samples but after that constant drawing from list (either this or update to pytorch 0.4.1)
        val_images = [val_dataset[i][0] for i in indices]
        val_targets = [val_dataset[i][1] for i in indices]

        accuracy_score, pred_right = accuracy(alexnet,(val_images,val_targets))

        if accuracy_score > 0.55:
            torch.save((val_images,val_targets), "../imagenet_nice_acc_{}.torch".format(img_size))
            break

def test2():
    data,targets = load_imagenet(600)

    for idx,d in enumerate(data):
        show_img(d)
        print(targets[idx])
        print(targets[idx])

def torch_to_numpy():
    import torch
    import numpy as np
    import os
    import pickle

    folder_torch = "/HDD/advml/pixel_attack_stats/only_stats/"
    folder_pickle = "/HDD/advml/pixel_attack_stats/pickle/"

    things = os.listdir(folder_torch)

    for file in things:

        a = torch.load(folder_torch+file)

        b = pickle.load(open(folder_pickle+file[:-6]+".pkl", "rb"))

if __name__ == "__main__":

    torch_to_numpy()