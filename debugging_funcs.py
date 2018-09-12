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
    plt.imshow(image.squeeze().permute(1, 2, 0).round().int())
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


def final_test():
    import torch
    import numpy as np

    cuda=False
    if torch.cuda.is_available():
        cuda=True
    _, alexnet, _, _, _, _ = load_all_NN()

    if cuda:
        print("Using Cuda")
        alexnet.cuda()

    pert_samples, iterations, data_cropped = torch.load(
            "/net/hci-storage02/userfolders/amatskev/pixel_attack/reproduction_init_0.torch")
    data_orig, targets_orig = data_cropped

    pert_samples = [i.cpu() for i in pert_samples]

    a = torch.stack(data_orig).cuda()
    b = torch.stack(pert_samples).cuda()
    a_res = alexnet(a).cpu()
    b_res = alexnet(b).cpu()

    a_max=torch.argmax(a_res, dim=1)
    b_max=torch.argmax(b_res, dim=1)
    sum_a=0
    sum_b=0
    for idx,i in enumerate(a_max):
        if a_max[idx]==targets_orig[idx]:
            sum_a+=1
        if b_max[idx] == targets_orig[idx]:
            sum_b += 1



    print(sum_a)
    print(sum_b)

    for idx, i in enumerate(data_orig):
        print(idx)
        assert(len(np.unique(np.where((data_orig[idx]-pert_samples[idx].cpu()))[1]))==1)

    print("success!")
    #data = torch.stack(pert_samples)
    #scores_of_pert_images = alexnet(data.cuda()).cpu()




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

        pickle.dump(a, open(folder_pickle+file[:-6]+".pkl", "wb"))

if __name__ == "__main__":
    final_test()