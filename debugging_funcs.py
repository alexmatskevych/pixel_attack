import torch
from DE import load_all_NN, load_imagenet, accuracy, extract_stats
import time

"""
THESE ARE ONLY SOME FUNCTIONS FOR DEBUGGING
"""


def timing(time_before, name):
    print("time for ", name, " :", time.time()- time_before)
    return time.time()

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

def test3():

    import torch
    import torchvision.models as models

    pert_samples, iterations, data_cropped = torch.load("/HDD/advml/optimize_f_crit_F_0_crit_paper.torch")
    data_cropped = torch.load("/HDD/advml/optimize_population_data.torch")
    alexnet = models.squeezenet1_0(pretrained=True)

    acc, pred_right = accuracy(alexnet,data_cropped)


    a = extract_stats(alexnet, data_cropped, pert_samples, pred_right, True)

def test():
    """test for cuda gpu programming"""
    _, _, _, _, _, inception = load_all_NN()

    data_cropped = load_imagenet(5)

    images, targets = data_cropped

    for idx, data in enumerate(images):
        a=inception(data.unsqueeze(0).cuda(cuda4)).squeeze()




    timecopy=time()

    tens=torch.zeros((len(images),3,227,227),device=cuda4)
    for idx,img in enumerate(images):
        tens[idx]=img

    print("Time for copy: ", time() - timecopy, " seconds")
    timebatch=time()


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


def right_targets(data_cropped, pert_samples):
    print("extracting right targets...")

    from copy import deepcopy
    import numpy as np
    print(len(pert_samples))
    data, targets = data_cropped
    true_targets=[]
    true_images=[]
    print("len(data): ", len(data))
    for idx, img in enumerate(pert_samples):
        found=False
        if idx%20==0:
            print("img {} of {}".format(idx, len(pert_samples)))
        for idx_orig, img_orig in enumerate(data):
            nonz = np.nonzero(img-img_orig)
            if len(nonz)==3 or len(nonz)==0:
                true_targets.append(deepcopy(targets[idx_orig]))
                true_images.append(deepcopy(img_orig))
                found=True
                break
        assert (found), "not found img !?"
    print("finished")
    return true_images, true_targets

def final_test():
    import torch
    import numpy as np
    import torchvision.models as models

    cuda=False

    alexnet = models.squeezenet1_0(pretrained=True)

    if cuda:
        print("Using Cuda")
        alexnet.cuda()

    pert_samples, iterations, data_cropped = torch.load(
            "/net/hci-storage02/userfolders/amatskev/pixel_attack/reproduction_init_0_bak.torch")
    data_orig, targets_orig = data_cropped

    pert_samples = [i.cpu() for i in pert_samples]
    score_orig=0
    score_pert=0

    a = torch.stack(data_orig)
    b = torch.stack(pert_samples)
    a_res = alexnet(a).cpu()
    b_res = alexnet(b).cpu()

    print("dimension sanity check")
    print("data_orig[0].unsqueeze(0).size(): ", data_orig[0].unsqueeze(0).size())
    print("pert_samples[0].unsqueeze(0).size(): ", pert_samples[0].unsqueeze(0).size())
    print("alexnet(data_orig[0].unsqueeze(0)).squeeze().size(): ", alexnet(data_orig[0].unsqueeze(0)).squeeze().size())
    print("alexnet(pert_samples[0].unsqueeze(0)).squeeze().size(): ", alexnet(pert_samples[0].unsqueeze(0)).squeeze().size())
    print("torch.stack(data_orig).size(): ",a.size())
    print("torch.stack(pert_samples).size(): ",b.size())

    for idx, orig in enumerate(data_orig):
        orig_res = torch.argmax(alexnet(orig.unsqueeze(0)).squeeze())
        score_orig +=  (orig_res== targets_orig[idx]).detach().numpy()

        pert_res = torch.argmax(alexnet(pert_samples[idx].unsqueeze(0)).squeeze())
        score_pert +=  (pert_res == targets_orig[idx]).detach().numpy()
        print("targets_orig[idx]: ",targets_orig[idx])
        print("orig_res: ", orig_res)
        print("(pert_res == targets_orig[idx]).detach.numpy(): ",(pert_res == targets_orig[idx]).detach().numpy())
        print("pert_res: ", pert_res)
        print("(pert_res == targets_orig[idx]).detach.numpy(): ",(pert_res == targets_orig[idx]).detach().numpy())



    print("score_orig: ", score_orig)
    print("score_pert: ", score_pert)


    a_max=torch.argmax(a_res, dim=1)
    b_max=torch.argmax(b_res, dim=1)
    sum_a=0
    sum_b=0
    for idx,i in enumerate(a_max):
        if a_max[idx]==targets_orig[idx]:
            sum_a+=1
        if b_max[idx] == targets_orig[idx]:
            sum_b += 1



    print("sum_a:", sum_a)
    print("sum_b:", sum_b)
    before=-1
    same_score= 0
    for idx, i in enumerate(data_orig):
        assert(len(np.unique(np.where((data_orig[idx]-pert_samples[idx].cpu()))[1]))==1)

        if before!=-1:
            if np.unique(np.where((data_orig[idx] - pert_samples[idx].cpu()))[1]) != before:
                same_score+=1

        before = np.unique(np.where((data_orig[idx] - pert_samples[idx].cpu()))[1])
    print("same_score: ", same_score)
    print("success!")
    #data = torch.stack(pert_samples)
    #scores_of_pert_images = alexnet(data.cuda()).cpu()




def torch_to_numpy():
    import torch
    import numpy as np
    import os
    import pickle

    folder_torch = "/HDD/advml/only_stats/"
    folder_pickle = "/HDD/advml/only_stats_pickle/"

    things = os.listdir(folder_torch)

    for file in things:

        a = torch.load(folder_torch+file)

        pickle.dump(a, open(folder_pickle+file[:-6]+".pkl", "wb"))

if __name__ == "__main__":
    test()

    pert_samples_smaller, iterations, data_cropped_smaller_true = torch.load("/HDD/advml/debug/NN_and_pixel_NN_resnet_pixel_1_390.torch")
    data_cropped_smaller_true = torch.load("/HDD/advml/debug/NN_and_pixel_NN_resnet_pixel_1_390_right_targets.torch")

    b = right_targets(data_cropped_smaller_true, pert_samples_smaller)
    #torch_to_numpy()

    #a = torch.load("/HDD/advml/testtest/NN_and_pixel_NN_alexnet_pixel_1.torch")
    print("fin")