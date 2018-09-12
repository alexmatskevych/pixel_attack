import numpy as np
from copy import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from time import time
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#debugging
#from debugging_funcs import de_normalize, show_img, norm_show


def load_all_NN():

    import torchvision.models as models

    resnet152 = models.resnet152(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet161(pretrained=True)
    inception = models.inception_v3(pretrained=True)

    return resnet152,alexnet,squeezenet,vgg16,densenet,inception

def load_imagenet(img_size=600,clear=False):
    import os
    import torchvision.datasets as datasets

    print("loading imagenet_{}".format(img_size))

    #load random images
    if os.path.exists("../imagenet_{}.torch".format(img_size)) and not clear:
        print("imagedata already exists...")
        return torch.load("../imagenet_{}.torch".format(img_size))

    # Data loading code
    if torch.cuda.is_available():
        print("using cuda")
        valdir = os.path.join("/net/hci-storage02/userfolders/amatskev/", "val/")
    else:
        valdir = os.path.join("../","val/")

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

    #create subset with only img_size images
    numbers=np.arange(len(val_dataset))
    indices=np.random.choice(numbers,img_size,replace=False)

    #no dataloader, we need shuffled samples but after that constant drawing from list (either this or update to pytorch 0.4.1)
    val_images = [val_dataset[i][0] for i in indices]
    val_targets = [val_dataset[i][1] for i in indices]

    torch.save((val_images,val_targets), "../imagenet_{}.torch".format(img_size))

    return val_images,val_targets


def perturbator_3001(NeuralNet, images_targets, pop_size=400, max_iterations=100, f_param=0.5, criterium="paper",
                     pixel_number=1, cuda=False, print_every=1):

    print("Running perturbator_3001...")
    total_time=time()

    #assertions
    assert(f_param>=0 and f_param <= 2),"0<=F<=2 !"
    assert (criterium=="paper" or criterium=="over50" or criterium=="under0.1" or criterium=="smaller"), "No valid criterium! "

    # extract images and targets
    images, targets = images_targets
    if cuda:
        images=torch.stack(images).cuda()
    else:
        images=torch.stack(images)


    list_pert_samples=[]
    list_iterations=[]

    soft=nn.Softmax(dim=0)

    sample_count = 0
    cand_score=0
    for idx, data in enumerate(images):

        #pick target
        target = targets[idx]

        #timestamp
        data_time=time()

        # draw the coordinates and rgb-values of the agents from random
        coords = np.random.random_integers(0, data.size()[-1]-1, (pop_size,pixel_number,2))
        rgb = np.array([(np.random.normal(0.5 / 0.229, 0.5 / 0.229, (pop_size, pixel_number, 1)) % 1 / 0.229) - (0.485 / 0.229),
                        (np.random.normal(0.5 / 0.224, 0.5 / 0.224, (pop_size, pixel_number, 1)) % 1 / 0.224) - (0.456 / 0.224),
                        (np.random.normal(0.5 / 0.225, 0.5 / 0.225, (pop_size, pixel_number, 1)) % 1 / 0.225) - (0.406 / 0.225)])[:, :, :, 0]\
            .swapaxes(0, 1).swapaxes(1, 2)

        #initialize scores for comparison fathers-sons
        fitness_list = np.ones(pop_size)


        # set iterator to zero
        iteration = 0
        found_candidate = False
        data_purb = 0

        while iteration < max_iterations:

            for i in range(pop_size):

                #make a copy so we do not alter the original image
                data_purb = data.clone()

                #replace rgbs on the x and y positions (coords) in data_purb with the new rgbs (rgb)
                if cuda:
                    data_purb[:, coords[i, :, 0], coords[i, :, 1]] = torch.cuda.FloatTensor(rgb[i].transpose())
                else:
                    data_purb[:, coords[i, :, 0], coords[i, :, 1]] = torch.FloatTensor(rgb[i].transpose())


                #data_purb[0] = (data_purb[0]/255. - 0.485) / 0.229
                #data_purb[1] = (data_purb[1]/255. - 0.456) / 0.224
                #data_purb[2] = (data_purb[2]/255. - 0.406) / 0.225

                # softmax
                if cuda:
                    score = soft(NeuralNet(data_purb.unsqueeze(0).cuda()).squeeze()).type(torch.float16)

                else:
                    score = soft(NeuralNet(data_purb.unsqueeze(0)).squeeze())

                true_score = score[target].cpu().detach().numpy()
                max_score = score.max().cpu().detach().numpy()

                #check if son is better than father
                if true_score > fitness_list[i]:
                    coords[i]=coords_fathers[i]
                    rgb[i]=rgb_fathers[i]

                #if son is better than father, check break criteria
                elif (criterium=="paper" and true_score < 0.05) or \
                        (criterium=="over50" and true_score > 0.5) or \
                        (criterium=="under0.1" and true_score < 0.001) or \
                        (criterium=="smaller" and true_score < max_score):

                    found_candidate = True
                    list_iterations.append(iteration)
                    list_pert_samples.append(data_purb.cpu())
                    print("FOUND")
                    cand_score+=1
                    break

                #if son is better than father but still not good enough
                else:
                    fitness_list[i]=true_score

            if found_candidate:
                break

            #remember the father gen
            coords_fathers=coords.copy()
            rgb_fathers=rgb.copy()

            # DE update agents
            random_numbers=np.array([np.random.choice(range(pop_size), 3, replace=False) for i in range(pop_size)])
            coords = (coords[random_numbers[:, 0]] + f_param * (coords[random_numbers[:, 1]] + coords[random_numbers[:, 2]])).\
                             astype(int) % data.size()[-1]
            rgb = (rgb[random_numbers[:, 0]] + f_param * (rgb[random_numbers[:, 1]] + rgb[random_numbers[:, 2]]))

            over0 = rgb[:, :, 0] > (1 - 0.485) / 0.229
            over1 = rgb[:, :, 1] > (1 - 0.456) / 0.224
            over2 = rgb[:, :, 2] > (1 - 0.406) / 0.225
            under0 = rgb[:, :, 0] < - 0.485 / 0.229
            under1 = rgb[:, :, 1] < - 0.456 / 0.224
            under2 = rgb[:, :, 2] < - 0.406 / 0.225

            rgb[:, :, 0][over0] -= 1 / 0.229
            rgb[:, :, 1][over1] -= 1 / 0.224
            rgb[:, :, 2][over2] -= 1 / 0.225
            rgb[:, :, 0][under0] += 1 / 0.229
            rgb[:, :, 1][under1] += 1 / 0.224
            rgb[:, :, 2][under2] += 1 / 0.225

            iteration += 1

        if not found_candidate:
            list_pert_samples.append(data_purb)
            list_iterations.append(iteration)

        #we do not want more than image_number pictures
        sample_count += 1

        if sample_count % print_every==0:
            print("image number {} with {} iterations in {} sec".format(sample_count,list_iterations[sample_count-1],time()-data_time))

            if print_every!=1:
                print("Time elapsed: {} min".format((time()-total_time)/60))
    print("CAND SCORE: ",cand_score)
    assert(1==1),"stop"
    return list_pert_samples, list_iterations


def reproduction(nr=1,print_every=50):
    """reproduction of the paper scores"""

    print("reproduction...")

    cuda=False
    if torch.cuda.is_available():
        cuda=True
    _, alexnet, _, _, _, _ = load_all_NN()

    if cuda:
        print("Using Cuda")
        alexnet.cuda()

    if cuda:

        if os.path.exists("/net/hci-storage02/userfolders/amatskev/pixel_attack/reproduction_init_{}.torch".format(nr)):
            print("path already exists for nr , loading...".format(nr))
            pert_samples, iterations, data_cropped = torch.load("/net/hci-storage02/userfolders/amatskev/pixel_attack/reproduction_init_{}.torch".format(nr))

        else:
            data_cropped = load_imagenet(600, False)
            pert_samples, iterations = perturbator_3001(alexnet, data_cropped, 400, cuda=cuda, print_every=50)

            print("saving...")
            torch.save((pert_samples, iterations, data_cropped), "/net/hci-storage02/userfolders/amatskev/pixel_attack/reproduction_init_{}.torch".format(nr))


    else:

        if os.path.exists("../reproduction_init_{}.torch".format(nr)):
            print("path already exists for nr , loading...".format(nr))
            pert_samples, iterations, data_cropped= torch.load("../reproduction_init_{}.torch".format(nr))

        else:
            data_cropped = load_imagenet(600, True)
            pert_samples, iterations = perturbator_3001(alexnet, data_cropped, 400, cuda=cuda)

            print("saving...")
            torch.save((pert_samples,iterations,data_cropped),"../reproduction_init_{}.torch".format(nr))

    print("extracting stats!")
    acc, pred_right = accuracy(alexnet, data_cropped)
    torch.save((acc, pred_right),
               "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/reproduction_acc_{}.torch".format(
                   nr))
    torch.save(extract_stats(alexnet, data_cropped, pert_samples, pred_right, True),
               "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/reproduction_init_stats_only_true_{}.torch".format(nr))
    torch.save(extract_stats(alexnet, data_cropped, pert_samples, pred_right, False),
               "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/reproduction_init_stats_{}.torch".format(nr))
    torch.save(extract_stats_with_false(alexnet, data_cropped, pert_samples),
               "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/reproduction_init_stats_false_{}.torch".format(nr))




def reproduction_loop():
    for i in np.arange(3):
        reproduction(i, print_every=50)


def accuracy(NeuralNet,vals):
    """calculates accuracy of the neural net with only our cropped batch of examples"""
    #
    print("Computing accuracy")
    soft=nn.Softmax(dim=0).cuda()

    #extract images and targets
    samples_orig, targets_orig = vals

    data=torch.stack(samples_orig)

    #calculate scores
    if torch.cuda.is_available():
        NeuralNet.cuda()
        scores_of_orig_images = NeuralNet(data.cuda()).cpu()
    else:
        scores_of_orig_images = soft(NeuralNet(data))

    #calculate percentage of right predictions
    accuracy=np.sum(scores_of_orig_images.argmax(1).detach().numpy() == targets_orig)/len(targets_orig)
    right_preds = np.where(scores_of_orig_images.argmax(1).detach().numpy() == targets_orig)[0]

    print("accuracy: ", accuracy)

    return accuracy, right_preds



def extract_stats(NeuralNet ,vals,pert_samples,pred_right, look_at_true=True):
    """extracts stats of the perturbed samples"""

    if not torch.cuda.is_available():
        NeuralNet.cpu()

    # extract targets
    targets_orig = vals[1]

    # use only data, which were predicted right in the first place
    if look_at_true:
        targets_orig = [targets_orig[i] for i in pred_right]
        pert_samples = [pert_samples[i].cpu() for i in pred_right]
    else:
        pert_samples = [sample.cpu() for sample in pert_samples]

    # computing scores of all perturbed image samples
    if torch.cuda.is_available():
        data = torch.stack(pert_samples)
        scores_of_pert_images = NeuralNet(data.cuda()).cpu()
    else:
        scores_of_pert_images = NeuralNet(torch.stack(pert_samples))

    # scores of the true targets in perturbated predictions
    target_scores_in_pert_pred = scores_of_pert_images[np.arange(len(targets_orig)), targets_orig]

    # calculate which of the classes are bigger than the prediction score for the target class
    bigger_than_target_pert = scores_of_pert_images.detach().numpy() > np.matrix(target_scores_in_pert_pred.detach().numpy()).transpose()


    # calculate percentage of classes bigger than the target class score
    success_rate = np.sum(bigger_than_target_pert.any(1))/len(targets_orig)

    # calculate confidence (average probability of target classes)
    confidence = np.sum(np.max(scores_of_pert_images.detach().numpy(), axis=0))/len(target_scores_in_pert_pred)

    # calculate how many classes are bigger than target
    number_of_bigger_classes = np.sum(bigger_than_target_pert, axis=1)
    #print("number of more probable classes: ", number_of_bigger_classes)


    return success_rate, confidence, number_of_bigger_classes




def extract_stats_with_false(NeuralNet ,vals, pert_samples):
    """computing stats with success_rate: the predictions which do not match with original prediction
    (including the false predicted in orig prediction)"""

    if not torch.cuda.is_available():
        NeuralNet.cpu()
    # extract images and targets
    samples_orig, targets_orig = vals

    pert_samples = [sample.cpu() for sample in pert_samples]

    # transform data
    data = torch.stack(samples_orig)
    pert_data=torch.stack(pert_samples)

    # calculate scores
    if torch.cuda.is_available():
        NeuralNet.cuda()
        scores_of_orig_images = NeuralNet(data.cuda()).cpu()
        scores_of_pert_images = NeuralNet(pert_data.cuda()).cpu()

    else:
        scores_of_orig_images = NeuralNet(data)
        scores_of_pert_images = NeuralNet(pert_data)

    max_scores_orig=scores_of_orig_images.argmax(1).detach().numpy()
    max_scores_pert=scores_of_pert_images.argmax(1).detach().numpy()

    confidence = np.sum(np.max(scores_of_pert_images.detach().numpy(), axis=0))/len(max_scores_pert)

    accuracy = np.sum(max_scores_orig == targets_orig) / len(targets_orig)

    target_scores_in_pert_pred = scores_of_pert_images[np.arange(len(targets_orig)), targets_orig]
    bigger_than_target_pert = scores_of_pert_images.detach().numpy() > np.matrix(target_scores_in_pert_pred.detach().numpy()).transpose()
    number_of_bigger_classes = np.sum(bigger_than_target_pert, axis=1)


    diff_orig_pert = max_scores_orig != max_scores_pert

    success_rate=np.sum(diff_orig_pert)/len(diff_orig_pert)



    return success_rate, confidence, number_of_bigger_classes

def optimize_population():
    """optimize population """

    print("Optimizing population...")

    cuda=False
    if torch.cuda.is_available():
        cuda=True

    _, alexnet, _, _, _, _ = load_all_NN()

    if cuda:
        print("Using Cuda")
        alexnet.cuda()

    data_cropped = load_imagenet(600)

    acc, _ = accuracy(alexnet,data_cropped)

    #because we run whole 2 in one run, we want good imageset
    while acc<0.50:
        data_cropped = load_imagenet(600,True)
        acc, _ = accuracy(alexnet, data_cropped)

    for population_size in [100,200,300,400,500,600,700,800,900,1000]:

        print("--------------------------------------------------------------")
        print("POPULATION: ",population_size)

        if os.path.exists("/net/hci-storage02/userfolders/amatskev/pixel_attack/optimize_population_{}.torch".format(population_size)):

            pert_samples, iterations,data_cropped = torch.load("/net/hci-storage02/userfolders/amatskev/pixel_attack/optimize_population_{}.torch".format(population_size))

        else:
            pert_samples, iterations = perturbator_3001(alexnet, data_cropped, population_size,cuda=cuda, print_every=50)

            print("saving...")
            torch.save((pert_samples, iterations, data_cropped), "/net/hci-storage02/userfolders/amatskev/pixel_attack/optimize_population_{}.torch".format(population_size))

        print("extracting stats!")
        acc, pred_right = accuracy(alexnet, data_cropped)
        torch.save((acc, pred_right),
                   "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/optimize_population_acc_{}.torch".format(
                       population_size))
        torch.save(extract_stats(alexnet, data_cropped, pert_samples, pred_right, True),
                   "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/optimize_population_stats_only_true_{}.torch".format(
                       population_size))
        torch.save(extract_stats(alexnet, data_cropped, pert_samples, pred_right, False),
                   "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/optimize_population_stats_{}.torch".format(
                       population_size))
        torch.save(extract_stats_with_false(alexnet, data_cropped, pert_samples),
                   "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/optimize_population_stats_false_{}.torch".format(
                       population_size))


def optimize_f_crit():
    """optimize f and crit """

    print("Optimizing f and crit...")
    cuda = False
    if torch.cuda.is_available():
        cuda = True

    _, alexnet, _, _, _, _ = load_all_NN()

    if cuda:
        print("Using Cuda")
        alexnet.cuda()

    data_cropped = load_imagenet(600)

    for F in [0.5, 0, 1, 1.5, 2]:

        for crit in ["smaller", "paper"]:

            print("F: ",F,", crit: ", crit)

            if os.path.exists(
                    "/net/hci-storage02/userfolders/amatskev/pixel_attack/optimize_f_crit_F_{}_crit_{}.torch".format(F,crit)):

                pert_samples, iterations, data_cropped=torch.load("/net/hci-storage02/userfolders/amatskev/pixel_attack/optimize_f_crit_F_{}_crit_{}.torch".format(F,crit))

            else:
                pert_samples, iterations = perturbator_3001(alexnet, data_cropped, 400, f_param=F, criterium=crit, cuda=cuda, print_every=50)

                print("saving...")
                torch.save((pert_samples, iterations, data_cropped), "/net/hci-storage02/userfolders/amatskev/pixel_attack/optimize_f_crit_F_{}_crit_{}.torch".format(F,crit))

            print("extracting stats!")
            acc, pred_right = accuracy(alexnet, data_cropped)
            torch.save((acc, pred_right),
                       "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/optimize_f_crit_acc_F_{}_crit_{}.torch".format(F,crit))
            torch.save(extract_stats(alexnet, data_cropped, pert_samples, pred_right, True),
                       "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/optimize_f_crit_stats_only_true_F_{}_crit_{}.torch".format(F,crit))
            torch.save(extract_stats(alexnet, data_cropped, pert_samples, pred_right, False),
                       "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/optimize_f_crit_stats_F_{}_crit_{}.torch".format(F,crit))
            torch.save(extract_stats_with_false(alexnet, data_cropped, pert_samples),
                       "/net/hci-storage02/userfolders/amatskev/pixel_attack/only_stats/optimize_f_crit_stats_false_F_{}_crit_{}.torch".format(F,crit))

def NN_and_pixel():
    """optimize """

    resnet152, alexnet, squeezenet, vgg16, densenet, inception = load_all_NN()

    if os.path.exists("../NN_and_pixel_data.torch"):
        data_cropped=torch.load("../NN_and_pixel_data.torch")

    else:
        data_cropped = load_imagenet(5)
        torch.save(data_cropped,"../NN_and_pixel_data.torch")

    NN_list=["alexnet", "squeezenet", "vgg16", "densenet"]  # resnet and inception give errors in predictions(not our fault,
                                                            # just something with the images) TODO: test resizing image

    for idx_nn,NN in enumerate([alexnet, squeezenet, vgg16, densenet]):

        for pixel_nr in [3,5,10]:

            print("NN: ",NN_list[idx_nn],", pixel number: ",pixel_nr)

            pert_samples, iterations = perturbator_3001(NN, data_cropped,5,pixel_number=pixel_nr)
            stats=extract_stats(alexnet, data_cropped, pert_samples)

            torch.save((pert_samples, iterations, stats), "../NN_and_pixel_NN_{}_pixel_{}.torch".format(NN_list[idx_nn],pixel_nr))

if __name__ ==  '__main__':
    #1
    #reproduction_loop()

    ##2
    #optimize_population()
    optimize_f_crit()

    #3
    #NN_and_pixel()