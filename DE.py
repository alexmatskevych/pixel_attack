import numpy as np
from copy import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from time import time
import os

#debugging
from debugging_funcs import de_normalize, show_img, norm_show


def load_all_NN():

    import torchvision.models as models

    resnet152 = models.resnet152(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    squeezenet = models.squeezenet1_0(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)
    densenet = models.densenet161(pretrained=True)
    inception = models.inception_v3(pretrained=True)

    return resnet152,alexnet,squeezenet,vgg16,densenet,inception

def load_imagenet(img_size=600):
    import os
    import torchvision.datasets as datasets
    # Data loading code
    valdir = os.path.join("../","val/")

    # normally, we would use this transformation for our pretrained nets
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # use this transformation to have RGB values in [0, 255], transform as above later on
    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1./255., 1./255., 1./255.])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    #create subset with only img_size images
    numbers=np.arange(len(val_dataset))
    indices=np.random.choice(numbers,img_size,replace=False)

    #no dataloader, we need shuffled samples but after that constant drawing from list (either this or update to pytorch 0.4.1)
    val_images = [val_dataset[i][0] for i in indices]
    val_targets = [val_dataset[i][1] for i in indices]
    return val_images,val_targets


def perturbator_3001(NeuralNet ,images_targets, pop_size=400, max_iterations=100, f_param=0.5, criterium="paper",
                     pixel_number=1):

    #assertions
    assert(f_param>=0 and f_param <= 2),"0<=F<=2 !"
    assert (criterium=="paper" or criterium=="over50" or criterium=="under0.1"), "No valid criterium! "


    # extract images and targets
    images,targets=images_targets

    list_pert_samples=[]
    list_iterations=[]

    soft=nn.Softmax(dim=0)

    sample_count = 0

    for idx, data in enumerate(images):

        #pick target
        target = targets[idx]

        #timestamp
        data_time=time()

        # draw the coordinates and rgb-values of the agents from random
        coords = np.random.random_integers(0,data.size()[-1]-1, (pop_size,pixel_number,2))
        rgb = (np.random.normal(128,127,(pop_size,pixel_number, 3)) % 255).round()

        #initialize scores for comparison fathers-sons
        fitness_list=np.ones(pop_size)

        # set iterator to zero
        iteration = 0
        found_candidate = False
        data_purb = 0

        while iteration < max_iterations:

            for i in range(pop_size):

                #make a copy so we do not alter the original image
                data_purb=data.clone()

                #replace rgbs on the x and y positions (coords) in data_purb with the new rgbs (rgb)
                data_purb[:, coords[i, :, 0], coords[i, :, 1]] = torch.tensor(rgb[i].transpose(), dtype=torch.float) % 255

                #normalize the image for the network
                data_purb[0] = (data_purb[0]/255. - 0.485) / 0.229
                data_purb[1] = (data_purb[1]/255. - 0.456) / 0.224
                data_purb[2] = (data_purb[2]/255. - 0.406) / 0.225

                # softmax
                score = soft(NeuralNet(data_purb.unsqueeze(0)).squeeze())
                true_score = score[target]

                #check if son is better than father
                if true_score>fitness_list[i]:
                    coords[i]=coords_fathers[i]
                    rgb[i]=rgb_fathers[i]

                #if son is better than father, check break criteria
                elif (criterium=="paper" and true_score < 0.05) or \
                        (criterium=="over50" and true_score > 0.5) or \
                        (criterium=="under0.1" and true_score < 0.001):

                    found_candidate = True
                    list_iterations.append(iteration)
                    list_pert_samples.append(data_purb)
                    break

                #if son is better than father but still not good enough
                else:
                    fitness_list[i]=true_score

            #remember the father gen
            coords_fathers=coords.copy()
            rgb_fathers=rgb.copy()

            # DE update agents
            random_numbers=np.array([np.random.choice(range(pop_size), 3 , replace=False) for i in range(pop_size)])
            coords = (coords[random_numbers[:,0]] + f_param * (coords[random_numbers[:,1]] + coords[random_numbers[:,2]])).\
                             astype(int) % data.size()[-1]
            rgb = ((rgb[random_numbers[:,0]] + f_param * (rgb[random_numbers[:,1]] + rgb[random_numbers[:,2]])) % 255).round()


            if found_candidate:
                break

            iteration += 1

        if not found_candidate:
            list_pert_samples.append(data_purb)
            list_iterations.append(iteration)

        #we do not want more than image_number pictures
        sample_count += 1
        print("image number {} with {} iterations in {} sec".format(sample_count,list_iterations[sample_count-1],time()-data_time))

    return list_pert_samples, list_iterations


def reproduction():
    """reproduction of the paper scores"""

    _, alexnet, _, _, _, _ = load_all_NN()

    #if already saved
    if os.path.exists("../reproduction_init.torch"):
        print("path already exists,loading...")
        pert_samples, iterations, data_cropped= torch.load("../reproduction_init.torch")

    else:
        data_cropped = load_imagenet(10)
        pert_samples, iterations = perturbator_3001(alexnet, data_cropped, 100)

        torch.save((pert_samples,iterations,data_cropped),"../reproduction_init.torch")

    torch.save(accuracy(alexnet,data_cropped), "../reproduction_accuracy.torch")
    torch.save(extract_stats(alexnet, data_cropped, pert_samples), "../reproduction_stats.torch")



def accuracy(NeuralNet,vals):
    """calculates accuracy of the neural net with only our cropped batch of examples"""
    #extract images and targets
    samples_orig,targets_orig = vals

    #calculate scores
    scores_of_orig_images=NeuralNet(torch.stack(samples_orig))

    #calculate percentage of right predictions
    accuracy=np.sum(scores_of_orig_images.argmax(1).detach().numpy() == targets_orig)/len(targets_orig)

    return accuracy

def extract_stats(NeuralNet ,vals,pert_samples):
    """extracts stats of the perturbed samples"""

    # extract targets
    targets_orig = vals[1]

    #computing scores of all perturbed image samples
    scores_of_pert_images=NeuralNet(torch.stack(pert_samples))

    #scores of the true targets in perturbated predictions
    target_scores_in_pert_pred = scores_of_pert_images[np.arange(len(targets_orig)), targets_orig]

    #calculate which of the classes are bigger than the perdiction score for the target class
    bigger_than_target_pert =scores_of_pert_images.detach().numpy() > np.matrix(target_scores_in_pert_pred.detach().numpy()).transpose()

    #calculate percentage of classes bigger than the target class score
    success_rate=np.sum(bigger_than_target_pert.any(1))/len(targets_orig)

    #calculate confidence (average probability of target classes)
    confidence=np.sum(target_scores_in_pert_pred.detach().numpy())/len(target_scores_in_pert_pred)

    #calculate how many classes are bigger than target
    number_of_bigger_classes=np.sum(bigger_than_target_pert, axis=1)

    return success_rate, confidence, number_of_bigger_classes

def optimize_population():
    """optimize population """

    _, alexnet, _, _, _, _ = load_all_NN()

    if os.path.exists("../optimize_population_data.torch"):
        data_cropped=torch.load("../optimize_population_data.torch")
    else:
        data_cropped = load_imagenet(10)
        torch.save(data_cropped,"../optimize_population_data.torch")

    for population_size in [5,10,50,100,200,300,400,500]:

        print("population: ",population_size)
        pert_samples, iterations = perturbator_3001(alexnet, data_cropped, population_size)
        stats=extract_stats(alexnet, data_cropped, pert_samples)

        torch.save((pert_samples, iterations, stats), "../optimize_population_{}.torch".format(population_size))

def optimize_f_crit():
    """optimize f and crit """

    _, alexnet, _, _, _, _ = load_all_NN()

    if os.path.exists("../optimize_f_crit_data.torch"):
        data_cropped=torch.load("../optimize_f_crit_data.torch")
    else:
        data_cropped = load_imagenet(5)
        torch.save(data_cropped,"../optimize_f_crit_data.torch")

    for F in [0,0.5,1,1.5,2]:

        for crit in ["paper","over50","under0.1"]:

            print("F: ",F,", crit: ",crit)

            pert_samples, iterations = perturbator_3001(alexnet, data_cropped,5,f_param=F, criterium=crit)
            stats=extract_stats(alexnet, data_cropped, pert_samples)

            torch.save((pert_samples, iterations, stats), "../optimize_f_crit_F_{}_crit_{}.torch".format(F,crit))

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
    #reproduction()

    #2
    #optimize_population()
    #optimize_f_crit()

    #3
    NN_and_pixel()