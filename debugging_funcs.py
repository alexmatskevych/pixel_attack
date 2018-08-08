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