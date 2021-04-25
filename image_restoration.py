import numpy as np
from skimage.color import rgb2gray
from imageio import imread
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import sol5_utils
import scipy
from scipy.ndimage.filters import convolve
from tensorflow.keras.layers import Input, Conv2D, Activation, Add
GRAY_IMAGE = 1
INTENSITIES = 255
BIN_HIS = 256


def normelized(image):
    " this function get an image and return image normelized by 255"
    image /= INTENSITIES
    return image


def read_image(filename, representation):
     """This function reads an image file and
     # converts it into a given representation.
     # :param filename: image file
     # :param representation: (1) - grayscale image
     # (2) - RGB image
     # :return: image type np.float64
     # """
     im = imread(filename)
     # converting to matrix of float
     im_float = im.astype(np.float64)
     if representation == GRAY_IMAGE:
        im_float = rgb2gray(im_float)
     return normelized(im_float)

def load_dataset(filenames , batch_size , corruption_func ,crop_size):
 """

 :param filenames: A list of filenames of clean images.
 :param batch_size: The size of the batch of images for each iteration of
 Stochastic Gradient Descent.
 :param corruption_func:A function receiving a numpy’s array
 representation of an image as a single argument,
 and returns a randomly corrupted version of the input image.
 :param crop_size:A tuple (height, width) specifying the crop size of the
 patches to extrac
 :return: source_batch, target_batch
 """""

 im_dict={}
 while True:
         height, width = crop_size
         shape =(batch_size, crop_size[0], crop_size[1], 1)
         source_batch= np.ndarray(shape)
         target_batch = np.ndarray(shape)
         for i in range(batch_size):
             im = choose_image(filenames, im_dict)
             x_size_large,y_size_large = choose_patch_size(im,crop_size)
             clean_patch = getPatch(im ,x_size_large,y_size_large)
             corruption_patch =corruption_func(clean_patch)
             h_large ,w_large =clean_patch.shape
             x_final = np.random.randint(0, h_large-height)
             y_final =np.random.randint(0, w_large-width)
             final_patch = clean_patch[x_final:x_final+height, y_final
             : y_final+width]
             final_patch = final_patch[:, :, np.newaxis]
             final_corruption_patch =corruption_patch[x_final:x_final+height,y_final
             :y_final+width]
             final_corruption_patch = final_corruption_patch[:, :, np.newaxis]
             source_batch[i] =final_corruption_patch -0.5
             target_batch[i] =final_patch -0.5
         yield source_batch, target_batch

def choose_patch_size(im, crop_size):

     """sample the size of a larger random crop, of size 3 × crop_size"""
     h, w = crop_size
     h_im, w_im = im.shape
     if h*3 > h_im or w*3 > w_im:
        return im.shape
     else:
        return h*3, w*3

def getPatch(im, x_patch_size_,y_patch_size):
     """sample a larger random crop, of size 3 × crop_size"""
     x_im, y_im = im.shape
     random_x = np.random.randint(0, x_im - x_patch_size_)
     random_y = np.random.randint(0, y_im - y_patch_size)
     patch = im[random_x:random_x+x_patch_size_, random_y:random_y+y_patch_size]
     return patch

def choose_image(images, dict):

 """choose image from a list of filenames,
 if its a new image ,it read the image and add its to the dictonary
 otherwise it return the value from the current dict"""

 idx = np.random.randint(0, len(images)-1)
 if images[idx] not in dict:
    dict[images[idx]] = read_image(images[idx], 1)
 return dict[images[idx]]

def resblock(input_tensor, num_channels):
     """
     :param input_tensor:symbolic input tensor
     :param num_channels:number of channels for each of its
     convolutional layers
     :return:symbolic output tensor of the layer configuration
     """

     first_conv = Conv2D(num_channels, (3, 3), padding="same")(input_tensor)
     first_relu = Activation('relu')(first_conv)
     second_cov = Conv2D(num_channels, (3, 3), padding="same")(first_relu)
     adding = Add()([input_tensor, second_cov])
     output = Activation('relu')(adding)
     return output

def build_nn_model(height ,width ,num_channels ,num_res_blocks):
    """
    build_nn_model and return untrained Keras model
    """
    input = Input(shape=(height, width, 1))
    first_conv = Conv2D(num_channels, (3, 3), padding="same")(input)
    block_input = Activation('relu')(first_conv)
    for block in range(num_res_blocks):
        block_input = resblock(block_input, num_channels)
    output = Conv2D(1, (3, 3), padding='same')(block_input)
    output = Add()([input, output])
    return Model(inputs=input, outputs=output)

def train_model(model ,images , corruption_func, batch_size ,steps_per_epoch, num_epochs, num_valid_samples):
     """

     :param model:a general neural network model for image restoration.
     :param images:a list of file paths pointing to image files. You should
     assume these paths are complete, and
     should append anything to them.
     :param corruption_func: same as described in section
     :param batch_size:the size of the batch of examples for each iteration of SGD
     :param steps_per_epoch:The number of update steps in each epoch
     :param num_epochs:The number of epochs for which the optimization will run
     :param num_valid_sampels:The number of samples in the validation set to
     test on after every epoc
     :return:
     """

     image_num = len(images)
     ratio = int(image_num * 0.8)
     training = images[:ratio]
     validation = images[ratio:]
     crop_size = (model.input_shape[1], model.input_shape[2])
     training_data_set = load_dataset(training, batch_size, corruption_func,
     crop_size)
     validation_data_set = load_dataset(validation, batch_size, corruption_func,
     crop_size)
     model.compile(loss='mean_squared_error', optimizer=Adam(beta_2=0.9))
     model.fit_generator(generator=training_data_set,
     steps_per_epoch =steps_per_epoch, epochs=num_epochs,
     validation_data =validation_data_set,
     validation_steps =num_valid_samples)
def restore_image(corrupted_image , base_model):
     """
     :param corrupted_image:a grayscale image of shape (height, width) and with
     values in the [0, 1] range of type float64
     :param base_model : a neural network trained to restore small patches
     :return:
     """
     h, w =corrupted_image.shape
     input =Input(shape=(h, w, 1))
     base = base_model(input)
     new_model = Model(inputs=input, outputs=base)
     x = (corrupted_image - 0.5)[np.newaxis, :, :, np.newaxis]
     y = new_model.predict(x)[0, :, :, 0]
     return (0.5 + y).clip(0, 1).astype(np.float64)

def add_gaussian_noise(image , min_sigma , max_sigma):
     """

     :param image: a grayscale image with values in the [0, 1] range of type
     float64.
     :param min_sigma:a non-negative scalar value representing the minimal
     variance of the gaussian distribution.
     :param max_sigma: a non-negative scalar value larger than or equal to
     min_sigma, representing the maximal
     variance of the gaussian distribution.
     :return: image with noise
     """
     rand_val = np.random.uniform(min_sigma, max_sigma)
     corrupted = np.random.normal(0, rand_val, image.shape)
     return (((image + corrupted) * 255).round() / 255).clip(0, 1)

def learn_denoising_model(num_res_blocks = 5 ,quick_mode = False):
    """
    :param num_res_blocks: int value
    :param quick_mode:boolean value
    :return:new model """
    if quick_mode:
        batch_size = 10
        epoch_steps = 3
        epochs = 2
        samples = 30
    else:
        batch_size = 100
        epoch_steps = 100
        epochs = 5
        samples = 1000
        images = sol5_utils.images_for_denoising()
        new_model = build_nn_model(24, 24, 48, num_res_blocks)
        func = lambda im: add_gaussian_noise(im, 0, 0.2)
        train_model(new_model, images, func, batch_size, epoch_steps, epochs,
        samples)
    return new_model


def add_motion_blur(image, kernel_size, angle):
     """

     :param image: a grayscale image with values in the [0, 1] range of type
     float64.
     :param kernel_size:an odd integer specifying the size of the kernel
     (even integers are ill-defined)
     :param angle: an angle in range [0,pi)
     :return: blure image
     """
     kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
     blure_im = scipy.ndimage.filters.convolve(image, kernel)
     return blure_im

def random_motion_blur(image , list_of_kernel_sizes):
     """

     :param image: a grayscale image with values in the [0, 1] range of type
     float64.
     :param list_of_kernel_sizes: a list of odd integers.
     :return:
     """
     rand_angel = np.random.uniform(0, np.pi)
     kernel_idx = np.random.choice(list_of_kernel_sizes)
     return ((add_motion_blur(image, kernel_idx,
     rand_angel) * 255).round() / 255).clip(0, 1)

def learn_deblurring_model(num_res_blocks = 5 ,quick_mode = False):
     """learn the model and train it """
     if quick_mode:
         batch_size = 10
         steps_per_epoch = 3
         num_epochs = 2
         num_valid_sampels = 30
     else:
         batch_size = 100
         steps_per_epoch = 100
         num_epochs = 10
         num_valid_sampels = 1000
         images = sol5_utils.images_for_denoising()
         new_model = build_nn_model(16, 16, 32, num_res_blocks)
         func = lambda im: random_motion_blur(im, [7])
         train_model(new_model, images, func, batch_size, steps_per_epoch, num_epochs,
        num_valid_sampels)
     return new_model