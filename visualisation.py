import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
import PIL

parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True,
                    help='Path to the image to be found activations on')
parser.add_argument('--target', required=True,
                    help='The target class to find activations on')
parser.add_argument('--model', required=True,
                    help='Path to the pretrained model')
parser.add_argument('--export', required=False,
                    default=False, help='Path to the pretrained model')


opt = parser.parse_args()
print(opt)


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_name, module in self.model._modules.items():
            print(module_name)
            if module_name == 'fc':
                return conv_output, x
            x = module(x)  # Forward
            #print(module_name, module)
            if module_name == self.target_layer:
                print('True')
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.fc(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_index=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_index is None:
            target_index = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_index] = 1
        # Zero grads
        self.model.fc.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        # Take averages for each gradient
        weights = np.mean(guided_gradients, axis=(1, 2))
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) -
                                     np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam


def save_matplotlib(org_img, activation_map):
    activation_map = PIL.Image.fromarray(
        np.uint8(cm.hsv(activation_map) * 255))
    # # image = org_img.transform((224, 224), PIL.Image.EXTENT, (0, 224, 224, 224))
    # print(image.size)
    # print(activation_map.size)
    # image_final = [i + np.float32(image) for i in np.float32(activation_map)]
    #image_final = PIL.Image.blend(image, activation_map, alpha=0.5)
    plt.imshow(org_img, cmap=plt.cm.gray,
               interpolation='nearest', origin='lower')
    # plt.imshow(activation_map, alpha=0.9, cmap=plt.cm.hsv,
    #            interpolation='bilinear', origin='lower')
    plt.plot(activation_map)
    plt.show()


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./results'):
        os.makedirs('./results')
    # Grayscale activation map
    path_to_file = os.path.join('./results', file_name + '_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_HSV)
    path_to_file = os.path.join('./results', file_name + '_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (224, 224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('./results', file_name + '_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))


def preprocess_PIL(image, resize=True):
    """
        Resizing and converting the image into a tensor

    Args:
        image_path (PIL_Image): PIL opened image
        resize (bool): Resize the image to 224 X 224
    Returns:
        image_variable (PyTorch Variable): The tensor of image in a PyTorch Variable
    """

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(image)
    image_tensor.unsqueeze_(0)
    image_variable = Variable(image_tensor, requires_grad=True)

    return image_variable


def preprocess_image(cv2im, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def load_model():
    checkpoint = torch.load(
        opt.model, map_location=lambda storage, loc: storage)

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    use_gpu = torch.cuda.is_available()
    use_gpu = False

    if use_gpu:
        model = model.cuda()

    model.load_state_dict(checkpoint)
    model.eval()

    return model

if __name__ == '__main__':
    image_path = opt.img
    target_class = int(opt.target)
    file_name_to_export = opt.export
    # Open CV preporcessing
    image = cv2.imread(image_path)
    image_prep = preprocess_image(image)
    # Load the model
    model = load_model()
    # Grad cam
    grad_cam = GradCam(model, target_layer='layer4')
    # Generate cam mask
    cam = grad_cam.generate_cam(image_prep, target_class)
    # Save mask
    save_class_activation_on_image(image, cam, file_name_to_export)
    print('Grad cam completed')
