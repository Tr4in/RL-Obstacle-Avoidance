import unreal
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from agent import Agent

FAST_DEPTH_MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '\\results\mobilenet-nnconv5dw-skipadd-pruned.pth.tar'
SCREENSHOT_SAVE_LOCATION = r'C:\Users\Bushw\OneDrive\Dokumente\Unreal Projects\DepthEstimation\Saved\Screenshots\Windows\Game\Screenshots\game'


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    cmap = plt.cm.viridis
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)



class Environment:
    def __init__(self, laser_lines):
        self.camera = unreal.GameplayStatics.get_all_actors_of_class(unreal.EditorLevelLibrary.get_game_world(), unreal.SceneCapture2D)[0]
        self.model_path = FAST_DEPTH_MODEL_PATH
        self.laser_lines = laser_lines
        assert os.path.isfile(self.model_path), \
        "=> no model found at '{}'".format(self.model_path)

        checkpoint = torch.load(self.model_path)
        
        if type(checkpoint) is dict:
            self.model = checkpoint['model']
            #unreal.log(model)

    def step(self, unreal_agent, action):        
        return self.get_observation()
        
        #img_merge = np.hstack([depth_pred_col])
        #save_image(img_merge, r'C:\Users\Bushw\OneDrive\Dokumente\Unreal Projects\DepthEstimation\Saved\Screenshots\Windows\Game\Screenshots\game_scene_pred.png')

    def reset(self):
        return self.get_state()

    def take_screenshot(self):
        renderTarget = self.camera.get_editor_property('capture_component2d').get_editor_property('texture_target')
        renderTarget.export_to_disk(SCREENSHOT_SAVE_LOCATION, unreal.ImageWriteOptions(unreal.DesiredImageFormat.PNG, async_= False))

    def get_state(self):
        self.take_screenshot()
        img = Image.open(SCREENSHOT_SAVE_LOCATION + '.png').convert('RGB')
        pil_to_tensor = transforms.ToTensor()(img).unsqueeze_(0)
        width, height = img.size

        self.model.eval()
        with torch.no_grad():
            pred = self.model(pil_to_tensor.cuda())

        #unreal.log(pred)
        depth_pred_cpu = np.squeeze(pred.data.cpu().numpy())

        d_min = np.min(depth_pred_cpu)
        d_max = np.max(depth_pred_cpu)
        depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

        return np.array(depth_pred_col[1:224:28,124], dtype = np.float32).flatten()
        

    def get_observation(self):
        reward = 1
        return (reward, self.get_state(), False)