import folder_paths

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from server import PromptServer
import cv2
import os
import random
import yaml

from .generator import VTONGenerator
from .landmark import VTONLandmark
from .warping import Warping

def load_checkpoint(model, checkpoint_path, device):
    print(f"Model: {model}")
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Device: {device}")
    params = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(params, strict=False)
    model.to(device)
    model.eval()
    return model

def infer(model_path, person_img, garment_img, mask_img, person_keypoints, garment_keypoints):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    ourgen_model = VTONGenerator(12, 3, 5, ngf=96, norm_layer=nn.BatchNorm2d)
    ourgen_model = load_checkpoint(
        ourgen_model, model_path + '/pytorch_model.bin',
        device)

    ourwarp_model = Warping()
    landmark_model = VTONLandmark()
    ourwarp_model = load_checkpoint(ourwarp_model, model_path + '/warp.pth',
                                    device)
    landmark_model.load_state_dict(
        torch.load(model_path + '/landmark.pth', map_location=device))
    landmark_model.to(device).eval()

    input_scale = 4

    with torch.no_grad():
        person_img = person_img.cpu()
        person_img = (np.array(person_img).copy() * 255).astype(np.uint8)
        garment_img = garment_img.cpu()
        garment_img = (np.array(garment_img).copy() * 255).astype(np.uint8)
        mask_img = mask_img.cpu()
        mask_img = (np.array(mask_img).copy() * 255).astype(np.uint8)

        clothes = cv2.resize(garment_img, (768, 1024))
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        clothes = input_transform(clothes).unsqueeze(0).to(device)

        cm = mask_img[:, :, 0]
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)
        cm = cm.unsqueeze(0).unsqueeze(0)
        cm = torch.FloatTensor((cm.numpy() > 0.5).astype(float)).to(device)

        im = person_img
        h_ori, w_ori = im.shape[0:2]
        im = cv2.resize(im, (768, 1024))
        im = input_transform(im).unsqueeze(0).to(device)

        if person_keypoints is None or person_keypoints == "" or garment_keypoints is None or garment_keypoints == "":
            h, w = 512, 384
            p_down = F.interpolate(im, size=(h, w), mode='bilinear')
            c_down = F.interpolate(clothes, size=(h, w), mode='bilinear')
            c_heatmap, c_property, p_heatmap, p_property = landmark_model(
                c_down, p_down)

            if garment_keypoints is None or garment_keypoints == "":
                N = c_heatmap.shape[0]
                cloth_pred_class = torch.argmax(c_property, dim=1)
                cloth_point_ind = torch.argmax(
                    c_heatmap.view(N, 32, -1), dim=2).cpu().numpy()
                cloth_y, cloth_x = 8 * (cloth_point_ind // 96), 8 * (cloth_point_ind % 96)
            
            if person_keypoints is None or person_keypoints == "":
                N = p_heatmap.shape[0]
                person_pred_class = torch.argmax(p_property, dim=1)
                person_point_ind = torch.argmax(
                    p_heatmap.view(N, 32, -1), dim=2).cpu().numpy()
                person_y, person_x = 8 * (person_point_ind // 96), 8 * (person_point_ind % 96)

        if garment_keypoints is not None and garment_keypoints != "":
            g_keypoints = yaml.load(garment_keypoints, Loader=yaml.FullLoader)['keypoints']
            cloth_pred_class = [[]]
            cloth_y = [[]]
            cloth_x = [[]]
            for i in range(32):
                cloth_x[0].append(g_keypoints[i][0])
                cloth_y[0].append(g_keypoints[i][1])
                cloth_pred_class[0].append(g_keypoints[i][2])
            cloth_pred_class = torch.tensor(cloth_pred_class).to(device)

        if person_keypoints is not None and person_keypoints != "":
            p_keypoints = yaml.load(person_keypoints, Loader=yaml.FullLoader)['keypoints']
            person_pred_class = [[]]
            person_y = [[]]
            person_x = [[]]
            for i in range(32):
                person_x[0].append(p_keypoints[i][0])
                person_y[0].append(p_keypoints[i][1])
                person_pred_class[0].append(p_keypoints[i][2])
            person_pred_class = torch.tensor(person_pred_class).to(device)

        cloth_color_map = {'1': (0, 0, 255), '0': (255, 0, 0)}
        paired_cloth = clothes[0].cpu()
        c_im = (np.array(paired_cloth.permute(1, 2, 0)).copy() + 1) / 2 * 255
        c_im = cv2.cvtColor(c_im, cv2.COLOR_RGB2BGR)
        for ind in range(32):
            cloth_point_class = int(cloth_pred_class[0, ind])
            if cloth_point_class < 0.9:
                continue
            cloth_point_color = cloth_color_map[str(cloth_point_class)]
            y, x = cloth_y[0][ind], cloth_x[0][ind]
            cv2.circle(c_im, (x, y), 2, cloth_point_color, 4)
            cv2.putText(
                c_im,
                str(ind), (x + 4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=cloth_point_color,
                thickness=1)

        person_color_map = {'2': (0, 0, 255), '1': (0, 255, 0), '0': (255, 0, 0)}
        paired_im = im[0].cpu()
        p_im = (np.array(paired_im.permute(1, 2, 0)).copy() + 1) / 2 * 255
        p_im = cv2.cvtColor(p_im, cv2.COLOR_RGB2BGR)
        for ind in range(32):
            person_point_class = int(person_pred_class[0, ind])
            if person_point_class < 0.9:
                continue
            point_color = person_color_map[str(person_point_class)]
            y, x = person_y[0][ind], person_x[0][ind]
            cv2.circle(p_im, (x, y), 2, point_color, 4)
            cv2.putText(
                p_im,
                str(ind), (x + 4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=point_color,
                thickness=1)

        valid_c_point = np.zeros((32, 2)).astype(np.float32)
        valid_p_point = np.zeros((32, 2)).astype(np.float32)
        c_point_heatmap = -1 * torch.ones(32, 1024, 768)
        p_point_heatmap = -1 * torch.ones(32, 1024, 768)

        r = 20
        for k in range(32):
            property_c, property_p = cloth_pred_class[0, k], person_pred_class[0, k] - 1
            if property_c > 0.1:
                c_x, c_y = cloth_x[0][k], cloth_y[0][k]
                x_min, y_min, x_max, y_max = max(c_x - r - 1, 0), max(
                    c_y - r - 1, 0), min(c_x + r, 768), min(c_y + r, 1024)
                c_point_heatmap[k, y_min:y_max,
                                x_min:x_max] = torch.tensor(property_c)
                valid_c_point[k, 0], valid_c_point[k, 1] = c_x, c_y
            if property_p > -0.99:
                p_x, p_y = person_x[0][k], person_y[0][k]
                x_min, y_min, x_max, y_max = max(p_x - r - 1, 0), max(
                    p_y - r - 1, 0), min(p_x + r, 768), min(p_y + r, 1024)
                p_point_heatmap[k, y_min:y_max,
                                x_min:x_max] = torch.tensor(property_p)
                if property_p > 0:
                    valid_p_point[k, 0], valid_p_point[k, 1] = p_x, p_y

        c_point_plane = torch.tensor(valid_c_point).unsqueeze(0).to(device)
        p_point_plane = torch.tensor(valid_p_point).unsqueeze(0).to(device)
        c_point_heatmap = c_point_heatmap.unsqueeze(0).to(device)
        p_point_heatmap = p_point_heatmap.unsqueeze(0).to(device)

        if input_scale > 1:
            h, w = 1024 // input_scale, 768 // input_scale
            c_point_plane = c_point_plane // input_scale
            p_point_plane = p_point_plane // input_scale
            c_point_heatmap = F.interpolate(
                c_point_heatmap, size=(h, w), mode='nearest')
            p_point_heatmap = F.interpolate(
                p_point_heatmap, size=(h, w), mode='nearest')

            im_down = F.interpolate(im, size=(h, w), mode='bilinear')
            c_down = F.interpolate(cm * clothes, size=(h, w), mode='bilinear')
            cm_down = F.interpolate(cm, size=(h, w), mode='nearest')

        warping_input = [
            c_down, im_down, c_point_heatmap, p_point_heatmap, c_point_plane,
            p_point_plane, cm_down, cm * clothes, device
        ]
        final_warped_cloth, last_flow, last_flow_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, \
            delta_y_all, local_warped_cloth_list, fuse_cloth, globalmap, up_cloth = ourwarp_model(warping_input)

        gen_inputs = torch.cat([im, up_cloth], 1)
        gen_outputs = ourgen_model(gen_inputs, p_point_heatmap)

        combine = torch.cat([gen_outputs[0]], 2).squeeze()
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        rgb = (cv_img * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (w_ori, h_ori))
    return bgr, c_im, p_im, cloth_x, cloth_y, cloth_pred_class, person_x, person_y, person_pred_class

class CozySALVTONCustomNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person" : ("IMAGE", {}),
                "garment" : ("IMAGE", {}),
                "mask" : ("IMAGE", {}),
            },
            "optional" : { 
                "person_landmarks": ("STRING", {"multiline": True, "default": ""}),
                "garment_landmarks": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("dressed", "person_landmarks", "cloth_landmarks",)
    FUNCTION = "run"
    CATEGORY = "examples"

    def __init__(self):
        self.lastPerson = None
        self.lastGarment = None

    def run(self, person, garment, mask, person_landmarks, garment_landmarks):
        tmp_dir = folder_paths.get_temp_directory()
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        tmpfileprefix = "SALVTON_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        persontmpi = 255. * person[0].cpu().numpy()
        persontmpi = Image.fromarray(np.clip(persontmpi, 0, 255).astype(np.uint8))
        persontmpi.save(os.path.join(tmp_dir, tmpfileprefix + "_person.jpg"), pnginfo=None, compress_level=1)
        garmenttmpi = 255. * garment[0].cpu().numpy()
        garmenttmpi = Image.fromarray(np.clip(garmenttmpi, 0, 255).astype(np.uint8))
        garmenttmpi.save(os.path.join(tmp_dir, tmpfileprefix + "_garment.jpg"), pnginfo=None, compress_level=1)

        PromptServer.instance.send_sync("salvton-landmarks-update", {
            "person_image": tmpfileprefix + "_person.jpg",
            "garment_image": tmpfileprefix + "_garment.jpg",
        })

        result_status = infer(
            "models/sal-vton",
            person[0],
            garment[0],
            mask[0],
            person_landmarks if self.lastPerson is not None and torch.equal(self.lastPerson, person[0]) else None,
            garment_landmarks if self.lastGarment is not None and torch.equal(self.lastGarment, garment[0]) else None,
        )
        self.lastPerson = person[0]
        self.lastGarment = garment[0]
        result, c_landmarks, p_landmarks, cloth_x, cloth_y, cloth_property, person_x, person_y, person_property = result_status

        clothyml = ""
        with open('custom_nodes/cozy-sal-vton/kptemplate.yml', 'r') as template:
            index = 0
            for line in template:
                if '[]' in line:
                    modified_line = line.replace('[]', f'[{cloth_x[0][index]}, {cloth_y[0][index]}, {cloth_property[0][index]}]')
                    clothyml += modified_line
                    index += 1
                else:
                    clothyml += line

        personyml = ""
        with open('custom_nodes/cozy-sal-vton/kptemplate.yml', 'r') as template:
            index = 0
            for line in template:
                if '[]' in line:
                    modified_line = line.replace('[]', f'[{person_x[0][index]}, {person_y[0][index]}, {person_property[0][index]}]')
                    personyml += modified_line
                    index += 1
                else:
                    personyml += line

        tmpfileprefix = "SALVTON_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        cv2.imwrite(os.path.join(tmp_dir, tmpfileprefix + "_garment_l.jpg"), c_landmarks)
        cv2.imwrite(os.path.join(tmp_dir, tmpfileprefix + "_person_l.jpg"), p_landmarks)

        PromptServer.instance.send_sync("salvton-landmarks-update", {
            "person": personyml,
            "garment": clothyml,
            "person_image_l": tmpfileprefix + "_person_l.jpg",
            "garment_image_l": tmpfileprefix + "_garment_l.jpg",
        })

        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result = np.array(result).astype(np.float32) / 255.0
        result = torch.from_numpy(result)[None,]

        c_landmarks = cv2.cvtColor(c_landmarks, cv2.COLOR_BGR2RGB)
        c_landmarks = np.array(c_landmarks).astype(np.float32) / 255.0
        c_landmarks = torch.from_numpy(c_landmarks)[None,]

        p_landmarks = cv2.cvtColor(p_landmarks, cv2.COLOR_BGR2RGB)
        p_landmarks = np.array(p_landmarks).astype(np.float32) / 255.0
        p_landmarks = torch.from_numpy(p_landmarks)[None,]

        return (result, c_landmarks, p_landmarks,)
