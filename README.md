# Cozy SAL-VTON Clothes Try-on Node

A ComfyUI node to dress-up your friends or your characters.

Made with 💚 by the CozyMantis squad.

![ComfyUI try-on node](./assets/node.jpeg)

## Installation

- Copy all the node files into the `custom_nodes/cozy-sal-vton` directory, then run `pip install -r requirements.txt` to install the required dependencies.
- Download the required models from the [official release](https://www.modelscope.cn/models/iic/cv_SAL-VTON_virtual-try-on/summary) by using git: `git clone https://www.modelscope.cn/iic/cv_SAL-VTON_virtual-try-on.git`
- Copy all the files in the above repository to the `models/sal-vton` directory.

## Usage

The SAL-VTON models have been trained on the [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset, so for best results you'll want:

- images that have a white/light gray background
- upper-body clothing items (tops, tshirts, bodysuits, etc.)
- an input person standing up straight, pictured from the knees/thighs up.

The virtual try-on node takes three images as input:

- the first image is the person you want to dress up
- the second image is the clothing item you want to put on them
- the third image is the mask of the clothing item (generate this via any background removal node)

There are two stages to the process:

- in the first stage, coordinates are generated for a series of predefined landmarks on the person's body and the clothing item
- in the second stage, the landmark coordinates are used to warp the garment and generate a new image of the person wearing the clothing item

The node will output:

- the image of the person wearing the clothing item
- an image displaying the landmark coordinates for the person
- an image displaying the landmark coordinates for the clothing item

The coordinates will be auto-generated the first time you run the node. If needed, you can correct the fit by manually adjusting the landmark coordinates and re-running the workflow. Press the "Update Landmarks" button to bring up the landmark editor.

![ComfyUI try-on node](./assets/overlay.jpeg)

Press one of the buttons in the left column to select a specific landmark, then click on the images to assign it.

Each landmark also needs to be assigned a category for each input. The categories are represented in the dropdowns next to each landmark:

- the first dropdown represents the person landmark category, and is one of: 0 (if the landmark does not exist in the provided image), 1 (if the landmark exists but is occluded by other parts of the body), or 2 (if the landmark exists and is not occluded)
- the second dropdown represents the clothing item landmark category, and is one of: 0 (if the landmark does not exist in the provided image), or 1 (if the landmark exists)

## Acknowledgements

Based on the excellent paper ["Linking Garment With Person via Semantically Associated Landmarks for Virtual Try-On"](https://openaccess.thecvf.com/content/CVPR2023/papers/Yan_Linking_Garment_With_Person_via_Semantically_Associated_Landmarks_for_Virtual_CVPR_2023_paper.pdf) by Keyu Yan, Tingwei Gao, HUI ZHANG, Chengjun Xie.
