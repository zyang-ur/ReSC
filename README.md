# Improving One-stage Visual Grounding by Recursive Sub-query Construction
[Improving One-stage Visual Grounding by Recursive Sub-query Construction](https://arxiv.org/pdf/2008.01059.pdf)

by [Zhengyuan Yang](http://cs.rochester.edu/u/zyang39/), [Tianlang Chen](http://cs.rochester.edu/u/tchen45/), [Liwei Wang](http://www.deepcv.net/), and [Jiebo Luo](http://cs.rochester.edu/u/jluo)

European Conference on Computer Vision (ECCV), 2020


### Introduction
We propose a recursive sub-query construction framework to address previous one-stage visual grounding methods' limitations on grounding long and complex queries. For more details, please refer to our [paper](https://arxiv.org/pdf/2008.01059.pdf).

<p align="center">
  <img src="http://cs.rochester.edu/u/zyang39/resq/resc.png" width="60%"/>
</p>

[1] Yang, Zhengyuan, et al. ["A fast and accurate one-stage approach to visual grounding"](https://arxiv.org/pdf/1908.06354.pdf). ICCV 2019.

### Prerequisites

* Python 3.6 (3.5 tested)
* Pytorch 0.4.1 and 1.4.0 tested (other versions in between should work)
* Others ([Pytorch-Bert](https://pypi.org/project/pytorch-pretrained-bert/), etc.) Check requirements.txt for reference.

## Installation

1. Clone the repository

    ```
    git clone https://github.com/zyang-ur/ReSC.git
    ```

2. Prepare the submodules and associated data

* RefCOCO, RefCOCO+, RefCOCOg, ReferItGame Dataset: place the data or the soft link of dataset folder under ``./ln_data/``. We follow dataset structure [DMS](https://github.com/BCV-Uniandes/DMS). To accomplish this, the ``download_dataset.sh`` [bash script](https://github.com/BCV-Uniandes/DMS/blob/master/download_data.sh) from DMS can be used.
    ```bash
    bash ln_data/download_data.sh --path ./ln_data
    ```

<!-- * Flickr30K Entities Dataset: place the data or the soft link of dataset folder under ``./ln_data/``. The formated Flickr data is availble at [[Gdrive]](https://drive.google.com/open?id=1A1iWUWgRg7wV5qwOP_QVujOO4B8U-UYB), [[One Drive]](https://uofr-my.sharepoint.com/:f:/g/personal/zyang39_ur_rochester_edu/Eqgejwkq-hZIjCkhrgWbdIkB_yi3K4uqQyRCwf9CSe_zpQ?e=dtu8qF).
    ```
    cd ln_data
    tar xf Flickr30k.tar
    cd ..
    ``` -->
<!-- * Flickr30K Entities Dataset: please download the images for the dataset on the website for the [Flickr30K Entities Dataset](http://bryanplummer.com/Flickr30kEntities/) and the original [Flickr30k Dataset](http://shannon.cs.illinois.edu/DenotationGraph/). Images should be placed under ``./ln_data/Flickr30k/flickr30k_images``.
 -->

* Data index: download the generated index files and place them as the ``./data`` folder. Availble at [[Gdrive]](https://drive.google.com/file/d/1fVwdDvXNbH8uuq_pHD_o5HI7yqeuz0yS/view?usp=sharing), [One Drive].
    ```
    rm -r data
    tar xf data.tar
    ```

* Model weights: download the pretrained model of [Yolov3](https://pjreddie.com/media/files/yolov3.weights) and place the file in ``./saved_models``. 
    ```
    sh saved_models/yolov3_weights.sh
    ```
More pretrained models are availble in the performance table [[Gdrive]](https://drive.google.com/drive/folders/1L2GUOUmhBDLA3WgaXfzLoNhJFf609vCe?usp=sharing), [One Drive] and should also be placed in ``./saved_models``.


### Training
3. Train the model, run the code under main folder. 
Using flag ``--large`` to access the ReSC-large model. ReSC-base is the default.

    ```
    python train.py --data_root ./ln_data/ --dataset referit \
      --gpu gpu_id --resume saved_models/ReSC_base_referit.pth.tar
    ```

4. Evaluate the model, run the code under main folder. 
Using flag ``--test`` to access test mode.

    ```
    python train.py --data_root ./ln_data/ --dataset referit \
      --gpu gpu_id --resume saved_models/ReSC_base_referit.pth.tar --test
    ```
### Implementation Details
We train 100 epoches with batch size 8 on all datasets expect RefCOCOg, where we find training 20/40 epoches have the best performance. We fix the bert weights during training as the default. The language encoder can be finetuned with the flag ``--tunebert``. We observe a small improvenment on some datasets (e.g. RefCOCOg). Please check other experiment settings in our [paper](https://arxiv.org/pdf/2008.01059.pdf).

## Performance and Pre-trained Models
Pre-trained models are availble in [[Gdrive]](https://drive.google.com/drive/folders/1L2GUOUmhBDLA3WgaXfzLoNhJFf609vCe?usp=sharing), [One Drive].
<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>Ours-base (Acc@0.5)</th>
            <th>Ours-large (Acc@0.5)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>RefCOCO</td>
            <td>val: 76.74</td>
            <td>val: 78.09</td>
        </tr>
        <tr>
            <td>testA: 78.61</td>
            <td>testA: 80.89</td>
        </tr>
        <tr>
            <td>testB: 71.85</td>
            <td>testB: 72.97</td>
        </tr>
        <tr>
            <td rowspan=3>RefCOCO+</td>
            <td>val: 63.21</td>
            <td>val: 62.97</td>
        </tr>
        <tr>
            <td>testA: 65.94</td>
            <td>testA: 67.13</td>
        </tr>
        <tr>
            <td>testB: 56.08</td>
            <td>testB: 55.43</td>
        </tr>
        <tr>
            <td rowspan=3>RefCOCOg</td>
            <td>val-g: 61.12</td>
            <td>val-g: 62.22</td>
        </tr>
        <tr>
            <td>val-umd: 64.89</td>
            <td>val-umd: 67.50</td>
        </tr>
        <tr>
            <td>test-umd: 64.01</td>
            <td>test-umd: 66.55</td>
        </tr>
        <tr>
            <td rowspan=2>ReferItGame</td>
            <td>val: 66.78</td>
            <td>val: 67.15</td>
        </tr>
        <tr>
            <td>test: 64.33</td>
            <td>test: 64.70</td>
        </tr>
    </tbody>
</table>
<!--         <tr>
            <td>Flickr30K Entities</td>
            <td></td>
            <td></td>
        </tr> -->

### Citation

    @inproceedings{yang2020improving,
      title={Improving One-stage Visual Grounding by Recursive Sub-query Construction},
      author={Yang, Zhengyuan and Chen, Tianlang and Wang, Liwei and Luo, Jiebo},
      booktitle={ECCV},
      year={2020}
    }
	@inproceedings{yang2019fast,
	  title={A Fast and Accurate One-Stage Approach to Visual Grounding},
	  author={Yang, Zhengyuan and Gong, Boqing and Wang, Liwei and Huang
	    , Wenbing and Yu, Dong and Luo, Jiebo},
	  booktitle={ICCV},
	  year={2019}
	}

### Credits
Our code is built on [Onestage-VG](https://github.com/zyang-ur/onestage_grounding).

Part of the code or models are from 
[DMS](https://github.com/BCV-Uniandes/DMS),
[film](https://github.com/ethanjperez/film),
[MAttNet](https://github.com/lichengunc/MAttNet),
[Yolov3](https://pjreddie.com/darknet/yolo/) and
[Pytorch-yolov3](https://github.com/eriklindernoren/PyTorch-YOLOv3).
