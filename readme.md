## Get Started
1.create your evn `conda create -n mytable python=3.9 -y` and install required packages including: [Pytorch](https://pytorch.org/) with torch version 1.13

2.run `pip install -r requirements.txt`

3.Download all the model weights from [HuggingFace](https://huggingface.co/poloclub/UniTable/tree/main).

4.For more details, please check [this](https://github.com/poloclub/unitable)

## Inference
Try out the officially provided inference `TSR-框架.ipynb` with table image!

## Train
1.Prepare dataset available [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet).

2.Training run like `make experiments/ssp_2m_pub_bbox_base/.done_finetune`

3.For more, see `Makefile` and `CONFIG.mk`

## Acknowledgment
Thanks to [unitable](https://github.com/poloclub/unitable) for providing the open-source framework.

Thanks to [ppocr](https://github.com/PaddlePaddle/PaddleOCR) for providing framework.
