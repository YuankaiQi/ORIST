# The Road to Know-Where: An Object-and-Room Informed Sequential BERT for Indoor Vision-Language Navigation
This is the repository of [ORIST](https://arxiv.org/abs/2104.04167) (ICCV 2021).

Some code in this repo are copied/modified from opensource implementations made available by
[PyTorch](https://github.com/pytorch/pytorch),
[HuggingFace](https://github.com/huggingface/transformers),
[OpenNMT](https://github.com/OpenNMT/OpenNMT-py),
[Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch),
and [UNITER](https://github.com/ChenRocks/UNITER)
The object features are extracted using [BUTD](https://github.com/peteanderson80/bottom-up-attention), with expanded object bounding boxes of [REVERIE](https://github.com/YuankaiQi/REVERIE).


## Requirements
We will provide Docker image very soon.

## Features
 * Implemeted distributed data parallel training for R2R task (pytorch).
 * Some code optimization for fast training

## Quick Start

1. Download processed data and pretrained models with the following command.
    * Processed data:
        * [ResNet Image feature in h5 format](https://drive.google.com/file/d/1CSpzu_u0WpoJX4SOUnVKBdlj8vf8WpwN/view?usp=sharing)
        * [Object feature](https://drive.google.com/file/d/1fka2w03_Ck9hVgYXahJ-eu4jN_2PZlb8/view?usp=sharing)
        * [BUTD Image feature used only when no objects exist](https://drive.google.com/file/d/17OgBE3zutg5QGI7TaCXLpYS1LIa_250Z/view?usp=sharing)
    * For evaluation only:
        * [REVERIE model](https://drive.google.com/file/d/1mCpUWdP8cKy2tnH2nION08XA-YPm-wNM/view?usp=sharing)
        * [NDH_Oracle model](https://drive.google.com/file/d/1RFZX8KGX7-5hkL5IkBx4tuf-K_y-pV3G/view?usp=sharing)
        * [NDH_Navigator model](https://drive.google.com/file/d/1aqPuSU_tUdwcdcJnd7n9LyoalEIG5Rg_/view?usp=sharing)
        * [NDH_Mix model](https://drive.google.com/file/d/14yqDwlN6re8gyKLlUPDsNSJvLudsUXaM/view?usp=sharing)
        * [R2R model](https://drive.google.com/file/d/1K-eFdeZsqy0ZJSbcfO3sYPdHv-X9Q4-B/view?usp=sharing)
    * For training:
        * [Initialization model weights for training](https://drive.google.com/file/d/1lcyq4rPGNAiI0V2JfcH2LwzcvESUkIdx/view?usp=sharing)
        * [Speaker for R2R training](https://drive.google.com/file/d/1ScdMQpj7X34J03jyq0p28HJFK_chREgy/view?usp=sharing)
        * [Aug path for R2R training](https://drive.google.com/file/d/12030ewR9KI15eh3mthPhEMWI3dLEzw05/view?usp=sharing)
    
2. Build Matterport3D simulator

Build EGL version using CMake:
```bash
mkdir build && cd build
cmake -DOSMESA_RENDERING=ON ..
make
cd ../
```

3. Run inference:
    sh eval_scripts/xxx.sh

4. Run training:
    sh run_scripts/xxx.sh


## Citation

If you find this code or data useful for your research, please consider citing:
```
@inproceedings{orist,
  author    = {Yuankai Qi and
               Zizheng Pan and
               Yicong Hong and
               Ming{-}Hsuan Yang and
               Anton van den Hengel and
               Qi Wu},
  title     = {The Road to Know-Where: An Object-and-Room Informed Sequential BERT for Indoor Vision-Language Navigation},
  booktitle   = {ICCV},
  year      = {2021}
}

@inproceedings{reverie,
  author    = {Yuankai Qi and
               Qi Wu and
               Peter Anderson and
               Xin Wang and
               William Yang Wang and
               Chunhua Shen and
               Anton van den Hengel},
  title     = {{REVERIE:} Remote Embodied Visual Referring Expression in Real Indoor
               Environments},
  booktitle = {CVPR},
  pages     = {9979--9988},
  year      = {2020}
}
```
