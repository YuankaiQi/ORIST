# The Road to Know-Where: An [O]()bject-and-[R]()oom [I]()nformed [S]()equential BER[T]() for Indoor Vision-Language Navigation
This is the repository of [ORIST](https://openaccess.thecvf.com/content/ICCV2021/html/Qi_The_Road_To_Know-Where_An_Object-and-Room_Informed_Sequential_BERT_for_ICCV_2021_paper.html) (ICCV 2021). 

<p align="center">
<img src="orist.png" width="100%">
</p>



Some code in this repo are copied/modified from opensource implementations made available by
[PyTorch](https://github.com/pytorch/pytorch),
[HuggingFace](https://github.com/huggingface/transformers),
[OpenNMT](https://github.com/OpenNMT/OpenNMT-py),
[Nvidia](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch),
and [UNITER](https://github.com/ChenRocks/UNITER)
The object features are extracted using [BUTD](https://github.com/peteanderson80/bottom-up-attention), with expanded object bounding boxes of [REVERIE](https://github.com/YuankaiQi/REVERIE).

## Features of the Code
 * Implemented distributed data parallel training (pytorch).
 * Some code optimization for fast training
 

## Requirements
 * Install Docker with GPU support (There are lots of tutorials, just google it.)
 * Pull the docker image: 
 ```
 docker pull qykshr/ubuntu:orist 
 ```

## Quick Start

1. Download the processed data and pretrained models:
    * Processed data:
        * [ResNet Image feature in h5 format](https://drive.google.com/file/d/1CSpzu_u0WpoJX4SOUnVKBdlj8vf8WpwN/view?usp=sharing)
        * [Object feature](https://drive.google.com/file/d/1oiqkwCIeOl8m8AX13zAF56ngsf4rKbB_/view?usp=sharing)
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

   Build OSMesa version using CMake:
   ```bash
   mkdir build && cd build
   cmake -DOSMESA_RENDERING=ON ..
   make
   cd ../
   ```
   
   Other versions can refer to [here](https://github.com/YuankaiQi/REVERIE#26-compile-the-matterport3d-simulator)

3. Run inference:
   
   sh eval_scripts/xxx.sh
    

4. Run training:

   sh run_scripts/xxx.sh


## Citation

If this code or data is useful for your research, please consider citing:
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
  pages     = {1655--1664},
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
