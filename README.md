# Semantic-Aware Hyper-Space Deformable Neural Radiance Fields for Facial Avatar Reconstruction

This repository provides the official PyTorch implementation for the following paper:

**Semantic-Aware Hyper-Space Deformable Neural Radiance Fields for Facial Avatar Reconstruction**

![image-20240111121039177](README.assets/framework.png)

## Abstract

> Facial avatar reconstruction from a monocular video is a fundamental task for many computer vision applications. We present a novel framework based on semantic-aware hyper-space deformable NeRF that can reconstruct high-fidelity face avatars from either 3DMM coefficients or audio features. Our proposed framework can deal with both local facial movement and global movement of the head and torso by using semantic guidance and a unified hyper-space deformation module. Specifically, we adopt a dynamic weighted ray sampling strategy for paying different attention to different parts of semantic regions and enhance the deformable NeRF by incorporating semantic guidance for capturing the fine-grained details of various facial parts. Furthermore, we introduce a hyper-space deformation module that transforms the observation space coordinates to the canonical hyper-space coordinates for learning natural facial deformation and head-torso movements. We conduct extensive experiments to demonstrate the effectiveness of our framework and show that our method outperforms the existing state-of-the-art methods. 

## Acknowledgments

Part of the code is borrowed from [Nerface](https://github.com/gafniguy/4D-Facial-Avatars)and [AD-NeRF](https://github.com/YudongGuo/AD-NeRF).