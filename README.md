<div align="center">

## ObjCtrl-2.5D: Training-free Object Control with Camera Pose

<div align='center'>
 <a href='https://wzhouxiff.github.io/'>Zhouxia Wang</a>
&ensp; <a href='https://nirvanalan.github.io/'>Yushi Lan</a>
&ensp; <a href='https://shangchenzhou.com/'>Shangchen Zhou</a>
&ensp; <a href='https://www.mmlab-ntu.com/person/ccloy/index.html'>Chen Change Loy</a>
</div>

<div align='center'>
 <a href='https://www.mmlab-ntu.com/'>S-Lab, NTU Singapore</a>
</div>


 [![Project Page](https://img.shields.io/badge/Project%20Page-green
)](https://wzhouxiff.github.io/projects/ObjCtrl-2.5D/)
&ensp; [![HF Demo](https://img.shields.io/badge/Demo-orange
)](https://huggingface.co/spaces/yslan/ObjCtrl-2.5D) &ensp; [![Paper](https://img.shields.io/badge/Paper-gray)]() &ensp; [![arXiv](https://img.shields.io/badge/arXiv-red)]()

</div>

---

<div>
<img src="assets/demo/demo.gif" class="img-responsive" , width="1000">
</div>

---

This repo is an official implementation of [ObjCtrl-2.5D: Training-free Object Control with Camera Pose](). 
ObjCtrl-2.5D models a 3D trajectory, extended from a 2D trajectory with depth information, as a sequence of camera poses, 
enabling object motion control using an existing camera motion control I2V generation model without training. Dynamic results achieved by our ObjCtrl-2.5D are provided in our [![Project Page](https://img.shields.io/badge/Project%20Page-green
)](https://wzhouxiff.github.io/projects/ObjCtrl-2.5D/).

## 📝 Changelog

- [ ] 202412: Release both **code** and **demo** (🤗 [![HF Demo](https://img.shields.io/badge/Demo-orange
)](https://huggingface.co/spaces/yslan/ObjCtrl-2.5D)) of ObjCtrl-2.5D.

## ⚙️ Environment
    conda create -n objctrl2.5d python=3.10
    conda activate objctrl2.5d
    pip install -r requirements.txt

## 💫 Running Gradio Demo
    python -m app

## :books: Citation
If you make use of our work, please cite our paper.
```bibtex
@inproceedings{objctrl2.5d,
  title={{ObjCtrl-2.5D}: Training-free Object Control with Camera Poses},
  author={Wang, Zhouxia and Lan, Yushi and Zhou, Shangchen and Loy, Chen Change},
  booktitle={arXiv},
  year={2024}
}
```

## 🤗 Acknowledgment
We appreciate the authors of [SVD](https://stability.ai/stable-video), [CameraCtrl](https://github.com/hehao13/CameraCtrl), [SAM2](https://github.com/facebookresearch/sam2), and [ZoeDepth](https://github.com/isl-org/ZoeDepth) for their awesome works.
## ❓ Contact
For any question, feel free to email `zhouzi1212@gmail.com`.