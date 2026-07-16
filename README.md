<h1> Universal Manipulation Exoskeleton: Learning Compliant Whole-body Policies with Real-time Torque Feedback</h1>
<div align="center">

[Litian Liang](https://www.litianliang.org/)$^{1,\ast}$, [Jingxi Xu](https://jxu.ai/)$^{1,2,\ast}$, [Xinda Qi](https://www.linkedin.com/in/xinda-qi-42467316a)$^{1}$, [Yujun Cai](https://vanoracai.github.io/)$^{1}$, [Houzhu Ding](https://www.linkedin.com/in/houzhuding/)$^{1}$, [Luqi Wang](luqi.w@antgroup.com)$^{1}$, <br>[Zhixin Sun](https://www.linkedin.com/in/zhixin-sun-867335292/)$^{1}$, [Jyh-Herng Chow](https://www.linkedin.com/in/jyh-herng-chow-5831103/)$^{1}$, [Ming Yang](https://users.ece.northwestern.edu/~mya671/)$^{1}$, [Mark Cutkosky](https://me.stanford.edu/people/mark-cutkosky)$^{2}$

$^1$ Ant Group, $^2$ Stanford University<br> 
$^{\ast}$ Equal Contribution

[Website](https://ume-exo.github.io/) | [Paper](https://arxiv.org/pdf/2606.14218) | [Video](https://youtu.be/1x9Pnb9bv7M?si=n2_0ytXx5yQv7l2F) | [X](https://x.com/litian_liang/status/2066541466286215570)

<img style="width:100%;" src="ume/docs/assets/unsheathe_15mb.gif">

</div>

Universal Manipulation Exoskeleton (UME) is an upper-limb exoskeleton system that provides real-time haptic torque feedback during real robot teleoperation. UME also provide a simple framework to transfer human teleoperation data to autonomous robot policies.

If you find this codebase useful, consider citing:

```bibtex
@article{liang2026universal,
  title={Universal Manipulation Exoskeleton: Learning Compliant Whole-body Policies with Real-time Torque Feedback},
  author={Liang, Litian and Xu, Jingxi and Qi, Xinda and Cai, Yujun and Ding, Houzhu and Wang, Luqi and Sun, Zhixin and Chow, Jyh-Herng and Yang, Ming and Cutkosky, Mark},
  journal={arXiv preprint arXiv:2606.14218},
  year={2026}
}
```

If you have any questions, please contact [Litian Liang](https://www.litianliang.org/) at `l6liang [at] ucsd [dot] edu` or [Jingxi Xu](https://jxu.ai/) at `jxu [at] cs [dot] columbia [dot] edu`.

**Table of Contents**

If you want to start reproducing the hardware and software, you should [get started here](ume/docs/00_starter.md).

 - 🏃‍♀️ [Getting Started](ume/docs/00_starter.md)
 - 🦾 [Universal Manipulation Exoskeleton](ume/docs/01_hardware.md)
   - 🛠️ [Hardware Guide](ume/docs/01_hardware.md)
   - 💻 [Software Guide](ume/docs/02_software.md)
   - 📷 [Data Collection](ume/docs/02_software.md#data-collection)
 - 🤖 [Model Training and Evaluation](ume/docs/02_software.md#model-training)
   - 🚂 [Training](ume/docs/02_software.md#model-training)
   - 📊 [Evaluation](ume/docs/02_software.md#model-evaluation-on-real-robot)
 - 🌍 [Real World Deployment](ume/docs/02_software.md#model-evaluation-on-real-robot)

# Acknowledgements
This work is supported by Ant Group.

We would like to thank Hao Li and Tian-Ao (Teo) Ren from the Stanford BDML Lab for insightful discussions on the mechanical design of UME. We are grateful to Yifan Hou, Zhanyi Sun, and Professor Shuran Song from the Stanford REAL Lab for valuable discussions on autonomous robot experimental design and future applications. We are grateful to Mikael Jorda for feedback on the design of the retargeting algorithm.

We would also like to thank Xiangpeng Miao from Damiao Technology for actuator suggestions and support, Heng Le from X-ARM for assistance with the robotic arms, and Haoxing Guo from Hexfellow Robotics for support with the power caster module.
