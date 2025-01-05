# A Generative Self-supervised Framework for Cognitive Radio Leveraging Time-Frequency Features and Attention-based Fusion  

This repository provides the implementation of the generative self-supervised framework described in the following paper:

> **Chen, Shuai; Feng, Zhixi; Yang, Shuyuan; Ma, Yue; Liu, Jun; Qi, Zhuoyue**  
> *A Generative Self-supervised Framework for Cognitive Radio Leveraging Time-Frequency Features and Attention-based Fusion*.  
> IEEE Transactions on Wireless Communications, 2024.  
> DOI: [10.1109/TWC.2024.3513980](https://doi.org/10.1109/TWC.2024.3513980)  

---

## **Setup**

### **Step 1: Install Dependencies**
Ensure you have Python installed (version >= 3.8 is recommended). Install the required dependencies using the provided `requirements.txt` file:

### **Step 2: Prepare the Datasets**
Follow the dataset processing instructions provided in the paper. The datasets used in the experiments are:

- RadioML2016.10a  
- RadioML2016.10b  
- RML2016.04c  
- ADS-B short dataset

### **Step 3: Run Pretraining**

Execute the pretrain.py script to perform pretraining:
```bash
python pretrain.py
```
This script will:

- Load the datasets from the data/ directory.
- Perform self-supervised pretraining using the settings described in the paper.

### **Citation**
If you find this repository useful in your work, please consider citing the following paper:
```bash
@ARTICLE{10804099,
  author={Chen, Shuai and Feng, Zhixi and Yang, Shuyuan and Ma, Yue and Liu, Jun and Qi, Zhuoyue},
  journal={IEEE Transactions on Wireless Communications}, 
  title={A Generative Self-supervised Framework for Cognitive Radio Leveraging Time-Frequency Features and Attention-based Fusion}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Time-frequency analysis;Data mining;Spectrogram;Transformers;Modulation;Radio communication;Noise reduction;Cognitive radio;Time-domain analysis;Generative framework;self-supervised learning (SSL);cognitive radio technology (CRT)},
  doi={10.1109/TWC.2024.3513980}
}
```
