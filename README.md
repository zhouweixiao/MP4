# LCM<sup>3</sup>DS
- LCM<sup>3</sup>DS is a large-scale multi-scenario multi-domain dialogue summarization corpus annotated by ChatGPT.
- LCM<sup>3</sup>DS corpus is currently available on both [**Google Drive**](https://drive.google.com/file/d/1ZtuLcSJKlWJRNdPL8rlo0a2NCbcmDwq-/view?usp=sharing) and [**Baidu Netdisk**](https://pan.baidu.com/s/10oEgcjp2htMSIqz8GWc_kQ?pwd=fy5q).
- LCM<sup>3</sup>DS corpus is a standardized high-quality corpus that you can use for pretraining on your own model architecture.
![图片描述](data_stats.png)

# MP4
We will release the pre-trained MP4 models along with the code by **October 31, 2023**.

## Downstream Datasets
- Downstream datasets are currently available on both [**Google Drive**](https://drive.google.com/file/d/1riZX1yraagpgLIKf5YexuGXqmIa9O0DL/view?usp=sharing) and [**Baidu Netdisk**](https://pan.baidu.com/s/142DGWCutzOSwzYDk9ma-qg?pwd=n8rj).

|Dataset|Train|Val|Test|Domain|
|:---:|:---:|:---:|:---:|:---:|
| SAMSum | 14,731 | 818 | 819 | ODDS-Online |
| DIALOGSUM | 12,460 | 500 | 500 | ODDS-Daily |
| TWEETSUMM | 869 | 108 | 110 | CSDS-Tweet |

- The inference results of ChatGPT (zero-shot) on **SAMSum test set** (Appendix A of our paper) can be obtained on [**Google Drive**](https://drive.google.com/file/d/1Kr54RJHBe1czkFJjgDI3CbRQdjH8IxHa/view?usp=sharing) and [**Baidu Netdisk**](https://pan.baidu.com/s/14afZGYldAu0-X7uC8d31uA?pwd=9et9).

|Prompt|R-1|R-2|R-L|
|:---:|:---:|:---:|:---:|
| Preceding | 37.90 | 15.19 | 35.89 |
| InstructGPT | **42.17** | **16.84** | **39.26** |
| Subsequent | 40.08 | 15.41 | 37.22 |

## Inference with Our Fine-tuned SOTA Models
We will release the relevant guideline scripts before **October 31, 2023**.

## Fine-tuning with Our Pre-trained MP4 Models
We will release the relevant guideline scripts before **October 31, 2023**.

## Pre-training with Our Speaker-BART Model
- Domain-Aware Pre-training (DAP) is used for further understanding multi-scenario multi-domain dialogues, and its downstream tasks are suitable for dialogue-related tasks, not limited to dialogue summarization. The corpus with *20% masking ratio* can be found on [**Google Drive**](https://drive.google.com/file/d/1NrbLvIAh2Y0enIouXOGjsBsFvNDFpGYh/view?usp=sharing) and [**Baidu Netdisk**](https://pan.baidu.com/s/1NE1yC-ICo21YJO9k6AXJHg?pwd=mw4c), and the corpus with *40% masking ratio* can be found on [**Google Drive**](https://drive.google.com/file/d/1nxeR0nVjjqmK1u2nZByWqQDVULQpkhpZ/view?usp=sharing) and [**Baidu Netdisk**](https://pan.baidu.com/s/1rszc2pIs6ZjBHTtQFq9Qgg?pwd=9a5r).
- Task-Oriented Pre-training (TOP) is for dialogue summarization tasks, and "*dialogue-summary*" parallel data can be obtained by extracting from **LCM<sup>3</sup>DS**.
- We will release the relevant guideline scripts before **October 31, 2023**.
