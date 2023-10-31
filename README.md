# Read Me

<br/>

**Paper:** Discourse Structures Guided Fine-grained Propaganda Identification<br/>
**Accepted:** The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)<br/>
**Authors:** Yuanyuan Lei, Ruihong Huang<br/>
**Paper Link:** https://arxiv.org/pdf/2310.18544.pdf

<br/>

## Task Description
Identifying propaganda content at sentence-level and token-level.

<br/>

## Data Description
* **Propaganda:** We used the propaganda dataset for the propaganda identification experiments (https://aclanthology.org/D19-1565/) (https://propaganda.qcri.org/fine-grained-propaganda-emnlp.html)<br/>
* **PDTB:** We used PDTB 2.0 dataset to train the discourse relations model (https://aclanthology.org/L08-1093/) (https://catalog.ldc.upenn.edu/LDC2008T05)<br/>
* **NewsDiscourseData:** We used the news discourse structure dataset to train the news discourse role model (https://github.com/prafulla77/Discourse_Profiling)<br/>

<br/>

## Code Description
* **sentence_identification_additional.py:** sentence-level propaganda identification with feature concatenation model (section 3.1)<br/>
* **sentence_identification_distill.py:** sentence-level propaganda identification with knowledge distillation model (section 3.2)<br/>
* **token_identification_additional.py:** token-level propaganda identification with feature concatenation model (section 3.1)<br/>
* **token_identification_distill.py:** token-level propaganda identification with knowledge distillation model (section 3.2)<br/>

<br/>

## Citation
If you are going to cite this paper, please use the form:

Yuanyuan Lei and Ruihong Huang. 2023. Discourse Structures Guided Fine-grained Propaganda Identification. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore. Association for Computational Linguistics.

@misc{lei2023discourse,
      title={Discourse Structures Guided Fine-grained Propaganda Identification}, 
      author={Yuanyuan Lei and Ruihong Huang},
      year={2023},
      eprint={2310.18544},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

<br/>
