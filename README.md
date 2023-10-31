# Read Me

<br/>

**Paper:** Discourse Structures Guided Fine-grained Propaganda Identification<br/>
**Accepted:** The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023)<br/>
**Authors:** Yuanyuan Lei, Ruihong Huang

<br/>

## Task Description
Identifying propaganda content at sentence-level and token-level.

<br/>

## Data Description
* **Propaganda:** We used the propaganda dataset (Da San Martino et al., 2019) for the propaganda identification experiments.<br/>
* **PDTB:** We used PDTB 2.0 dataset (Prasad et al., 2008) to train the discourse relations model.<br/>
* **NewsDiscourseData:** We used the news discourse dataset (Choubey et al., 2020) to train the news discourse role model.<br/>

<br/>

## Code Description
* **sentence_identification_additional.py:** sentence-level propaganda identification with feature concatenation model (section 3.1)<br/>
* **sentence_identification_distill.py:** sentence-level propaganda identification with knowledge distillation model (section 3.2)<br/>
* **token_identification_additional.py:** token-level propaganda identification with feature concatenation model (section 3.1)<br/>
* **token_identification_distill.py:** token-level propaganda identification with knowledge distillation model (section 3.2)<br/>

<br/>

## Citation
If you are going to cite this paper, please use the form:

Yuanyuan Lei, Ruihong Huang. 2023. Discourse Structures Guided Fine-grained Propaganda Identification. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore. Association for Computational Linguistics.

<br/>
