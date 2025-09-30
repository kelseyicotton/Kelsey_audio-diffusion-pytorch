#kelsey #project #audio

# DL Model 2025

## Kivanc Notes

- At some point in future when we have a barebone diffusion framework, we could look into sociophonetics: <https://academic.oup.com/edited-volume/28007/chapter-abstract/211779039?redirectedFrom=fulltext>

## 2025-07–21 Meeting Notes

- Meeting notes:
  - Next step: let's dive deeper into how diffusion models work in most core aspects 
  - Let's try to run audio diffusion pytorch as a coding excercise
    - After we can try AudioLDM as well if you want to try
  - Major aim milestone: a detailed pipeline, where we decide which modality guides the voice generation.
    - We talked about phonemes, and audio features. Note that, whatever modality we choose to use, in most cases we also need an encoder for it. We can also use generic audio encoders in this step. 
  - Note: #todo/kivanc check the audio guided video diffusion audio part

## Kivanc notes 2025-07-18
#kivanc

- In our draft model, F0 is fundemantal frequency or Fo as in Zhang et al 2023? Let's chat about the reasoning behind this!
  - #kelsey - the fundamental frequency

Wang et al -> Gus is cool!

- Cascaded diffusion -> Have you tested the computation time? It could be ok for live-coding, but I have doubts that it would be fast enough for realtime. Additionally, Why can't we achieve the same results with one diffusion instead of many?
  - Is the following sentence AI generated? That paper generates MIDI, not audio. Am I missing sth here? 
    - #kelsey - lol, no that's me linking the wrong reference - 

    - *Cascaded Diffusion Models have similarly proven effective in generating high fidelity audio material [Wang et al](https://openreview.net/forum?id=sn7CYWyavh).*
    - Still, I think cascaded diffusion could be a really cool venue to dive into.
    - Let's take a step back and remember the main idea here. How can we guide diffusion with conditioning? That, I think, is the main approach in this proposal. Thus, we should look into that front on the side. 
- CREPE is quite old. I wonder if there are better alternatives for F0 detection. Before looking into this, let's discuss the idea behind usign F0 first. I have some inputs here.
    - #kelsey - some others include PRAAT, Dr SPEECH 3.0, RAPT, Nebula (these are all also a little on the old side), CREPE seemed to be mentioned quite frequently in a lot of the newer voice synthesis papers - but im open to cheeck out newer options
      - SwiftFO seems to be a newer option
      - there's also mention in some of the newer vocoder papers about neural approaches, i'll do some more digging into this
- We need voice datasets. I think this is the only major missing point in the proposal. 
  - #kelsey - yup! good point, i've been gathering links for datasets of more extra-normal sounds but pickings are slim and they generally seem to be licenced for NC. 

- There is no mention to sampling in this proposal, which is one of the great affordances of diffusion models. We should discuss this aspect as well. 
  - #kelsey - yup, good point

- It would be good for you to read the original diffusion paper.
  - OG paper: <https://arxiv.org/abs/2006.11239>
  - A good summary here as well: <https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction>
  - And also dive a bit deeper into diffusion: <https://www.youtube.com/watch?v=HWesR4t049w>
- Main recommendation: Would you like to go through the pytorch diffusion audio repo that you added as an excercise? I think it could be really good for you to see all aspects of a standard diffusion model. The exercise would be copying all the code of the diffusion repo by typing to a new repo/jupyter-notebook, and while doing so, reading each function and understanding what they do. 


- Great that we have a venue aim here! I think we should aim for: 
  - 1- IJCAI Creativity track in January
  - 2- If rejected, then IEEE
- This is so fun! Happy to join this jam.  

## Cascaded Diffusion Models for Voice Generation

Diffusion models have demonstrated success at generating high fidelity voice material in speech and singing contexts (Zhang et al 2023, Choi et al 2025, Hono et al 2024, Takahashi 2023), but have yet to be implemented in generating expressive and [non-verbal affect bursts](https://psycnet.apa.org/record/2003-04703-007) (these are phonated sounds). Cascaded Diffusion Models have similarly proven effective in generating high fidelity audio material [Wang et al](https://openreview.net/forum?id=sn7CYWyavh).

The use of cascaded diffusion models breaks down the denoising process into multiple, sequential stages. This multi-stage process further enables opportunities to condition each de-noising stage on extracted voice features to improve the quality of the generated voice material, and improve convergence speeds. This project aims to build upon existing diffusion model development in voice and speech domains (such as [Byun et al 2025](https://ieeexplore.ieee.org/document/11027554); [Choi et al 2025](https://ieeexplore.ieee.org/document/10850769): [Popov et al 2021](https://arxiv.org/abs/2105.06337); [Lu et al 2022](https://arxiv.org/abs/2202.05256); Kong et al 2020), adapting existing architectures to accept raw audio input and to integrate injected conditioning of pitch and timbral vocal characteristics into the coarse and medium diffusion stages.

## Prospective Model Outline

![Prospective Outline Here:](IMG_20250703_135816402_HDR.jpg)

- Extractors:
  - Precedence and demonstrated success using CREPE for F0 (in Cui et al 2024)
- Audio Diffusion PyTorch: [https://github.com/archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
  - How can we integrate this, maybe experimenting with conditioned injection on F0 and mcep to improve speed of convergence?

## Venues to Target

| Venue                                                      | URL                                                                                                                                 | Important Date                        | Notes                                                                       |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | --------------------------------------------------------------------------- |
| ICASSP                                                     | [https://2026.ieeeicassp.org](https://2026.ieeeicassp.org)                                                                             | Papers: September 17                  | TooSoon                                                                     |
| ICML                                                       | [https://icml.cc/Conferences/2025/Dates](https://icml.cc/Conferences/2025/Dates)                                                       | Papers (based on 2025 dates): January | Feasible                                                                    |
| IJCAI                                                      | [https://2025.ijcai.org/important-dates/](https://2025.ijcai.org/important-dates/)                                                     | Papers (based on 2025 dates): January | Feasible: Submission to Special Track on AI, the Arts and Creativity Papers |
| IEEE Transactions on Audio, Speech and Language Processing | [https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=10723155](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=10723155) | Rolling                               | Journal!                                                                    |

## Related Work that this is inspired by/builds upon:

Cui, Jianwei, Yu Gu, Chao Weng, Jie Zhang, Liping Chen, and Lirong Dai. “Sifisinger: A High-Fidelity End-to-End Singing Voice Synthesizer Based on Source-Filter Model.” In  *ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* , 11126–30, 2024. [https://doi.org/10.1109/ICASSP48485.2024.10446786](https://doi.org/10.1109/ICASSP48485.2024.10446786).

Zhang, Zewang, Yibin Zheng, Xinhui Li, and Li Lu. “WeSinger 2: Fully Parallel Singing Voice Synthesis via Multi-Singer Conditional Adversarial Training.” In  *ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* , 1–5, 2023. [https://doi.org/10.1109/ICASSP49357.2023.10095102](https://doi.org/10.1109/ICASSP49357.2023.10095102)

Borsos, Zalán, Matt Sharifi, Damien Vincent, Eugene Kharitonov, Neil Zeghidour, and Marco Tagliasacchi. “SoundStorm: Efficient Parallel Audio Generation.” arXiv, May 16, 2023. [https://doi.org/10.48550/arXiv.2305.09636](https://doi.org/10.48550/arXiv.2305.09636)

Byun, Dong-Min, Seung-Bin Kim, and Seong-Whan Lee. “Hierarchical Diffusion Model for Zero-Shot Singing Voice Synthesis With MIDI Priors.” IEEE Transactions on Audio, Speech and Language Processing 33 (2025): 2326–36. [https://doi.org/10.1109/TASLPRO.2025.3577324](https://doi.org/10.1109/TASLPRO.2025.3577324)

Chen, Guan-Yuan, Ya-Fen Yeh, and Von-Wun Soo. “RAT: Radial Attention Transformer for Singing Technique Recognition.” In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5, 2023. [https://doi.org/10.1109/ICASSP49357.2023.10095390](https://doi.org/10.1109/ICASSP49357.2023.10095390)

Chen, Liping, Wenju Gu, Kong Aik Lee, Wu Guo, and Zhen-Hua Ling. “Pseudo-Speaker Distribution Learning in Voice Anonymization.” IEEE Transactions on Audio, Speech and Language Processing 33 (2025): 272–85. [https://doi.org/10.1109/TASLP.2024.3519879](https://doi.org/10.1109/TASLP.2024.3519879)

Chen, Nanxin, Yu Zhang, Heiga Zen, Ron J. Weiss, Mohammad Norouzi, and William Chan. “WaveGrad: Estimating Gradients for Waveform Generation.” arXiv, October 9, 2020. [https://doi.org/10.48550/arXiv.2009.00713](https://doi.org/10.48550/arXiv.2009.00713)

Chen, Po-Wei, and Von-Wun Soo. “A Few Shot Learning of Singing Technique Conversion Based on Cycle Consistency Generative Adversarial Networks.” In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5, 2023. [https://doi.org/10.1109/ICASSP49357.2023.10097009](https://doi.org/10.1109/ICASSP49357.2023.10097009).

Chen, Sanyuan, Chengyi Wang, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, et al. “Neural Codec Language Models Are Zero-Shot Text to Speech Synthesizers.” IEEE Transactions on Audio, Speech and Language Processing 33 (2025): 705–18. [https://doi.org/10.1109/TASLPRO.2025.3530270](https://doi.org/10.1109/TASLPRO.2025.3530270)

Chen, Zehua, Xu Tan, Ke Wang, Shifeng Pan, Danilo Mandic, Lei He, and Sheng Zhao. “InferGrad: Improving Diffusion Models for Vocoder by Considering Inference in Training.” arXiv, February 8, 2022. [https://doi.org/10.48550/arXiv.2202.03751](https://doi.org/10.48550/arXiv.2202.03751)

Choi, Ha-Yeong, Sang-Hoon Lee, and Seong-Whan Lee. “Personalized and Controllable Voice Style Transfer With Speech Diffusion Transformer.” IEEE Transactions on Audio, Speech and Language Processing 33 (2025): 922–34. [https://doi.org/10.1109/TASLPRO.2025.3533362](https://doi.org/10.1109/TASLPRO.2025.3533362)

Cui, Jianwei, Yu Gu, Chao Weng, Jie Zhang, Liping Chen, and Lirong Dai. “Sifisinger: A High-Fidelity End-to-End Singing Voice Synthesizer Based on Source-Filter Model.” In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 11126–30, 2024. [https://doi.org/10.1109/ICASSP48485.2024.10446786](https://doi.org/10.1109/ICASSP48485.2024.10446786)

Gao, Xiaoxue, Yiming Chen, Xianghu Yue, Yu Tsao, and Nancy F. Chen. “TTSlow: Slow Down Text-to-Speech With Efficiency Robustness Evaluations.” IEEE Transactions on Audio, Speech and Language Processing 33 (2025): 693–704. [https://doi.org/10.1109/TASLPRO.2025.3533357](https://doi.org/10.1109/TASLPRO.2025.3533357)

Hono, Yukiya, Kei Hashimoto, Yoshihiko Nankaku, and Keiichi Tokuda. “PeriodGrad: Towards Pitch-Controllable Neural Vocoder Based on a Diffusion Probabilistic Model.” In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 12782–86, 2024. [https://doi.org/10.1109/ICASSP48485.2024.10448502](https://doi.org/10.1109/ICASSP48485.2024.10448502)

Kong, Zhifeng, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. “DiffWave: A Versatile Diffusion Model for Audio Synthesis.” arXiv, March 30, 2021. [https://doi.org/10.48550/arXiv.2009.09761](https://doi.org/10.48550/arXiv.2009.09761)

Kreuk, Felix, Gabriel Synnaeve, Adam Polyak, Uriel Singer, Alexandre Défossez, Jade Copet, Devi Parikh, Yaniv Taigman, and Yossi Adi. “AudioGen: Textually Guided Audio Generation.” arXiv, March 5, 2023. [https://doi.org/10.48550/arXiv.2209.15352](https://doi.org/10.48550/arXiv.2209.15352)

Lee, Sang-gil, Heeseung Kim, Chaehun Shin, Xu Tan, Chang Liu, Qi Meng, Tao Qin, Wei Chen, Sungroh Yoon, and Tie-Yan Liu. “PriorGrad: Improving Conditional Denoising Diffusion Models with Data-Dependent Adaptive Prior.” arXiv, February 20, 2022. [https://doi.org/10.48550/arXiv.2106.06406](https://doi.org/10.48550/arXiv.2106.06406)

Li, Xuyuan, Zengqiang Shang, Hua Hua, Peiyang Shi, Chen Yang, Li Wang, and Pengyuan Zhang. “SF-Speech: Straightened Flow for Zero-Shot Voice Clone.” IEEE Transactions on Audio, Speech and Language Processing 33 (2025): 1706–18. [https://doi.org/10.1109/TASLPRO.2025.3557242](https://doi.org/10.1109/TASLPRO.2025.3557242)

Lu, Yen-Ju, Yu Tsao, and Shinji Watanabe. “A Study on Speech Enhancement Based on Diffusion Probabilistic Model.” arXiv, November 21, 2021. [https://doi.org/10.48550/arXiv.2107.11876](https://doi.org/10.48550/arXiv.2107.11876)

Lu, Yen-Ju, Zhong-Qiu Wang, Shinji Watanabe, Alexander Richard, Cheng Yu, and Yu Tsao. “Conditional Diffusion Probabilistic Model for Speech Enhancement.” arXiv, February 10, 2022. [https://doi.org/10.48550/arXiv.2202.05256](https://doi.org/10.48550/arXiv.2202.05256)

Popov, Vadim, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, and Mikhail Kudinov. “Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech.” arXiv, August 5, 2021. [https://doi.org/10.48550/arXiv.2105.06337](https://doi.org/10.48550/arXiv.2105.06337)

Sha, Binzhu, Xu Li, Zhiyong Wu, Ying Shan, and Helen Meng. “Neural Concatenative Singing Voice Conversion: Rethinking Concatenation-Based Approach for One-Shot Singing Voice Conversion.” In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 12577–81, 2024. [https://doi.org/10.1109/ICASSP48485.2024.10446066](https://doi.org/10.1109/ICASSP48485.2024.10446066)

Sheng, Zheng-Yan, Li-Juan Liu, Yang Ai, Jia Pan, and Zhen-Hua Ling. “Voice Attribute Editing With Text Prompt.” IEEE Transactions on Audio, Speech and Language Processing 33 (2025): 1641–52. [https://doi.org/10.1109/TASLPRO.2025.3557193](https://doi.org/10.1109/TASLPRO.2025.3557193)

Takahashi, Naoya, Mayank Kumar, Singh, and Yuki Mitsufuji. “Hierarchical Diffusion Models for Singing Voice Neural Vocoder.” In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5, 2023. [https://doi.org/10.1109/ICASSP49357.2023.10095749](https://doi.org/10.1109/ICASSP49357.2023.10095749)

Tan, Xu, Jiawei Chen, Haohe Liu, Jian Cong, Chen Zhang, Yanqing Liu, Xi Wang, et al. “NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality.” arXiv, May 10, 2022. [https://doi.org/10.48550/arXiv.2205.04421](https://doi.org/10.48550/arXiv.2205.04421)

Tan, Xu, Tao Qin, Frank Soong, and Tie-Yan Liu. “A Survey on Neural Speech Synthesis.” arXiv, July 23, 2021. [https://doi.org/10.48550/arXiv.2106.15561](https://doi.org/10.48550/arXiv.2106.15561)

Welker, Simon, Julius Richter, and Timo Gerkmann. “Speech Enhancement with Score-Based Generative Models in the Complex STFT Domain.” arXiv, July 7, 2022. [https://doi.org/10.48550/arXiv.2203.17004](https://doi.org/10.48550/arXiv.2203.17004)

Xie, Yuankun, Yi Lu, Ruibo Fu, Zhengqi Wen, Zhiyong Wang, Jianhua Tao, Xin Qi, et al. “The Codecfake Dataset and Countermeasures for the Universally Detection of Deepfake Audio.” IEEE Transactions on Audio, Speech and Language Processing 33 (2025): 386–400. [https://doi.org/10.1109/TASLPRO.2025.3525966](https://doi.org/10.1109/TASLPRO.2025.3525966)

Yamamoto, Ryuichi, Reo Yoneyama, and Tomoki Toda. “NNSVS: A Neural Network-Based Singing Voice Synthesis Toolkit.” In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5, 2023. [https://doi.org/10.1109/ICASSP49357.2023.10096239](https://doi.org/10.1109/ICASSP49357.2023.10096239)

Yoshioka, Daiki, Yusuke Yasuda, and Tomoki Toda. “Nonparallel Spoken-Text-Style Transfer for Linguistic Expression Control in Speech Generation.” IEEE Transactions on Audio, Speech and Language Processing 33 (2025): 333–46. [https://doi.org/10.1109/TASLPRO.2024.3522757](https://doi.org/10.1109/TASLPRO.2024.3522757)

Zhang, Zewang, Yibin Zheng, Xinhui Li, and Li Lu. “WeSinger 2: Fully Parallel Singing Voice Synthesis via Multi-Singer Conditional Adversarial Training.” In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5, 2023. [https://doi.org/10.1109/ICASSP49357.2023.10095102](https://doi.org/10.1109/ICASSP49357.2023.10095102)

Zhu, Xinfa, Yuanjun Lv, Yi Lei, Tao Li, Wendi He, Hongbin Zhou, Heng Lu, and Lei Xie. “Vec-Tok Speech: Speech Vectorization and Tokenization for Neural Speech Generation.” IEEE Transactions on Audio, Speech and Language Processing 33 (2025): 1243–54. [https://doi.org/10.1109/TASLPRO.2025.3546559](https://doi.org/10.1109/TASLPRO.2025.3546559)


## Considerations for Using the Model in Performance Contexts

- Live coding context: use wave script in strudel repo- trigger RT parsing
  - audio in- audio out
    - #kivanc -> FUN! Some later, much later inputs here, to have some sort of visualizations on strudel side.
    - #kelsey -> visualisation would be fun as hell!
