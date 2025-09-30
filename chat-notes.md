# Cascaded diffusion models for audio generation


I need help understanding a discussion I had with my PhD supervisor

I am working on a project that involves diffusion models for audio generation. I have a background in voice, speech and singing and have been looking at different architectures for generating audio output.

Recently, I trained the Pytorch audio diffusion model ([https://github.com/archinetai/audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)) on the LJ Speech dataset. The purpose of this was to get more familiar with the architecture.

I have previously been interested in Cascaded Diffusion Models. Here is some earlier text I have worked on, on this:

The use of cascaded diffusion models breaks down the denoising process into multiple, sequential stages. This multi-stage process further enables opportunities to condition each de-noising stage on extracted voice features to improve the quality of the generated voice material, and improve convergence speeds. This project aims to build upon existing diffusion model development in voice and speech domains (such as [Byun et al 2025]([https://ieeexplore.ieee.org/document/11027554](https://ieeexplore.ieee.org/document/11027554)); [Choi et al 2025]([https://ieeexplore.ieee.org/document/10850769](https://ieeexplore.ieee.org/document/10850769)): [Popov et al 2021]([https://arxiv.org/abs/2105.06337](https://arxiv.org/abs/2105.06337)); [Lu et al 2022]([https://arxiv.org/abs/2202.05256](https://arxiv.org/abs/2202.05256)); Kong et al 2020), adapting existing architectures to accept raw audio input and to integrate injected conditioning of pitch and timbral vocal characteristics into the coarse and medium diffusion stages.

Here is a model sketch:![1759235063510](image/chat-notes/1759235063510.png)

In my last meeting with my supervisor, they expressed concerns that Cascaded Diffusion might be a bit overkill and sent me some papers on finetuning methods (PEFT), and suggested I look into pretrained audio UNets.

I am trying to make sense of what I should do now

## Response

Looking at your diagram and the context you've provided, I can help you understand what your supervisor might be suggesting and how to move forward.

**What your supervisor is likely thinking:**

Your cascaded diffusion approach is quite complex - you're proposing a multi-stage pipeline where you'd need to train separate models for coarse, medium, and fine diffusion stages, plus extractors for pitch (F0) and timbral features (MFCC). This is a significant undertaking that could take substantial time and computational resources.

**The PEFT alternative they're suggesting:**

Parameter-Efficient Fine-Tuning (PEFT) methods would let you:

* Start with a pretrained audio U-Net (like from the model you already trained)
* Add small, trainable modules or adapt only specific layers
* Condition on voice characteristics without rebuilding the entire architecture
* Achieve similar goals with much less computational overhead

**Practical next steps to clarify the direction:**

1. **Examine the PEFT papers** your supervisor sent - look specifically for how they handle conditioning on audio features like pitch and timbre
2. **Prototype a simpler approach** : Take your existing trained model and experiment with adding conditioning inputs (F0, MFCC) directly to the U-Net rather than cascading multiple models
3. **Define your research contribution** : Are you trying to improve audio quality, training efficiency, controllability, or something else? This will help determine whether cascaded diffusion's complexity is justified
4. **Schedule a follow-up meeting** to discuss specific PEFT techniques and how they could address your research goals

Your supervisor likely wants you to achieve similar controllability over voice characteristics but through a more efficient path. The question is whether the additional complexity of cascaded diffusion provides enough benefit over simpler conditioning approaches to justify the extra work.

What specific aspect of voice generation are you most interested in improving?

## Kelsey

In regards to your point 1: **Examine the PEFT papers** your supervisor sent - look specifically for how they handle conditioning on audio features like pitch and timbre

Here are the URLS to the papers my supervisor sent me:

* [https://arxiv.org/pdf/2403.14608](https://arxiv.org/pdf/2403.14608)
* [https://arxiv.org/html/2409.06633v2](https://arxiv.org/html/2409.06633v2)
* [https://dl.fbaipublicfiles.com/encodec/Diffusion/paper.pdf](https://dl.fbaipublicfiles.com/encodec/Diffusion/paper.pdf)
* [https://ai.meta.com/resources/models-and-libraries/audiocraft/](https://ai.meta.com/resources/models-and-libraries/audiocraft/)

## Response

I'll fetch these papers to help you understand how they handle conditioning on audio features and what specific PEFT approaches they're suggesting.

![favicon](https://claude.ai/_next/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3Darxiv.org%26sz%3D32&w=32&q=75)

Fetched **Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey**

arxiv.org

![favicon](https://claude.ai/_next/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3Darxiv.org%26sz%3D32&w=32&q=75)

Fetched **High-Efficient Diffusion Model Fine-tuning with Progressive Sparse Low-Rank Adaptation**

arxiv.org

Failed to fetch **https://dl.fbaipublicfiles.com/encodec/Diffusion/paper.pdf**

![favicon](https://claude.ai/_next/image?url=https%3A%2F%2Fwww.google.com%2Fs2%2Ffavicons%3Fdomain%3Dmeta.com%26sz%3D32&w=32&q=75)

Fetched **AudioCraft - Meta AI**

meta.com

Now I'll analyze these papers to understand how they handle conditioning on audio features like pitch and timbre:

## Analysis of the PEFT Papers Your Supervisor Sent

Based on the papers your supervisor shared, here's what they suggest about PEFT approaches and how they handle audio feature conditioning:

### **1. The PEFT Survey Paper (Han et al.)**

This comprehensive survey categorizes PEFT methods into four main types:

* **Additive** (like adapters, soft prompts)
* **Selective** (fine-tuning subset of parameters)
* **Reparameterized** (LoRA and variants)
* **Hybrid** (combinations of approaches)

 **Key insight for your work** : The survey shows that PEFT methods can be applied effectively to diffusion models, with LoRA being particularly successful for parameter-efficient adaptation [Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey](https://arxiv.org/pdf/2403.14608).

### **2. SaRA Paper (Progressive Sparse Low-Rank Adaptation)**

This paper introduces a novel PEFT method specifically for diffusion models that:

* Identifies "ineffective" parameters (those with smallest absolute values) in pre-trained diffusion models and re-utilizes them for fine-tuning [High-Efficient Diffusion Model Fine-tuning with Progressive Sparse Low-Rank Adaptation](https://arxiv.org/html/2409.06633v2)
* Uses nuclear norm-based low-rank constraints to prevent overfitting
* Introduces "unstructural backpropagation" to reduce memory costs significantly

 **Relevance to your pitch/timbre conditioning** : This approach could allow you to add pitch and timbral conditioning without the complexity of cascaded diffusion.

### **3. AudioCraft (Meta's Audio Generation Framework)**

AudioCraft provides a unified framework with MusicGen, AudioGen, and EnCodec models that use single autoregressive Language Models operating over compressed discrete audio tokens, with conditioning capabilities for text-to-audio applications [AudioCraft - Meta AI](https://ai.meta.com/resources/models-and-libraries/audiocraft/).

 **Key architectural insight** : Instead of cascaded diffusion, AudioCraft uses:

* EnCodec for audio compression/tokenization
* Single autoregressive model with token interleaving
* Various conditioning mechanisms (text, melodic features)

## **What Your Supervisor is Likely Suggesting**

Based on these papers, your supervisor probably wants you to consider:

### **1. Simpler Conditioning Architecture**

Instead of cascaded diffusion with separate coarse/medium/fine stages:

* Use a single diffusion model with LoRA or SaRA adaptations
* Add pitch (F0) and timbral (MFCC) conditioning directly to the U-Net
* Leverage existing pre-trained audio models rather than training from scratch

### **2. Parameter-Efficient Approaches**

Rather than training multiple diffusion stages:

* Apply LoRA to specific layers of your existing trained model
* Use SaRA's approach to identify and retrain "ineffective" parameters
* Focus conditioning on key bottleneck layers

### **3. Practical Implementation Path**

1. **Start with your existing model** : Take the Pytorch audio diffusion model you've already trained
2. **Add conditioning inputs** : Inject F0 and MFCC features using cross-attention or FiLM layers
3. **Apply PEFT** : Use LoRA or SaRA to fine-tune only specific parameters for voice characteristics
4. **Compare complexity vs. benefit** : Evaluate if this simpler approach achieves similar controllability to cascaded diffusion

### **Next Steps I'd Recommend**

1. **Experiment with LoRA conditioning** : Try adding LoRA layers to your existing model with F0/MFCC inputs
2. **Compare approaches** : Test both simple conditioning vs. your original cascaded idea on a small dataset
3. **Measure efficiency gains** : Document training time, memory usage, and parameter count differences
4. **Schedule follow-up** : Present these preliminary results to your supervisor

The core message seems to be: "Can you achieve your research goals (controllable voice synthesis) with a more efficient, proven approach before investing in the full complexity of cascaded diffusion?"

## Kelsey
![Roman et al.  From Discrete Tokens to HighFidelity Audio Using MultiBand Diffusion.pdf](https://claude.ai/api/3bb73d17-05c3-4be3-a1a8-e78ca22ca114/files/a0226504-c401-4f2a-b811-9f53d3f40308/thumbnail)pdf


Ok. I am looking at the second paper my supervisor sent me:

This section seems like it might be especially relevant?

"4.1 Model & Hyperparameters  Architecture. Similarly to Chen et al. [2020a], Kong et al. [2020b], Lee et al. [2021], we use a fully convolutional symmetrical U-net network [Ronneberger et al., 2015] with an alternation of two residual blocks [Défossez et al., 2021] and downsampling (resp. upsampling in the decoder) convolutional blocks of stride 4. The input audio conditioning is incorporated in the bottleneck of the network whereas the timestep t is embedded through a learned lookup table and added at every layer. According to the recommendation of Hoogeboom et al. [2023], it is advisable to allocate additional computational resources close to the bottleneck of the network when applying diffusion to high-dimensional data. Hence, we opted for a growth rate of 4. The weight of one model is 1 GB. A visual description of the model architecture can be seen in Fig. A.4 in the Appendix.  Input Conditioning. We use the latent representation of the publicly available EnCodec models at 24kHz [Défossez et al., 2022] which are frozen during training. The embedding sequence is upsampled using linear interpolation to match the dimension of the UNet bottleneck. In the experiments we include reconstructions using 1, 2 and 4 for EnCodec code books which correspond to bit rates of respectively 1.5kbps, 3kbps and 6kbps, when using multiple code books the embedding used is simply the average of the different code books.  Schedule. We trained our diffusion models using our proposed power schedule with power p = 7.5, β0 = 1.0e−5 and βT = 2.9e−2. Although we use very few diffusion steps (20) at generation time, we observed that it is beneficial to use many steps at training time (1000). First, it increases the versatility of the model since one can sample using any subset of steps S ⊆ {1, . . . 1000}. Second, it  allows the model to be trained on a more diverse range of noise levels √α ̄t. In the experiment section  we always use the simplest time steps sub sampling i.e. S = {i ∗ 1000  N , i ∈ {0, 1, ..., N }} where N is  the number of sampling steps (20 if not precised).  Frequency eq. processor. In the experiments we use a band processor that uses 8 mel scale frequency bands with ρ = 0.4. We compute the values of the bands σd  i on an internal music dataset.  Band Splitting. As described in 3 we use separate diffusion processes. In this work we always use a split of 4 frequency bands equally space in mel-scale using julius 1 Those bands are not related to the processor bands. The 4 models share the same hyperparameters and schedule.  Training We train our models using Adam optimizer with batch size 128 and a learning rate of 1e-4. It takes around 2 days on 4 Nvidia V100 with 16 GB to train one of the 4 models."

## Response

You've identified a highly relevant section! The Multi-Band Diffusion paper provides a concrete alternative architecture that directly addresses your pitch and timbre conditioning goals in a much simpler way than cascaded diffusion.

## **Key Insights for Your Project**

### **Input Conditioning Architecture**

The paper shows how to incorporate audio conditioning "in the bottleneck of the network" using frozen EnCodec representations that are upsampled via linear interpolation to match the U-Net bottleneck dimensions. This is exactly the kind of conditioning mechanism you could adapt for F0 and MFCC features.

### **Why This Matters for Your Work**

Instead of your cascaded approach with separate coarse/medium/fine stages, you could:

1. **Single U-Net with conditioning** : Use one diffusion model with F0 and MFCC features injected at the bottleneck
2. **Proven conditioning strategy** : The paper demonstrates successful conditioning using "frozen" feature representations that are interpolated to match network dimensions
3. **Manageable complexity** : One 1GB model vs. multiple cascaded stages

### **Practical Implementation Path**

Based on this architecture, you could:

* **Extract F0 and MFCC features** from your training audio
* **Inject them at the U-Net bottleneck** (similar to how they use EnCodec tokens)
* **Apply LoRA or SaRA** to the conditioning layers specifically
* **Keep the base diffusion model frozen** and only train the conditioning mechanism

### **Resource Requirements**

Their training took "around 2 days on 4 Nvidia V100 with 16 GB to train one of the 4 models" - much more manageable than training multiple cascaded stages from scratch.

## **Critical Questions to Discuss with Your Supervisor**

1. **Does this simpler conditioning approach achieve your research goals?** Can single-stage conditioning provide sufficient control over voice characteristics?
2. **What's the actual research contribution?** If you use established conditioning methods, what novel aspect are you investigating?
3. **Evaluation strategy** : How would you measure controllability of pitch/timbre compared to existing methods?

The Multi-Band Diffusion paper essentially shows a working example of exactly the kind of conditioning architecture your supervisor likely has in mind - proven, efficient, and much less complex than cascaded diffusion while still achieving high-quality results.

## Kelsey

What are your initial speculations on these questions you identified?

**Critical Questions to Discuss with Your Supervisor**

1. **Does this simpler conditioning approach achieve your research goals?** Can single-stage conditioning provide sufficient control over voice characteristics?
2. **What's the actual research contribution?** If you use established conditioning methods, what novel aspect are you investigating?
3. **Evaluation strategy** : How would you measure controllability of pitch/timbre compared to existing methods?

## Response

Here are my honest assessments of these critical questions:

## **1. Single-stage conditioning effectiveness**

**Likely sufficient for basic control, but may have limitations:**

* Multi-Band Diffusion shows that bottleneck conditioning works well for general audio quality
* However, their conditioning (EnCodec tokens) captures overall audio content, not specific voice characteristics
* F0 and MFCC features are more limited - F0 is 1D pitch contour, MFCC captures spectral envelope but not fine-grained timbral details
* You might achieve "coarse" control over pitch and basic timbre, but probably not the nuanced voice characteristic control that cascaded diffusion could theoretically provide

 **Critical limitation** : Single-point injection may not give you the hierarchical control over different aspects of voice generation that motivated your original cascaded approach.

## **2. Research contribution concerns**

**This is the weakest point in the simpler approach:**

* Conditioning diffusion models on audio features is well-established
* Multi-Band Diffusion already demonstrates sophisticated audio conditioning
* Adding F0/MFCC conditioning to existing architectures would be incremental engineering, not novel research

**Potential contributions could be:**

* Comparative analysis of conditioning strategies for voice control
* Novel evaluation metrics for voice characteristic controllability
* Demonstrating that simpler approaches match complex cascaded methods

 **But honestly** : This feels more like a good engineering project than PhD-level research unless you identify a genuinely novel angle.

## **3. Evaluation strategy challenges**

**This reveals deeper issues:**

* How do you objectively measure "controllability" of voice characteristics?
* Existing metrics (FID, IS) don't capture voice-specific qualities
* Perceptual studies are expensive and subjective
* You'd need to compare against what baseline? There aren't many controllable voice synthesis systems to benchmark against

 **The evaluation problem might be harder than the technical problem** , which suggests the research scope needs clearer definition.

## **My honest take**

Your supervisor seems to be steering you toward a more tractable project, but this raises questions about whether the simpler approach addresses the core research motivations that led you to cascaded diffusion in the first place. You may need to choose between:

* **Ambitious research** (cascaded diffusion) with higher risk/complexity
* **Safer engineering** (PEFT conditioning) with clearer deliverables but limited novelty

Neither is inherently wrong, but they represent different research philosophies and career implications.

## Kelsey

Let's come to this point:

**2. Research contribution concerns**
**This is the weakest point in the simpler approach:**

* Conditioning diffusion models on audio features is well-established
* Multi-Band Diffusion already demonstrates sophisticated audio conditioning
* Adding F0/MFCC conditioning to existing architectures would be incremental engineering, not novel research
  **Potential contributions could be:**
* Comparative analysis of conditioning strategies for voice control
* Novel evaluation metrics for voice characteristic controllability
* Demonstrating that simpler approaches match complex cascaded methods
  **But honestly** : This feels more like a good engineering project than PhD-level research unless you identify a genuinely novel angle.

My particular musical focus is on extended vocal techniques (scream, sighing, vocal fry)

## Response

**This changes everything.** Extended vocal techniques represent a genuinely under-explored area in computational audio that could justify the complexity of your original cascaded approach.

## **Why Extended Vocal Techniques Matter for Research**

**Established methods likely fail here:**

* Standard F0/MFCC features are designed for "normal" phonation
* Vocal fry has irregular/subharmonic F0 patterns that standard pitch extractors can't handle
* Screaming involves complex spectral characteristics and nonlinear vocal fold dynamics
* Sighing includes breath noise and transitional vocal states
* These techniques often involve multiple simultaneous sound sources (vocal folds + false folds + breath turbulence)

**Your cascaded approach becomes justified:**

* Different stages could handle different acoustic phenomena (subharmonics vs. noise vs. harmonics)
* Coarse stage: Basic vocal fold vibration patterns
* Medium stage: Overtone structure and spectral shaping
* Fine stage: Breath noise, irregularities, and micro-textures

## **Potential Novel Contributions**

1. **New feature representations** for extended techniques (beyond F0/MFCC)
2. **Hierarchical modeling** of complex vocal phenomena that can't be captured in single-stage approaches
3. **First controllable synthesis** of these techniques (massive gap in the literature)
4. **Novel evaluation metrics** for non-traditional vocal sounds

## **Research Positioning**

This isn't just "voice synthesis" - it's  **computational modeling of extreme vocal biomechanics** . You'd be working at the intersection of:

* Audio signal processing
* Voice science/phonetics
* Generative modeling
* Musical expression

## **Strategic Recommendation**

**Reframe your conversation with your supervisor:**

* Emphasize that extended techniques require fundamentally different approaches than speech
* Standard conditioning methods assume "normal" vocal production
* Your cascaded approach addresses a real limitation in current methods
* This positions you in an almost completely unexplored research area

The PEFT papers your supervisor sent are relevant for efficiency, but they don't address the core challenge that extended vocal techniques may simply not be modelable with single-stage conditioning approaches.

 **Bottom line** : Your original instinct about needing more sophisticated architecture may be correct for this specific domain.

## Kelsey

Is there a way to bridge the concern my supervisor seems to have?

## Response

Yes, there's a strategic approach that could satisfy both your research ambitions and your supervisor's practical concerns:

## **A Progressive Research Strategy**

**Phase 1: Prove the Problem Exists (PEFT approach)**

* Start with the simpler PEFT conditioning approach your supervisor suggests
* Apply it specifically to extended vocal techniques dataset
* Document where it fails - this becomes your motivation for more complex methods
* Use established architectures but novel evaluation metrics for extended techniques

**Phase 2: Justify Complexity (Cascaded approach)**

* With clear evidence that simple conditioning fails for your specific use case
* Develop your cascaded diffusion approach as a solution to documented limitations
* Position it as "necessary complexity for a genuinely difficult problem"

## **Immediate Diplomatic Strategy**

**Reframe your next conversation:**

1. **Acknowledge their expertise** : "The PEFT approaches are clearly more efficient for standard audio synthesis"
2. **Position as empirical question** : "I'd like to test whether simple conditioning can handle extended techniques before assuming we need complexity"
3. **Propose phased approach** : "Could I start with LoRA conditioning to establish baseline performance, then pursue cascaded methods only if results justify it?"

## **Technical Compromise**

**Hybrid architecture:**

* Single diffusion model with PEFT conditioning (supervisor's preference)
* But design conditioning mechanism specifically for extended techniques
* Use multiple conditioning pathways (F0, spectral irregularity, noise characteristics)
* Apply LoRA to different parts of the network for different voice characteristics

## **Research Framing Benefits**

This approach gives you:

* **Supervisor buy-in** : You're following their technical guidance
* **Empirical foundation** : Clear evidence if/when simple methods fail
* **Fallback position** : Even if simple methods work, you've contributed novel evaluation and datasets for extended techniques
* **Escalation path** : Documented justification for complexity if needed

 **Key insight** : Frame this as "efficient exploration of solution space" rather than "compromise away from ideal approach." Your supervisor likely wants to see you can work systematically through increasing complexity rather than jumping to the most complex solution first.

## Kelsey

Ok. So let's make a game plan.

Given that I have already had success training the PyTorch Diffusion Model, what would be my next steps to attempt

A: Single diffusion model with PEFT conditioning (supervisor's preference)
B: conditioning mechanism specifically for extended techniques
C: Use multiple conditioning pathways (F0, spectral irregularity, noise characteristics)
D:Apply LoRA to different parts of the network for different voice characteristics

## Response

Here's a concrete game plan, building on your existing PyTorch diffusion model:

## **Phase 1: Dataset and Feature Extraction (2-3 weeks)**

**A. Collect Extended Vocal Techniques Dataset**

* Record or source examples of vocal fry, screaming, sighing, breathy voice
* Ensure high-quality recordings (24kHz+) with clean isolated vocals
* Aim for ~30-60 minutes per technique initially

**B. Develop Feature Extraction Pipeline**

* **F0 extraction** : Use CREPE or YIN for fundamental frequency (handles irregular pitch better than standard methods)
* **Spectral irregularity** : Implement harmonic-to-noise ratio, spectral centroid variation over time
* **Noise characteristics** : Extract spectral rolloff, zero-crossing rate, breath noise detection
* Create time-aligned feature vectors that match your diffusion model's temporal resolution

## **Phase 2: Simple LoRA Conditioning (2-3 weeks)**

**C. Implement Basic Conditioning**

* Add LoRA adapters to your existing U-Net at bottleneck layers
* Start with single conditioning pathway (F0 only)
* Inject features via cross-attention or FiLM layers
* Keep base model frozen, train only LoRA weights

**D. Baseline Evaluation**

* Train on standard speech first to verify conditioning works
* Then test on extended techniques dataset
* Document failure modes - this becomes your research motivation

## **Phase 3: Multi-Pathway Conditioning (3-4 weeks)**

**E. Expand Conditioning Architecture**

* Apply different LoRA adapters to different network layers:
  * Early layers: Noise/breath characteristics
  * Middle layers: Spectral irregularity
  * Later layers: Fine-grained pitch control
* Implement multiple conditioning pathways simultaneously

**F. Evaluation Framework**

* Develop metrics specific to extended techniques (spectral deviation, pitch stability measures)
* User studies with musicians familiar with these techniques

## **Immediate Next Steps (This Week)**

1. **Literature review** : Search for any existing work on computational modeling of extended vocal techniques
2. **Dataset planning** : Identify sources for extended technique recordings
3. **Feature extraction research** : Investigate which audio features best capture the characteristics you want to control
4. **Meeting prep** : Prepare this phased approach to present to your supervisor

## **Key Implementation Details**

* Use your existing LJ Speech trained model as the base
* Apply conditioning at multiple resolution levels within the U-Net
* Start with lower-dimensional conditioning vectors (16-64 dims) and expand if needed
* Track training metrics that measure conditioning effectiveness, not just audio quality

This approach gives you empirical evidence for whether simple conditioning suffices, while setting up the foundation for cascaded methods if needed.

## Kelsey

Do you have suggestions for voice datasets containing examples of extended vocal techniques?

## Response

Finding datasets with extended vocal techniques is challenging since most existing audio datasets focus on standard speech or singing. Here are your best options:

## **Academic/Research Datasets**

**Voice Science Collections:**

* **PVQD (Pathological Voice Quality Database)** - Contains vocal disorders including vocal fry, but primarily clinical rather than musical
* **Saarbruecken Voice Database** - Has some non-modal phonation examples
* **HINT (Hearing in Noise Test)** - Limited, but may have some breath-related vocalizations

**Music-focused:**

* **MIR-1K** - Primarily standard singing, but may contain some extended techniques in contemporary pieces
* **SingME** - Small dataset, check for any contemporary vocal music

## **Practical Collection Strategy**

**YouTube/Online Sources:**

* Extended technique tutorials by vocal coaches
* Contemporary classical performances (Cathy Berberian, Diamanda Galás)
* Metal vocal technique videos
* Be aware of copyright restrictions - you may need to create derivative works or get permissions

**Recording Your Own:**

* Work with local singers, voice students, or drama departments
* Record controlled examples of each technique
* This gives you clean, consistent data but requires significant time investment

**Collaborative Approach:**

* Partner with voice science researchers who may have existing recordings
* Contact music schools with contemporary vocal programs
* Reach out to composers working with extended techniques

## **Specific Challenges**

**Data Quality Issues:**

* Most existing examples are in musical contexts with accompaniment
* Isolation and clean recording of extended techniques is difficult
* Many techniques are inherently noisy, making feature extraction challenging

**Ethical Considerations:**

* Some extended techniques can be vocally demanding or potentially harmful
* Ensure any recordings are made safely with proper vocal training

## **Recommendation**

Start with a small, carefully curated dataset focusing on one technique (vocal fry is safest and most common). You can demonstrate your conditioning approach on limited data, then expand to other techniques once the methodology is proven. This also aligns with your supervisor's preference for starting simple before adding complexity.


### [kelsey-dl-model.md](kelsey-dl-model.md)

## Kelsey

Can you do a compare/contrast of the approach we have identified and discussed here versus the attached

## Response

Here's a detailed comparison between our strategic PEFT approach and your original cascaded diffusion proposal:

## **Architectural Complexity**

**Your Original Plan:**

* Three-stage cascaded diffusion (coarse → medium → fine)
* Separate models for each stage with different conditioning mechanisms
* Sequential processing with inter-stage dependencies
* Complex training pipeline requiring coordination between stages

**Our Strategic Approach:**

* Single diffusion model with PEFT conditioning
* Multiple conditioning pathways within one architecture
* Parallel processing of different voice characteristics
* Simpler training with frozen base model + trainable adapters

## **Research Focus Alignment**

**Your Original Plan:**

* Directly targets extended vocal techniques from the start
* Assumes complexity is necessary for the problem
* High-risk, high-reward research proposition

**Our Strategic Approach:**

* Empirically validates whether complexity is needed
* Provides fallback contributions even if simple methods work
* Lower risk while maintaining path to complexity

## **Supervisor Relationship Management**

**Your Original Plan:**

* Potentially conflicts with supervisor's efficiency concerns
* Requires convincing them that complexity is justified upfront
* May seem like ignoring their PEFT paper recommendations

**Our Strategic Approach:**

* Directly engages with supervisor's suggested methods
* Demonstrates consideration of their expertise
* Creates empirical evidence for decision-making

## **Timeline and Resources**

**Your Original Plan:**

* Longer development cycle (6+ months for basic implementation)
* Higher computational requirements (multiple model training)
* Greater risk of technical failures blocking progress

**Our Strategic Approach:**

* Faster initial results (2-3 months to baseline)
* Lower computational overhead (LoRA training only)
* Incremental progress with clear milestones

## **Research Contributions**

**Your Original Plan:**

* Novel architecture for extended vocal techniques
* Potentially groundbreaking if successful
* High novelty but uncertain feasibility

**Our Strategic Approach:**

* Novel evaluation metrics and dataset for extended techniques
* Systematic comparison of conditioning approaches
* Guaranteed contributions even if simple methods succeed

## **Critical Assessment**

**Your original cascaded approach may still be superior for extended vocal techniques** because:

* These techniques involve fundamentally different acoustic phenomena
* Single-stage conditioning may genuinely be insufficient
* The complexity could be scientifically justified

**However, our strategic approach:**

* Provides empirical evidence for this claim
* Builds supervisor trust and collaboration
* Creates publishable intermediate results
* Maintains option to escalate to cascaded methods

## **Recommendation**

Your original instincts about needing cascaded diffusion for extended vocal techniques may be correct, but the strategic PEFT approach gives you a defensible path to prove this necessity rather than asserting it. This addresses both the scientific rigor your supervisor likely wants to see and your research ambitions for tackling genuinely challenging problems.