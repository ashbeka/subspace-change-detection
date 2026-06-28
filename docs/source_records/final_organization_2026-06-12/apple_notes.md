Research notes

- I hear PixelHop and PixelHop++ allows the model to capture spatial relationships and extract multi-scale features utilizing Successive Subspace Learning (SSL). I think this idea could be incredibly useful to help us extract the spatial information from the satellite images as we’re suffering with that.
- Green learning idea https://www.sciencedirect.com/science/article/pii/S104732032200205X , senpai told me it could be a really useful idea to apply into my research, he mentioned how we would have an information rich satellite image, then, instead of making one subspace per one 13-D vector, we have multiple hierarchy subspace levels, don’t remember exactly how the idea was, but it roughly was something like: start L1 buy having the whole image as a subspace, then in L2 have four subspaces representing the image, then in L3 have 16 subspaces representing the satellite image, and so on so fourth with other levels, then do some sort of summation for all these levels to get both the general features and finite features of the image, at least that’s what I remember. And then I would do the same thing for the same image but at a different time? I think but I’m not sure.
- While discussing green learning, senpai mentioned the idea of wavelet transform, he said this is more or less the concept where he get the inspiration from, and that it would be increadibly fruitful if I can learn it and try to implement it for my case. I think the Green learning research paper mentioned it too.
- In addition to explaining GL and wavelet transforms, he mentioned the idea of how image compression works and he sent this video https://www.youtube.com/watch?v=Kv1Hiv3ox8I , he told me to read about it and look how its similar to what I’m trying to do.
- The image below is from our meeting discussing the green learning, wavelet transforms, and image compression ideas.
￼**image1 goes here**
- I discussed an idea with my senpai which is an almost full subspace framework, what do I mean by that, in short, traditionally our lab methods revolve around having pipeline that consists of two main parts, a geometriacll processing part (subspaces), and a neural network processing part (DNN, transformers, etc…). So, for the first part, with our current tooling methods like DS, SSC, GL, Geodesic, Grassmanian, or other unmentioned methods, is it possible to have some sort of pipeline that is semi-unsupervised where we input the 13-D satellite image, and detect the change of the same image from two or more different timestamps, preserve the spatial information of the image, detect the change with DS, interpret the general change type with some sort of clustering like SSC or SAM-3 or whatever clustering that would make sense of the change, cause unless the change has been already manually labeled in some sort of benchmark like OSCD and we want to compete on which method accurately gets the change with the best metrics possible, how would we make sense of the change, I mean, what do we make of the white pixels that we get as change? Could we think of clustering similar change type together like urban/vegetation/industrial and see how our method competes? (We’d still need a benchmark). So that would be making change detection, but interpretable, at least by clustering then giving some meaningful labels to the cluster after comparing to the group truth. 
- When it comes to the research idea as a whole, maybe the answer is
    - Yes, subspace methods are useful, but only as a lightweight / label-efficient prior.
    - Yes, but not for full segmentation, more for feature extraction, anomaly scoring, or pre-training.
    - Maybe, but your current implementation is not the right framing.
    - No, not enough novelty, and you should pivot toward a different contribution while keeping some subspace idea as analysis.
    - or maybe it’s NONE OF THEM, do not get restricted to the four aforementioned ideas cause its dangerous to limit your mind.
- Ask “What problems exist in satellite change detection, and is there any place where subspace methods are genuinely useful?
- Why should subspace methods help change detection?
- What weakness of existing change detection methods do they address?
- What setting are they especially good for?
- Low labels? small data? interpretability? computational efficiency? robustness? temporal structure? disaster scenarios?
- What weakness of existing change detection methods do they address?
- Read and take note of some of the important references in “Time-series Anomaly Detection based on Difference Subspace between Signal Subspaces” paper.
- Maybe re-frame the research around “how to recognize pseudo-change from real change”? Like how pseudo-change has shadows, seasonality, registration difference, sensor effects.
- Should we try to represent the data in a covariance matrix to learn the hidden underlying patterns at work in our data regarding how the inner-works of data/channels affect each other? Maybe perform Eigen-decomposition on the covariant matrix?
- The change detection images that result from applying DS show a mostly correct segmentation result, could we steer the project as DS applied in CD producing good reliable segmentation maps?
- Don’t forget to apply naive methods like Euclidean difference and least squares idea to establish why these simple methods “suck”.
- contribution can be clearer: “GeoAI/deep models give strong end-to-end predictions, but our subspace-based method explores whether low-dimensional temporal-change structure can provide a more label-efficient and interpretable signal for disaster damage/change detection.”
- classes, patches, time windows, or image regions
- PCA-diff/my method ig gives one global change representation, while subspace clustering tries to discover multiple local/semantic change structures.
- PCA-diff: one global subspace of differences.
- Subspace clustering: many subspaces, possibly corresponding to different change patterns.
- Maybe a Possible idea:
    1. Detect changed areas using your DS/SSC method.
    2. Crop changed regions.
    3. Use GeoAI/SAM/CLIP-style tools to segment or describe objects inside the changed area.
    4. Cluster similar changes.
    5. Assign rough semantic labels such as:
        * building removed
        * new construction
        * road change
        * vegetation loss
        * flood/water expansion
        * damaged urban block

- novelty can be: subspace-based, label-efficient, interpretable change/damage detection, with GeoAI/ChangeStar/U-Net used as comparison and support tools.
- Detect temporal change with DS/take greenhouse patches and use them to train a CNN -> cluster similar change types -> use different combination of bands to detect different changes (some changes emit co2 emission like greenhouses)
- Create a google earth pro plugin that utilizes the KDS/KGDS on greenhouses specifically (or just change that is detected then clustered, or the change level severity over the years to see what most changed) and test its performance. (Problem? CD over different periods of times, everything changes doesn’t it (at least on the atmospheric level)? How can one obtain only meaningful change, and how do you categorize this change, at least initially with clustering then assign meaningful label/name for each cluster, could different channels work better with each other than other channels, or is the naive approach of using all channels better? Or simple RGB change?
- What about the issue of repeatable change? Such as rice fields, how they go green in spring and barren in every other season, how do we treat such change?
- Build a dataset to train subspaces on for clarifying what type of change is the detected change in the DS change maps, so that the DS maps of change can make sense (what type of change is what: agricultural, industrial, urban, etc…)
- Note for me to work on: you don’t have to use all 13-D in a satellite image to perform change detection, it could very well be that using some combination of bands that make sense to work together like say bands RGB with band NVDI and band humidity, having only those working together to make a subspace or a different combination of bands (we can make an automated system that tests all the different combinations of 13-D with either 1 band, 2 bands, 3 bands, 4 bands, etc…., and each band get combined with any other band to discover the different combination, then test out the change detection on and see which perform best and what insightful information each combination could give us.
- “Why this (maybe) matter: Linear PCA can only find straight weighted combinations of the 13 bands. KPCA can capture nonlinear relationships between band values. For example, if changed and unchanged pixels are separable only through a curved relationship between NIR/SWIR/visible bands, KPCA/KDS might capture something linear DS misses.”

Latest most important feedback (after presentation on 2025/12/16):

- I should understand how the subspace method was implemented for my case, for my project. I need to properly implement subspace if it wasn’t properly implemnted (do we use the one subspace per channel approach in oppose to one subspace per whole image approach where each vector is the 13-D pixel, or are we using the one vector per whole channel and subspace per whole 13 channels where each channel is a vector approach, or are we implementing it differently?), we need to understand how we implemented subspaces in our project, it goes back to me relying on AI tools to do the whole job for me instead of reviewing and understanding how the techniques were implanted and wether they were implemented properly or not. Sensei and my senpais told me to understand what I’m doing.
- What is the subspace here? What’s the dimension of the subspace (number of PCs not the space dimension) at hand? Are we losing information of each pixel vector?
- 13 channels, is each channel represented as a vector or subspace?
- Experiment with the two methods from the two papers sensei mentioned:
    - First paper: S3CCA: Smoothly Structured Sparse CCA For Partial Pattern Matching. Link: https://staff.aist.go.jp/takumi.kobayashi/publication/2014/ICPR2014.pdf 
    - Spotting finger spelled words from sign language video by temporally regularized canonical component analysis. Link: https://www.researchgate.net/profile/Kazuhiro-Fukui/publication/303563203_Spotting_fingerspelled_words_from_sign_language_video_by_temporally_regularized_canonical_component_analysis/links/5a8a413faca272017e621f5d/Spotting-fingerspelled-words-from-sign-language-video-by-temporally-regularized-canonical-component-analysis.pdf 
- Sensei said to look into Canonical Correlation Analysis (CCA).
- Current implementation of Subspace —> 5 PCs in 13-D space: dimensions = (5,13)  (I forgot what this notes meant, could be that we should try to make the subspace dimensions to adapt to the (5, 13) dimension if they weren’t already like that?
- How did the projection back to image from PCA/Subspace happen (math) in our project if it happened in the first place? Sensei asked that question.
- Have you tried training U-Net with the raw data without Otsu thresholding or no thresholding at all?
- Sensei asked me to look into K-PCA implementation, I don’t know why or how.
- Sensei asked me to look into Kernel PCA, I don’t know why or how. Try the Kernel PCA trick.
-  experiment with Itsu thresholding before U-Net training and after.
- Investigate the lost information of position/of each pixel vector if that was the case in our project.
- investigate the projection: how did we exactly extract the difference? How projection was done? How we extracted outputs?
- Projection energy? What’s that? Calculated how?
- Understand projection magnitude and how it was used here, how?
- 5 dimensional subspaces in 13-D constructed from 256x256 pixels being small or large?
- What’s the difference between DS and GDS? Understand that well, see which one was actually implemented here and learn.
- In method 1 each image is one vector, right?
- How did I get the pixel? (I forgot what the question meant at the time)
- What’s the number of (data pints and features) dimensions of input before we put it into PCA?
- Train raw data on unit without any thresholding, but normalize first, you could try for example the normalization from 0 to 1, doesn’t have to be only that form of normalization could be others.
- Understand how to project back the change onto the original image or the pixel space, like DS has different space so how do we project back?
- In K-PCA we can map low dimensional vector into high demential space like rows 256x256 and columns 13, “(dimension of first space become bigger than the number of datapoints)?” Gaussian spca and stuff (circles around and triangles inside example).

Old still important research notes:
￼**image2 goes here**
- discuss the application of DS and geodesic decomposition to various types of temporal sleep data. (Different research don’t mind this bullet point)
- Do DS on OSCD or other dataset to get change maps on all channels, then segment/classify each change into its own type.
- Have some sort of semantic change detection algorithm/method that detects what the change type is and how it progressed.
- Make a U-Net similar style architecture but for SM.
- Orientate the research to recognize what sort of change there is using DS and SM, differentiate change and label automatically maybe use SSC.
- Figure out if there’s a way to overcome seasonal change (such as snow).
- If this technology were to be used for purposes other than disaster damage assessment, what potential applications could be considered?
- Separate vegetation change from real change.
- Isolate spectral change in satellite imagery.
- SM, DS for spectral change isolation.
- Research Gap: Lack of Temporal Subspace Methods for Multi-Date Satellite Image Analysis in Rapid Disaster Assessment

1- since all phases rely on the same data, and we’re about to add xBD data, maybe we need to add the data into root level and adjust all the code that relies on it to point its way
2- regarding MultiSenGE, its not really convincing how we used it in the project so far, what do you recommend we use it (actually use it) in the project it to utilizer it?

Thank you, another thing, If I want to familiarize ChatGPT that lives on the web browser that apparently has better reasoning and researching ability than you, when by you I mean Codex who lives in the CLI, and I want to familiarize it with the project that we're currently working on, what document or documents should I give it so that it understands the full context, the full context with details, with thorough details of everything, what the project is, the goal, the limitation, the implementation, the code, the specifications, the pipeline, and any other, I'm just like saying words here, any other aspect of the project Do I give it like a single file? Do I give it like multiple files? Should I select like certain files from this project? Should I ask you to generate like one file that has everything specified in it? Maybe like one like mega file that has every single detail or almost all details about the whole project from the beginning to the end with the current results, or what do you think we should do? Discuss with me before you give any answer

Now, regarding the MultiSenGE dataset that we use in our research project, here's the thing. I'm sort of on the fence on if it's a good idea to keep it in our project because so far, the datasets that we have or we plan to use in our project are three. The first one is the OSCD, second one is the xBD, third one is MultiSenGE, and I thought we could utilize all of these. And I'm not sure what we exactly did in the project with the MultiSenGE dataset, but I think we only use it for testing or something. I'm not sure. So my question is, or rather my discussion with you is, what are we really doing here with the MultiSenGE dataset? I have a seminar coming up, and I'm going to present my research with my sense, and I'm going to have hard questions, and I don't think I'll be able to answer such questions when it comes to the MultiSenGE dataset in our dataset. What are we exactly utilizing it in our research, if any? I mean, testing, if the method works, that's it, really? Is that supposed to be a thing? I mean, sure, you can test it to see if the method works, but it's a huge dataset, and just using it for testing kind of doesn't make sense. Unless we utilize it in making the model, I mean, really utilize the dataset, the huge dataset, then maybe that would make sense. So I'm asking you, can you do your research, read the whole research that we've written so far? I also added the research notes repository to this project, which is the place where I write my research notes, and drafts, and everything, and ideas, and understand how can we, I mean, also remember, if we don't need to use this dataset, the MultiSenGE dataset, then we don't have to. We don't have to force ourselves to use something that might not be the best decision for our project. So yeah, tell me if there's a better way to use it, if we really need to use it, and how can we use it for better utilization, instead of just using it to test out the method? That's my understanding, forgive me for my poor understanding. But yeah, help me out here. Before you do any code changes or any change, discuss with me what you think.

We clarified that “change” in our project can mean binary change detection (changed vs unchanged) or the stronger task Semantic Change Detection (SCD), where each pixel is labeled by a from→to transition (e.g., vegetation→building) or, in a disaster framing, damage-state transitions (undamaged→destroyed, etc.). SCD is a real, established remote-sensing problem, but it’s harder because it needs semantic labels at both times and deals with imbalance. The key research idea we identified is that subspace methods (MSM/CMSM/GDS) could be repurposed from “image-set discrimination” into remote sensing by treating local feature neighborhoods (patch features) as subspaces and using subspace geometry (canonical-angle similarity / constrained separation) to produce more robust, data-efficient, interpretable semantics, either as (A) a standalone SCD pipeline (encode→local subspace→classify at t1 ,t2 →derive from→to) or (B) a hybrid deep SCD model where a subspace head or subspace-constraint loss regularizes the network for better separation under nuisance factors (seasonality, illumination, misalignment). The publishable claim is not “subspaces are magic,” but that adding subspace geometry can yield lighter models / fewer labels / improved robustness and interpretability compared to standard Siamese/fusion SCD baselines.

3- Hi Abood, thank you for the nice presentation! I am sure that we can extend the current research in various directions. For example, I am just wondering how we use geodesic information between two subspaces, as it can relate to smooth changes. Besides, although this idea is somewhat distant from computer vision, the explanations that VLM outputs for a set of satellite images may work, where each explanation is represented as a subspace. Checking its first- and second-order variations, considering the geodesic, may yield some interesting information. Can you share the slides used today with me?
Thanks for sharing the slides. We have recently proved mathematically that the geodesic projection works as expected. So, we can use it with ease.


——————————

All right, note this. I'm gonna try to make a convincing statement, problem statement, for my research by saying, for example, pixel-wise difference in change detection is naïve and look it only, for example, it can only show RGB difference, RGB bands difference, but what about if we want to visualize a change between multiple bands, as in, for example, satellite images have much more than three bands, how are we able to visualize that change from the other bands in addition with RGB band or as a standalone? Thus, our idea of difference of space, and so on and so forth. It could be a convincing statement, I'm not sure. If it comes to the point of explaining difference of space and our experiment, we could mention that the difference of space on images might not have been properly implemented because the idea of difference of space is relatively novel and, moreover, applying difference of space on satellite images has never been done before, so it's a bit tricky to optimize the implementation specifically for satellite images format. Something along these lines. I'm just giving random notes that could help me in the presentation soon. 

————————

Not sure how to articulate this, but regarding the research, I thought about how our project research is currently being implemented, specifically phase 2. I'm still really not aware or understanding of the whole thing, the whole structure in general, but I thought about the idea of, okay, we are training our model on difference, subspace, change data, change features, change maps, and we have the OSCD datasets that we're training on, and maybe like the XPD dataset that we can train on too. So how about we make some sort of a service, some sort of method that we can use that runs all over Sentinel-2 satellite images, globally, all over the world, automatically. It could, for example, download a 100GB patch of data, run it over, give the change features, change maps, then discard the 100GB of data, and then get another 100GB of data. Or it could simply loop over the established datasets, like the Multi-SIN GE or the Multi-SIN NA datasets, so that it can somehow generate change features, and the idea is basically automatically generating different subspace feature, different subspace change maps automatically, and maybe we could try to make some sort of a dataset, right? Like, for example, change detection or like a labeled change detection dataset, after we apply our current methodology of different subspace and training on the model-based unit, and then the whole thing with the SSC, sparse subspace clustering. I'm not sure. There is real potential here to do good work. 



————
Alright, so to give you a short update of what's happening, there's this, um, right now we're in the research notes repository, um, the one where we keep all of our research notes and planning and everything, and, um, I took the, uh, phase one spec sheet, uh, documents, and I put it in a different project, and it's been running for the past, like, 12 hours. So it's, it's like, uh, implementing the codes for the, uh, phase one, and I think it's doing a good job. So, um, I sort of want to, um, give you a heads up of what's, what's gonna happen next, uh, because here's the thing. I, I have, I have this, um, this research notes repository that is currently running here, and I have, again, the phase one repository that is running, and hopefully I can advance to phase two soon, but, um, I need, I need to, um, do you see all of these things? All, all, this is basically my research. All of this is my research. I need to, um, make two things to be able to show the research, because so far, I know the research is for me, but I don't have the tools or the ability to show it to, uh, my sensei or my presentation, which is happening soon, and they, I was requested two main things. First of all, they requested me, um, a document, like, for example, a PDF page that is at least four pages, that's firstly, and secondly, a presentation that is at least 20 minutes, right? Now, I'm not asking you to make the, uh, document and the presentation, but simply I want you to, um, um, note the instructions for how we're gonna write the, um, the notes, research notes, and the instructions. Uh, so, I mean, I wanted to note that how we're gonna write, like, the, um, research document is at least four pages, which, you know, I'm gonna, like, do, I'm gonna, like, um, try to, uh, like, paste it and, like, something like overleaf and, you know, like, write it, but, like, um, still, I need that, um, document and instruction of how to write it, which I'm gonna assign it to you, and the presentation, which is very, very, very important. Uh, they ask for at least 20 minutes, and, you know, like, how a typical research presentation goes. I am not sure, like, but, like, I think, like, we need to establish the methodology quite well. We're gonna need to, uh, like, we're gonna need to have, like, uh, figures, graphs, references, but, yeah, figures, graphs, references, methodology, the pipeline, um, the output or the expected output, the objective, and the statement, and the problem, and all of these, like, elements, and there are other important elements, which I haven't mentioned yet, which I'm sure you're well aware of, so keep that in mind. So, um, can you, uh, keep this in mind, because we, first of all, have phase one, and I think I want to advance into phase two, and connect it with phase one, and then, uh, work on the research document and the presentation, because by this point, we're gonna have the phase one and phase two. What do you think?


——————


Using our research DS+U-Net to label MultiSenGE/NA or xBD-S12 (if it wasn’t) -unlabeled dev playground for DS behaviour/visualization- 


Resnet50 on image net
- Vectorize 6 images each image is one vector (by resnet60)
- Each 6 vectors becomes one subspace, then you calculate DS between each six vectors

“We adapt and systematically evaluate Difference-Subspace change maps on Sentinel-2 (OSCD/MultiSenGE) and integrate them into an ordinal building-damage mapping pipeline”
rather than “First method to use subspace/difference for change detection”.

Geodesic change detection methods. Elevation, building, height, change detection map, damage map, before, after, heat map, red, blue. Reconstruction of building based on PCA change detection method. Geodesic change detection method. Reconstruction of building based on method. Geodesic change detection method.


By building-level descriptors, we basically mean:
Feature vectors that summarize one whole building as a single object, instead of looking at individual pixels.
So: one building → one descriptor (one feature vector).
In satellite / aerial imagery you can look at data at different levels:
* Pixel-level: features per pixel (spectral bands, NDVI, etc.)
* Patch-level: features per small image patch (e.g., 32×32 window)
* Building-level: features for the entire building footprint or mask
Building-level is object-centric: you treat each building as a single instance.This is useful for damage assessment, land-use / land-cover at object level, etc.

They embody characteristics used to define and evaluate a building's performance, features, or operational status for specific purposes such as sustainability assessments, performance evaluations, or architectural modeling. 

What can building-level descriptor contain?

(a) Geometric / shape features
From the building footprint or mask:
* Area, perimeter
* Compactness, elongation, rectangularity
* Height estimates (from DSM/DEM if available)
* Aspect ratio, orientation
(b) Spectral / radiometric features
Aggregate the spectral information over all pixels belonging to the building:
* Mean / median / std of each band (R, G, B, NIR, etc.)
* Indices like NDVI, NDBI averaged over the building
* Histograms of band values
(c) Texture features
Texture computed within the building mask:
* GLCM features (contrast, homogeneity, entropy, etc.)
* Local binary pattern (LBP) histograms
* Simple measures: variance, edge density, gradient stats
(d) Context / neighborhood features
Where the building sits and what’s around it:
* Distance to road / coastline / river
* Surrounding land-use type within some radius
* Density of nearby buildings (crowded vs isolated)
* Average height / area of neighbors
(e) Temporal / change features
For pre/post disaster (or multi-temporal) imagery:
* Change in mean spectral values (pre vs post)
* Change in NDVI, NDBI, etc.
* Change in texture metrics
* Difference of features: f post - f pref_\text{post} - f_\text{pre}fpost −fpre 
* Or concatenated [pre_features, post_features] in one vector

How do we get them in practice?
Pipeline idea:
1. Detect / segment buildings
    * From a building footprint dataset (OpenStreetMap, city GIS), or
    * From a building segmentation model (like a U-Net).
2. For each building mask:
    * Collect all pixels belonging to that building.
    * Compute geometry features from the polygon/mask.
    * Compute spectral stats, texture metrics, temporal differences.
    * Concatenate into a single feature vector.

How can this connect to our research?
In your damage & land-use project:
* Input level
    * You can work at pixel-level (segmentation masks), or
    * You can aggregate up to building-level descriptors.
* Damage classificationFor each building, you could:
    * Use a classifier (random forest, SVM, small MLP, even SSC clusters) on its descriptor to predict:
        * damaged / undamaged, or
        * damage grades (no/slight/moderate/severe).
* Temporal analysisYou can track how each building’s descriptor evolves over time:
    * Pre-disaster vs different post-disaster dates → to analyze recovery.
    * Clustering trajectories of building-level descriptors to find patterns:
        * quickly repaired vs slowly repaired vs never repaired buildings.
* Subspace / PCA angleYou can place building-level descriptors into:
    * A PCA subspace to visualize them (PC1–PC2 plots for damage classes).
    * A difference subspace / SSC framework to cluster buildings based on temporal change behavior instead of just pixel changes.



PCA image reconstruction (the medium)

For multispectral satellite images, PCA reconstruction can have two purposed: post DS/SSC processing visualization, or Compression & de-noising, or other purposes.

Key word: Spectral–temporal vectors

-potential uses for PCA reconstruction-: 
* “PCA features” = the low-dim vector z
* “PCA feature reconstruction” = take z and map it back to get f^\hat{f} f^ in the original feature coordinates.
* The error tells you how well this feature fits the “typical” subspace spanned by the PCs.
You can do this for:
* Pixel spectral features
* Building-level descriptors
* Intermediate features in your damage-segmentation network.


We can do Change detection via reconstruction error (baseline)
Very straightforward idea you could use (or cite as baseline):
1. Train PCA on pre-disaster pixel vectors (or pre+post, but labeled as “normal”).
2. For each pixel / patch in the post-disaster image:
    * Computer projection z and reconstruction x hat
    * Use reconstruction error as change score
3. Threshold or cluster the error map to get a binary change map.
Intuition:If a post-disaster pixel is “compatible” with the pre-disaster subspace, it reconstructs well ⇒ low error ⇒ likely unchanged.If it is very different (e.g. building destroyed, flooded area), it won’t fit the old subspace ⇒ high error ⇒ changed.

We can use it for Dimensionality reduction before SSC or clustering
PCA on U-Net / encoder features
Later, when you have a U-Net (or similar) trained for damage segmentation:
* Extract deep feature vectors from the encoder for many pixels/buildings.
* Run PCA on these deep features.
You can then:
1. Visualize them in 2D/3D (PC1, PC2, PC3) to see how damage levels or land-use classes distribute.
2. Use reconstruction error in the feature space as a sort of anomaly/damage indicator:
    * Train PCA on “normal” (undamaged) features.
    * For new images, high feature-space recon error could mark unusual, possibly damaged structures.
This mixes your “subspace mindset” with deep learning in a nice, explainable way.
Denoising / cloud & shadow mitigation
Multi-temporal satellite images often have:
* Noise, sensor artifacts.
* Clouds, shadows.
PCA reconstruction with only the leading components can act as a linear denoiser:
* It tends to keep stable, consistent structures (buildings, roads, land-use patterns).
* It can suppress random noise and sometimes small, inconsistent artifacts.
You could, for example:
* Pre-process your images with PCA reconstruction before feeding them to U-Net or your change detector.
* Or compare “original – PCA reconstruction” to highlight “stuff that doesn’t fit the main data subspace” (potentially useful for anomaly detection).

TLDR:
PCA reconstruction = project onto a low-dimensional subspace and project back; the difference measures “how typical” a sample is.
PCA image reconstruction = same thing but each data vector is an image; useful for compression, denoising, visualization.
PCA feature reconstruction = same idea on feature vectors (e.g., deep features, spectral–temporal features); reconstruction error can be an anomaly/change score.
For your research, PCA is:
* A canonical way to build subspaces (pre/post) for simple change detection.
* A dimensionality-reduction step before SSC or other methods.
* A baseline / interpretability tool to relate your fancy subspace + U-Net methods to something classical and easy to explain.

- The field of Change Detection (CD) is broadly divided into two sub-fields: Binary Change Detection (BCD) and the more complex Semantic Change Detection (SCD). The latter is more challenging as it requires identifying the ”from-to” semantic information, not just the presence of change.
- I GOT IT! Do a sliding windows change detection (but a new method here), where we, say have 10 images of the same patch taken at 10 dates, we perform Subspace methods on each image, then GDS on image 1 with image 2, then image 2 with image 3, then image 3 with image 4, so one so forth until we get to image 9 with image 10 (alternatively we get play with this we we do GDS on image 1 with image 2 then get the output, and USE THAT OUTPUT  and continue merging change images!! could use different sources as in different satellite imagery (adjust rotation and geo data for all the images to match), or use multiple dates depending on what we want to know.
- “Self-Supervised Learning: The AI generates its own labels by hiding parts of the training data (like masking a word in a sentence) and trying to predict it.” We could use the same idea for GDS
- DGM (difference guiding model)
- From the OSCD guys "The networks should ideally be able to learn to differentiate between artificialization [man-made] changes and natural changes, given that only artificialization changes are labelled as changes on the dataset."
- SERIOUSLY we could target contests like Data Fusion contest https://www.grss-ieee.org/community/technical-committees/2026-data-fusion-contest/ and use our methods.
- We could target the task of “data fusion” like https://ieeexplore.ieee.org/document/9246669 
- Use geometrical/subspace methods where neural networks are strong but insufficient: interpretability, low-label settings, temporal structure, diagnostic analysis, pseudo-change filtering, and hybrid pipelines.
- Understand if the difference is related to change or due to other factors such as radiometric calibration of an image, faulty data, etc..
- “If labels are abundant and the only goal is highest IoU/F1: use a strong neural network. If the goal includes interpretation, limited labels, temporal/geometric structure, diagnosing change types, or complementing neural models:  geometrical/subspace methods still have a role.”
- “DS/GDS can say something like: this change direction is caused by these spectral/spatial/date subspace differences. A neural network may give a good mask but not explain the geometric structure of the change”
- Look for and play with rich high band satellite imagery that captures a lot more data than the measly (but still great) Sentinel-2 satellite. 
- Important: why don’t we extract the change with SOTA methods, then use GDS to make sense of the change (add semantics to it)?
- Learn how “Siamese Networks” work.
- “Pseudo-labels are common in semi-supervised remote-sensing change detection because dense change labels are expensive.” So we can target getting pseudo-labeling as s goal (Recent semi-supervised CD papers explicitly use pseudo-labels or teacher-student pseudo-label frameworks for unlabeled image pairs).
- AI said: “In this project, DS/PCA/IR-MAD maps are first treated as unsupervised geometric priors: continuous evidence maps that may guide or diagnose change detection. They become pseudo-labels only if we threshold or otherwise convert them into training targets for a neural model.” So maybe?
- “The same difference map can be either a prior or a pseudo-label depending on how we use it.”
- From Google: “Training change detection deep learning models involves taking two time-sequenced images, extracting spatio-temporal features, and outputting a pixel-level change mask.”
- Identify the exact computer vision task, some but not all:
    * Image recognition
    * Image classification
    * Object detection
    * Image segmentation
    * Object tracking
    * Scene understanding
    * Facial recognition
    * Pose estimation
    * Optical character recognition
    * Image generation
    * Visual inspection-
- What people say is: “Use multispectral when you need:
    * High spatial resolution (sub-metre)
    * Frequent revisit for time-series monitoring
    * Standard index calculations (NDVI, NDWI, NDMI)
    * Land cover classification and change detection
    * Cost-effective coverage of large areas
    * Quick turnaround and simple processing
Use hyperspectral when you need to:
    * Identify specific minerals in mining exploration or environmental assessment
    * Discriminate between similar plant species for biodiversity mapping or invasive species detection
    * Map soil properties like organic carbon, clay content, or contamination levels
    * Detect subtle crop stress before it shows up in standard NDVI
    * Monitor specific water pollutants or algal species
    * Quantify atmospheric gas concentrations“
- Hyperspectral processing needs “Dimensionality reduction using Principal Component Analysis (PCA) or Minimum Noise Fraction (MNF) transforms to identify the most information-rich band combinations “ So this could be our debut to introduce SM into the mix
- GREAT IDEA! We can use Google Earth Engine Apps like “https://sat-io.earthengine.app/view/tanager” to demonstrate our idea and research project when its done being implemented along comparison to our methods and why our win!
- “The common challenges identified from the literature are the difficulty in image acquisition, the large image sizes, computational complexity and the effect of noise on the satellite images. The main problems encountered as part of change detection is the poor accuracy and kappa coefficient values obtained as part of some misclassifications.” WHICH MEANS maybe we can target de-noising? Or accuracy?
- Something to do with the ES below to utilize Hyper-spectral Satellite images with SM would be really cool.
- Some approached I came across: “Siamese Networks: Two parallel identical subnetworks extract features from Time 1 and Time 2 independently before fusing them. Early Fusion (Image Stacking): Concatenating bi-temporal images along the channel dimension at the input layer.“
- It is binary remote-sensing change detection / binary change segmentation from bi-temporal multispectral satellite images. It is not: damage segmentation, damage severity classification, semantic change detection, building-level damage assessment, greenhouse mapping, multi-class land-cover change classification. YET.
- FLOW: If you are developing a new AI model, you use a dataset to train your system. You first test it against your own simple baseline (e.g., your previous version) to see if it works. Finally, you run it through a standardized benchmark to see how it compares to industry standards or competing models
- Like how U-Net and DeepLab and PSPNet are improvements for FCNs (which is an improvement for CNN) at least in the field of semantic segmentation, we can create our own NN method built on the state-of-the-art semantic segmentation but incorporate GDS in the middle of it (or before it?) 
- Why not use GDS for the task of image segmentation? It could work seeing how it produces the differences and we can do processing to make out these differences as segments (segmentation)
- Panoptic segmentation
- Latent space
- From google “Framework Optimization: Implement using platforms like TorchGeo (PyTorch) or Esri's arcgis.learn.”
- What is the action space permitted within our project? (Preferably the latest most relevant action space)
- Could re-orientate the project around Hyper-spectral Satellite images Anomaly detection (Maybe with GDS?)
- Learn how to preserve key information in spatial representation (satellite)
- What I just discussed with Jang: doing the 13-D channel flattening, could be better to flatten each channel by itself to preserve its own spatial information. What I thought of is, it could be interesting to flatten each of the 13 channels to get 13 (256x256-D vectors), then do PCA on them or GDS to get the spatial information on those data (all from the pre-data), then do the same thing for the post- data, and then do something in-between them? Anyways Jang told me to try flattening on the Z-axis i.e. 13 vectors (each vector has x-dimension)
- Focus on satellite images semantic segmentation
- Ask for the Full schematic of the method
- Using GDS for Anti-aliasing/aliasing (the thing in video games with they AAA/AA “Anti-Aliasing”). Anti-Aliasing baed SM for RS imagery (or whatnot), DS for Anti-Aliasing
- priors means: extra information we give to a model before or alongside learning, to guide it toward likely change areas.
- The answer lies within using GDS change pseudo-labels are training and utilizing the different the different channels to do better in depth clustering of (change/no change) of the change maps or better yet utilize the special characteristics of objects to cluster the objects in change maps to their own category (houses, waterbodies, sand, whatever) based on how each of them have special charachteristics in the satellite images.
- “In machine learning, latent means "hidden" or "underlying". It refers to variables, features, or representations that are not explicitly labeled or directly observed in the raw data, but are instead discovered and learned by the AI model.”, Raw data (like a high-resolution image or a long paragraph of text) is incredibly complex and takes up a massive amount of data. Instead of dealing with massive, raw data points (like every pixel in an image), algorithms compress this information into a smaller, abstract format. These condensed, "hidden" characteristics are the latent variables. ANALOGY: If you describe a close friend to someone, you don't list their DNA sequence or a pixel-by-pixel photo. You use abstract, meaningful coordinates like height, hair type, and personality. These descriptors are the "latent" features—the hidden, distilled essence of the massive amount of raw data that makes up your friend. latent variables and latent features mean the exact same thing.
- Why don’t we flip the pipeline? Use the best state of the art change detection method to get change while preserving the other channels data, then use GDS to make semantic segmentation/clustering of the data?
- Understand the feature extraction done in our project
- Maybe Foundation Models are the move
- The band(s) selected for the change detection process will depend on the project’s goal and the exact features preferred for evaluating the change.
- Modern satellite pipelines often merge both techniques:
    1. Clustering groups raw pixels into small, uniform vector shapes based on color and texture (often called "superpixels").
    2. Segmentation rules or classifiers then label those clustered shapes into actual semantic categories like "building" or "highway."
- To detect change, a developer uses segment-geospatial to extract vectors of structures at Time 1 and Time 2, and then uses rasterio to mathematically subtract or intersect those vector masks to flag exactly where structures appeared, grew, or vanished.
- OMG IM STUPID!!! Look at the note above this, to extract meaningful change, we could SEMANTICALLY SEGMENT IT (the pre- or post- or both images) BEFOREHAND THEN RUN THE CHANGE DETECTION METHOD THRU IT TO UNDERSTAND HOW A (Thing) that we know what it is changed. Contextual Tracking: Instead of seeing that "pixels turned gray," your system sees that a "Forest" object turned into a "Building" object. This gives you semantic meaning (urbanization/deforestation).
- Run SAM on your whole greenhouses patch on a date (segment it) then on another date (segment it) then run a CD method and ask it to give you visual tinder style slides of the specific object level (or at least some hyper parameter Meter level like 25x25 meter patch or whatever available patch) and decide which is a greenhouse (or ask SAM to get you what it think a greenhouse is) and see which has been abandoned or not.
- How the Process Actually Flows in Code
    1. Segment and Label (Pre-Image): Run segment-geospatial on your "Before" image to find boundaries, and label those polygons (e.g., Object A = Wetland).
    2. Overlay and Compare (Post-Image): You project those exact same "Before" object boundaries directly onto your "After" image.
    3. Extract the Change: You use rasterio to analyze what the pixels inside those specific boundaries look like now. If the pixels inside the "Wetland" boundary now match the spectral signature of dry soil, you instantly extract a meaningful change: Wetland Loss.
- MAYBE see if GDS can do the segment part? (And after the CD part)
- OSCD was manually labeled, they also say The paper clearly states that the dataset uses a simple "Change / No Change" binary format, and that the labelers intentionally tracked urban construction (artificialization) while ignoring nature. On the focus of the labels:"The dataset is focused on urban areas, labelling as Change only urban growth and changes and ignoring natural changes (e.g. vegetation growth or sea tides)." On the binary format:"A label is assigned to each pixel: change or no change." On what algorithms are supposed to learn:"The networks should ideally be able to learn to differentiate between artificialization [man-made] changes and natural changes, given that only artificialization changes are labelled as changes on the dataset."
- Really good figures:
Do this now:
Pick one small Harmonized Sentinel-2 area.
Build/query a time sequence for it.
Report: number of frames, dates, spacing, bands, cloud/no-data filtering, alignment assumptions.
Define exactly what “one subspace per date” means.
Then run first DS, second DS, and geodesic decomposition on that sequence.

Really informative search when I searched “state of the art current change detection benchmarks” I got:

“
The State-of-the-Art (SOTA) benchmarks for Change Detection (CD) evaluate how effectively machine learning models identify structural, environmental, or semantic alterations over time. While LEVIR-CD and WHU-CD remain the primary foundational standards for remote sensing, modern benchmarks have evolved toward multimodal data, vision-language context, and unconstrained 3D/viewpoint scene changes. [1, 2, 3]



Foundational Remote Sensing Benchmarks
These benchmarks represent the traditional, widely cited standards used to evaluate standard Deep Learning (CNN, Transformer, and Mamba-based) models. [1, 2, 3, 4]
* LEVIR-CD: Consists of high-resolution bitemporal Google Earth images focusing on long-term urban building changes like new construction and villa developments. [1, 2, 3, 4, 5]
* WHU-CD: A high-resolution satellite and aerial dataset centered on precise building extraction and geometric change tracking. [1]
* SECOND: A multiclass benchmark tailored for Semantic Change Detection (SCD), analyzing 6 different land-cover categories to track transitions like vegetation-to-concrete. [1, 2, 3, 4]
* S2Looking: Specializes in rural and side-looking satellite imagery, presenting major perspective distortion challenges. [1, 2, 3]



Next-Generation Vision-Language & Multi-Modal Benchmarks
Recent SOTA models leverage Multimodal Large Language Models (MLLMs) and Vision-Language Architectures (like UniChange or Seg2Change) to combine visual difference maps with linguistic context. [1, 2]

Benchmark Dataset [1, 2, 3, 4]	Primary Data Modalities	Focus and Core Task
NYC-CD (2026)	Street-View Images + Natural Language	Multiclass change tracking using visual-linguistic reasoning.
DVL-Suite (2025)	High-Res Temporal Images + Instructions	42 U.S. megacities over 18 years; tests regional urban narratives and pixel segmentation.
LSMD (2026)	Bitemporal RGB + Near-Infrared (NIR)	Focuses on very small structural building changes using spectral complementarity.
LEVIR-CC / DUBAI-CC	Satellite Imagery + Descriptive Text	Generates complex natural language sentences describing urbanization shifts (Change Captioning).


Real-World Scene & 3D Change Detection
Moving beyond strict top-down satellite views, these benchmarks test models on viewpoint variations, unconstrained camera poses, and egocentric video streams.
* SceneDiff Benchmark (2025): The first multiview change detection dataset providing instance-level object annotations across 50 real-world scenes extracted from egocentric and video pairs. [1]
* ChangeVPR / GeSCD (2024): Designed to test the generalizability zero-shot capabilities of models across unseen domains, including urban, suburban, and rural environments. [1]
* O-SCD Baseline Framework: Evaluates online, pose-agnostic scene change detection from continuous multi-view streams, frequently utilizing 3D Gaussian Splatting updating. [1]



Domain-Specific & Environmental Benchmarks
* EuroMineNet (2025): A continental-scale benchmark spanning 133 European mining sites over a 9-year span (2015-2024) to track gradual and abrupt land-use destruction.
* MineNetCD: A global mining benchmark featuring 70k paired high-resolution image patches across 100 global mine locations.
* RB-SCD (2025): Formulated explicitly for Road and Bridge Semantic Change Detection, charting 11 unique structural and infrastructure alterations across diverse international cities.
* xBD: The prominent gold standard for building damage assessment following natural disasters, using 4 distinct categorical severity levels. [1, 2, 3, 4, 5]
To help tailor this to your requirements, are you looking to test algorithms for satellite remote sensing, autonomous driving/street views, or 3D indoor environments? Let me know if you would also like to see the current leaderboards or metric frameworks (e.g., F1, IoU, CA-TIoU) for any of these specific datasets. [1, 2]
”
* Hyperspectral unmixing seems like an appropriate task for GDS/Geodesic
* Do we hunt Spatial context?
* Ideas:
multitemporal change detection;
change-point or anomaly detection;
temporal dynamics analysis;
detecting abrupt changes in an otherwise gradual process.
time-series decomposition;
denoising;
trend and periodicity extraction;
forecasting;
anomaly or change-point detection.

* “Subspace method for 3D/PointCloud reconstruction of multiple satellite imagery (angels) from different resources for a single patch (DGM)
* “Does the scene's subspace trajectory respond, at the right time, to a physically documented event — and stay quiet when nothing happens — and can we tell an abrupt disturbance from gradual drift/recovery?”
* Kanai 2023: clean-prefix + threshold-free AUC vs SSA-min-angle baselines + Grassmann-MDS separability; Fukui 2024: internal self-consistency + qualitative event alignment
If: “The "set of samples" is fake. DS, MSM, CMSM, KMSM, KGDS — the entire CVLab family, and every piece of reference code on your disk (Venus sculptures, hand-pose, motion3D, faces) — are image-set methods. A subspace is meaningful when you have a set of related observations: 300 views of a sculpture, many frames of a face, a motion sequence. Bi-temporal change detection gives you one image per date. To manufacture a subspace, the project treats the ~1.26M pixels of a single image as the "set." That subspace describes the global spectral distribution of the whole scene — it has no notion of where anything is. This is the category error, and it is the literal content of Sensei's warning: "your algorithm can make a subspace, but it can break the spatial information.”” Then how about using hyper-spectral satellite images that have hundreds of bands and treating those bands as the “300 faces of Venus”?
“A hyperspectral spectrum is a rich, near-continuous signal of 200–300+ bands, yet classical change detectors crush it to a single scalar — the spectral angle (its direction) or the change-vector magnitude (its amplitude). Both discard the spectrum's higher-order geometric structure. We ask: does that discarded structure carry change information, and can a Difference-Subspace representation — which encodes the full, illumination-invariant geometry of the spectral signal — detect and characterize change more faithfully than these scalar measures? And how does that advantage depend on spectral dimensionality?”
How It's Used (pixel spectrum)

Earth Observation: Satellites and drones use this technology to map forest health, detect crop diseases, or identify water pollution.
Mineralogy & Geology: Used to pinpoint the exact chemical composition of rock formations or discover trace minerals.
Medical Imaging: Helps distinguish between healthy and diseased tissues (like cancerous cells) based on their distinct spectral signatures
* "Lack of Temporal Subspace Methods for Multi-Date Satellite Image Analysis in Rapid Disaster Assessment... track damage progression... predict recovery trajectories.
