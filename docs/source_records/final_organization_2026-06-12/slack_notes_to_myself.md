- re-framing problem statement could be:
    Deep models can work well, but need labels and can be hard to interpret.
    Many changes are “pseudo-changes”: shadows, seasonality, registration differences, sensor effects.
    Generalization across cities/events/sensors is still difficult.
    Classical unsupervised methods are interpretable, but often weaker than learned models.
    Other than that, we can explore foundation models, we can try to train and fine-tune foundation models using subspace methods because large-scale AI practical. Because foundation models contain billions of parameters, updating every single weight during training is incredibly expensive. Subspace methods solve this by forcing the model to learn or optimize within a much smaller, low-dimensional mathematical.
    Really important point "Better method understanding: testing global pixel DS vs patch/window DS answers a real methodological question: what is the correct subspace construction for satellite images?" so we need to understand the correct subspace construction for satellite images.
    Maybe we could say "because labeling satellite data is expensive, we could use SM DS/KDS outputs as priors (data to use in NN training) to train better NN for change detection."

- https://www.mdpi.com/1424-8220/25/23/7265#
-  IMPORTANT: maybe we can use Difference subspace methods to isolate different channels’ properties
- more channels better change detected
    since SM are good at classification/clustering.....
- "Main message
Your research is about using Difference Subspace methods for Sentinel-2 pre/post change detection, and you are currently checking whether the way you build subspaces is mathematically appropriate for satellite images.
2-Minute Script
My research is about change detection in satellite images. More specifically, I am currently working with OSCD, which is a Sentinel-2 dataset with pre-event and post-event images, and the task is binary change detection: changed area or unchanged area.
I am studying whether Difference Subspace methods can produce useful and interpretable change representations from these pre/post multispectral images.
In the original TPAMI Venus setting, one sample is a whole image view, and many views form one image-set subspace. In my current satellite adaptation, one sample is different: one valid Sentinel-2 pixel becomes a 13-dimensional vector, because Sentinel-2 has 13 bands. For each date image, I collect all valid pixel vectors and build one PCA subspace for the pre-image and one for the post-image. Then DS compares those two subspaces.
The important issue I am investigating now is that this construction can lose spatial information. PCA sees the distribution of spectral values, but it does not know whether a pixel came from a road, sea, building, or airport. The position is only used later when I reshape the score back into an image map.
So my next research step is to compare this global pixel-based subspace with more spatially aware versions, such as local-window DS and patch-based DS. The goal is to clarify what kind of subspace construction is appropriate for detecting changed areas in multispectral satellite images.Emphasize
OSCD binary change detection, not damage classification yet.
DS is being tested as an interpretable change representation.

The key research problem now is how to construct the subspace correctly.
Your next step is spatially aware DS: patch/window methods.
Avoid Saying

“My method solves disaster damage segmentation.”
“DS improves segmentation.”
“This is already the final method.”
[3:37 PM]A possible application is abandoned greenhouse mapping in Tsukuba. I am helping with manual labels for a related Earth observation project, and it made me think that interpretable change detection could support this kind of task. For example, subspace-based change maps may help identify areas whose spectral or spatial patterns changed over time. But this is still a future application; my current experiment is focused on OSCD binary change detection first."
- "Thank you Sensei. Yes, I understand your point. My current research purpose is not yet disaster damage classification, but more specifically: detecting areas changed between pre-event and post-event Sentinel-2 images, using OSCD as a binary change-detection dataset.
The purpose of applying DS is to test whether a subspace-based change representation can produce an interpretable change map, and whether that map can be useful as a prior/input for supervised segmentation.
I agree that my current global pixel-based subspace can break spatial information because it treats pixels as unordered 13-band vectors. So I think I need to clarify the problem setting again and compare this global spectral DS with more spatially aware constructions, such as local-window DS or patch-based DS.
I will revise the research problem as: “Can DS-based representations help detect changed areas in pre/post multispectral satellite images, and what subspace construction is appropriate for preserving useful spatial information?” Then I will test which construction is suitable before continuing with larger experiments.v"
- "I should understand how the subspace method was implemented for my case, for my project. I need to properly implement subspace if it wasn’t properly implemnted (do we use the one subspace per channel approach in oppose to one subspace per whole image approach where each vector is the 13-D pixel, or are we using the one vector per whole channel and subspace per whole 13 channels where each channel is a vector approach, or are we implementing it differently?), we need to understand how we implemented subspaces in our project, it goes back to me relying on AI tools to do the whole job for me instead of reviewing and understanding how the techniques were implanted and wether they were implemented properly or not. Sensei and my senpais told me to understand what I’m doing.
What is the subspace here? What’s the dimension of the subspace (number of PCs not the space dimension) at hand? Are we losing information of each pixel vector?
13 channels, is each channel represented as a vector or subspace?
Experiment with the two methods from the two papers sensei mentioned:
First paper: S3CCA: Smoothly Structured Sparse CCA For Partial Pattern Matching. Link: https://staff.aist.go.jp/takumi.kobayashi/publication/2014/ICPR2014.pdf 
Spotting finger spelled words from sign language video by temporally regularized canonical component analysis. Link: https://www.researchgate.net/profile/Kazuhiro-Fukui/publication/303563203_Spotting_fingerspelled_words_from_sign_language_video_by_temporally_regularized_canonical_component_analysis/links/5a8a413faca272017e621f5d/Spotting-fingerspelled-words-from-sign-language-video-by-temporally-regularized-canonical-component-analysis.pdf 

Sensei said to look into Canonical Correlation Analysis (CCA).
Current implementation of Subspace —> 5 PCs in 13-D space: dimensions = (5,13)  (I forgot what this notes meant, could be that we should try to make the subspace dimensions to adapt to the (5, 13) dimension if they weren’t already like that?
How did the projection back to image from PCA/Subspace happen (math) in our project if it happened in the first place? Sensei asked that question.
Have you tried training U-Net with the raw data without Otsu thresholding or no thresholding at all?
Sensei asked me to look into K-PCA implementation, I don’t know why or how.
Sensei asked me to look into Kernel PCA, I don’t know why or how. Try the Kernel PCA trick.
 experiment with Itsu thresholding before U-Net training and after.
Investigate the lost information of position/of each pixel vector if that was the case in our project.
investigate the projection: how did we exactly extract the difference? How projection was done? How we extracted outputs?
Projection energy? What’s that? Calculated how?
Understand projection magnitude and how it was used here, how?
5 dimensional subspaces in 13-D constructed from 256x256 pixels being small or large?
What’s the difference between DS and GDS? Understand that well, see which one was actually implemented here and learn.
In method 1 each image is one vector, right?
How did I get the pixel? (I forgot what the question meant at the time)
What’s the number of (data pints and features) dimensions of input before we put it into PCA?
Train raw data on unit without any thresholding, but normalize first, you could try for example the normalization from 0 to 1, doesn’t have to be only that form of normalization could be others.
Understand how to project back the change onto the original image or the pixel space, like DS has different space so how do we project back?
In K-PCA we can map low dimensional vector into high demential space like rows 256x256 and columns 13, “(dimension of first space become bigger than the number of datapoints)?” Gaussian spca and stuff (circles around and triangles inside example)."
- "discuss the application of DS and geodesic decomposition to various types of temporal sleep data.
[5:22 PM]another thing in addition, recently sensei asked me: "have you already understood the second-order difference subspace and geodesic projection? I just wondered whether the geodesic projection (decomposition) would be effective for your research. Kazu". what do I reply to him? based on our project real implementation and understanding. (edited)"
- https://openaccess.thecvf.com/content/WACV2025/papers/Bandara_Deep_Metric_Learning_for_[…]rvised_Remote_Sensing_Change_Detection_WACV_2025_paper.pdf
- https://gaza.mapping.jp/
- To do: utilizing usage of bands better
- important: when we put the priors into the netweork to guide the model, some sort of smooth transition from priors change maps into propoerly trained map that shows proper chagne maps and how it got there like some animations
- a small question, when we say raw what do we mean by that, I just need to udnersand so that I can explain at my seminar, like when we say raw is that only pre and post images for the OSCD dataset put through a U-Net NN? if yes how? if no then what and how? now regarding sliding window, explain the whole ordeal about it from all aspects
- so explain whast the interpretaion of centering the input subspace and un-centring it, and explain the interpretation of I - P where we subtract the identity matrix from a project matrix, idk what that supposed to mean, remember, the best way to demonstrate a point is with an example, but before exapmple give the proper general interpretaion. quick question, is the residual part after we proejct a subspace onto another supposed to be something within the domain of the "projected onto" subspace?
- If you want “final” claims: rerun with multiple seeds +
    pick a threshold on val (don’t rely only on 0.5), then
    report mean±std.
- construct a DS based on the OSCD where we make the DS based on pre and post images and thier surrounding images as a whole, then use the OSCD benchmark as a way to train a "subspace model" NOT a weight and biases NN type model but rather that one subspace method that adjust the class subspaces based on new inforatiom or something, we can use a smilar idea with taht, where we try to make the class subspaces/DS subspaces adjusted and conformed to the becnhmark GT as much as possible and we do that for all the 24 city patches we have, and somehow we try to make a sense of that and use it as a basis for trainng a U-Net model with teh change data
- or what if we built a DS using the 13 bands together? as in the first band with all pixles is one subspace, second band with all its pixles are another DS, so on so forth all these configuratiosn are worth discovering
- so I have a question here, i didnt get the full understanding of how we used DS in our project, but
  if I get it correctly (and please correct me if im wrong), we make a subspace using each pixel in an
  image with its 13 channel as a 1-D vector, and make a subspace with al
- I can use OSCD properly!!!! how? for each pair of pre and post images for a city, I got to the date for each image, and collect a bunch of images before and after each date (like 5 images before the pre image and 5 after the pre image with a 1-day duration, and do the same for post), why? to be able to build an actual proper GDS!!!
- why are we not doing DS on the whole pre and post images so that DS recognizing the difference on a global scale
[1:20 AM]instead of calling them optional priors and considerign them optional, maybe we can make them a main attraction of our resaerch
[1:22 AM]get DS maps -> feed and train a NN with it to recognize change (and what type in the future)
- use Change data to calculate developmet growth for example, water body expansion, humidty incease
[1:02 AM]why are we using DS per pixle and now sliding window approach
- Regarding Phase 1 and Phase 2 results, now, if I'm understanding this correctly, and correct me if I'm wrong, we have a problem when it comes to the benchmark, the OSCD benchmark, the Onero Satellite Change Detection Benchmark, which is that it might not be sufficient enough as a benchmark to compare our methods against. If I'm understanding this correctly, the different subspace change detection methods and multiple other methods pick up on a lot of change that is not necessarily masked or annotated in the change detection dataset, the benchmark. Correct or not? If that's the case, what do you suggest as a solution, based on your research that you're going to do, where you're going to look up online and try to analyze this problem in order to have more concrete, more trusted, and more, how do I say it, just like a better way to measure change? Because I'm having this sort of problem that might be probably the most important problem that I'm focusing on at this moment in my research, which is trying to define what change is right now. Because in our context, what is change? Let's discuss that together, too.
- All right. Now comes an extremely important part, the presentation. So in addition to the research report that I was asked to submit, which is about three to five pages, I was also asked to make a presentation that is at least 20 minutes. And I basically show my research in the presentation. So I'm a bit confused on how to make it, because I need the presentation to be comprehensive. But I need it to be easy to understand. And most importantly, I need it to be explainable. And I need it to be easy to understand for others, because most people, to be honest, for example, my presentation, I'm going to be presenting to fellow master's students in my department and some of my seniors. But a lot of them are not familiar with concepts like difference subspace, for example, or PCA difference, or chain detection methods, because all of us have different interests in research and computer science. So the presentation must be explainable, easy to understand. And it needs to have a lot of visuals. Not extremely a lot of visuals, but a lot of visuals. Because it can't be clouded with text or flooded with text. It needs to be easy interpretable. And it needs to be well understandable. So we could go on through this in different ways. But I think I could ask you to do the following, which is you need to generate me based on our project, based on our research project. You need to be able to give me a complete narration, a complete script for the 20 minutes, and for all the slides. But the thing is, I'm not sure if I should let you make the slides or not. Because from my research, I've learned that there is this service called Google Gemini that is really good at making presentations. But it doesn't have the research context that you and me have. So I could try outsourcing making the presentation to Google Gemini. But for that, you need to tell me what sort of context do I give it. Do I give it a phase one and two report? Do I give it the complete report, the research article? Do I give it the research papers? Because one very important thing, in our presentation, where we mention other topics, like different subspace, or unit, or whatever, we're going to need to take the figures, copy the figures, either by hand or automatically, from those papers and put them in the presentation. So you need to tell me, for example, OK, go to that research paper specifically and go to that figure, copy it and put it here, and then so on and so forth. Now, yes, as I said before, the complete script is very important. So you could try, OK, listen up. How about this? You give me the complete prompt that is concerned about making the presentation. In this prompt, you explain the presentation and how to generate it. You give the complete script. You give me the complete instructions. This is very important, instructions on the visuals, what sort of visuals we need on each slide, if needed, of course, what sort of visuals are needed, where to put them, and most importantly, where to get the visuals. Because, for example, we have some visuals in the outputs in phase one or phase two in our project that live locally. We have some visuals that live in the research papers that we're going to reference. We have some visuals that are diagrams that need to be made, diagrams, flowcharts, very important visuals that need to be made. So explain all these instructions. And regarding the visuals, there are many types of visuals. But the most important type of visuals that we're going to have to be able to get and generate are the following.
- DS Projection + PCA-Diff + L2 diff = best combination?
- SOME NOTES: senpais suggestion:
Try vectotizing multiple images where each image is one vector and you vectorize them and you use ResNet60 for example, you could also utilize ResNet50 on image net, you can try aggregating multiple images in one subspace instead of one per subspace, like for example you could try making. subspace for the first 15 images (15 days) of the month for one place then the other 15 for end of month, or 6 images for 6 months of the year and 6 other images for the others 6 months of the year for example to see if there's some sort of change.[9:43 PM]some keywords to think about and note: Geodesic change detection methods. Elevation, building, height, change detection map, damage map, before, after, heat map, red, blue. Reconstruction of building based on PCA change detection method. Geodesic change detection method. Reconstruction of building based on method. Geodesic change detection method.
[9:44 PM]some notes taken with the help of gpt5 online that needs to be taken into considaration and noted and distilled:
By building-level descriptors, we basically mean:
Feature vectors that summarize one whole building as a single object, instead of looking at individual pixels.
So: one building → one descriptor (one feature vector).
In satellite / aerial imagery you can look at data at different levels:

Pixel-level: features per pixel (spectral bands, NDVI, etc.)
Patch-level: features per small image patch (e.g., 32×32 window)
Building-level: features for the entire building footprint or mask
Building-level is object-centric: you treat each building as a single instance.
This is useful for damage assessment, land-use / land-cover at object level, etc.

They embody characteristics used to define and evaluate a building's performance, features, or operational status for specific purposes such as sustainability assessments, performance evaluations, or architectural modeling.

What can building-level descriptor contain?

(a) Geometric / shape features
From the building footprint or mask:

Area, perimeter
Compactness, elongation, rectangularity
Height estimates (from DSM/DEM if available)
Aspect ratio, orientation
(b) Spectral / radiometric features
Aggregate the spectral information over all pixels belonging to the building:

Mean / median / std of each band (R, G, B, NIR, etc.)
Indices like NDVI, NDBI averaged over the building
Histograms of band values
(c) Texture features
Texture computed within the building mask:

GLCM features (contrast, homogeneity, entropy, etc.)
Local binary pattern (LBP) histograms
Simple measures: variance, edge density, gradient stats
(d) Context / neighborhood features
Where the building sits and what’s around it:

Distance to road / coastline / river
Surrounding land-use type within some radius
Density of nearby buildings (crowded vs isolated)
Average height / area of neighbors
(e) Temporal / change features
For pre/post disaster (or multi-temporal) imagery:

Change in mean spectral values (pre vs post)
Change in NDVI, NDBI, etc.
Change in texture metrics
Difference of features: f post - f pre
f_\text{post} - f_\text{pre}
fpost −fpre 
Or concatenated [pre_features, post_features] in one vector


How do we get them in practice?
Pipeline idea:

Detect / segment buildings
From a building footprint dataset (OpenStreetMap, city GIS), or
From a building segmentation model (like a U-Net).

For each building mask:
Collect all pixels belonging to that building.
Compute geometry features from the polygon/mask.
Compute spectral stats, texture metrics, temporal differences.
Concatenate into a single feature vector.



How can this connect to our research?
In your damage & land-use project:

Input level
You can work at pixel-level (segmentation masks), or
You can aggregate up to building-level descriptors.

Damage classification
For each building, you could:
Use a classifier (random forest, SVM, small MLP, even SSC clusters) on its descriptor to predict:
damaged / undamaged, or
damage grades (no/slight/moderate/severe).


Temporal analysis
You can track how each building’s descriptor evolves over time:
Pre-disaster vs different post-disaster dates → to analyze recovery.
Clustering trajectories of building-level descriptors to find patterns:
quickly repaired vs slowly repaired vs never repaired buildings.


Subspace / PCA angle
You can place building-level descriptors into:
A PCA subspace to visualize them (PC1–PC2 plots for damage classes).
A difference subspace / SSC framework to cluster buildings based on temporal change behavior instead of just pixel changes.







PCA image reconstruction (the medium)

For multispectral satellite images, PCA reconstruction can have two purposed: post DS/SSC processing visualization, or Compression & de-noising, or other purposes.

Key word: Spectral–temporal vectors

-potential uses for PCA reconstruction-:

“PCA features” = the low-dim vector z
“PCA feature reconstruction” = take z and map it back to get f^\hat{f} f^ in the original feature coordinates.
The error tells you how well this feature fits the “typical” subspace spanned by the PCs.
You can do this for:

Pixel spectral features
Building-level descriptors
Intermediate features in your damage-segmentation network.




We can do Change detection via reconstruction error (baseline)
Very straightforward idea you could use (or cite as baseline):

Train PCA on pre-disaster pixel vectors (or pre+post, but labeled as “normal”).
For each pixel / patch in the post-disaster image:
Computer projection z and reconstruction x hat
Use reconstruction error as change score

Threshold or cluster the error map to get a binary change map.
Intuition:
If a post-disaster pixel is “compatible” with the pre-disaster subspace, it reconstructs well ⇒ low error ⇒ likely unchanged.
If it is very different (e.g. building destroyed, flooded area), it won’t fit the old subspace ⇒ high error ⇒ changed.

We can use it for Dimensionality reduction before SSC or clustering
PCA on U-Net / encoder features
Later, when you have a U-Net (or similar) trained for damage segmentation:

Extract deep feature vectors from the encoder for many pixels/buildings.
Run PCA on these deep features.
You can then:

Visualize them in 2D/3D (PC1, PC2, PC3) to see how damage levels or land-use classes distribute.
Use reconstruction error in the feature space as a sort of anomaly/damage indicator:
Train PCA on “normal” (undamaged) features.
For new images, high feature-space recon error could mark unusual, possibly damaged structures.

This mixes your “subspace mindset” with deep learning in a nice, explainable way.
Denoising / cloud & shadow mitigation
Multi-temporal satellite images often have:

Noise, sensor artifacts.
Clouds, shadows.
PCA reconstruction with only the leading components can act as a linear denoiser:

It tends to keep stable, consistent structures (buildings, roads, land-use patterns).
It can suppress random noise and sometimes small, inconsistent artifacts.
You could, for example:

Pre-process your images with PCA reconstruction before feeding them to U-Net or your change detector.
Or compare “original – PCA reconstruction” to highlight “stuff that doesn’t fit the main data subspace” (potentially useful for anomaly detection).


TLDR:
PCA reconstruction = project onto a low-dimensional subspace and project back; the difference measures “how typical” a sample is.
PCA image reconstruction = same thing but each data vector is an image; useful for compression, denoising, visualization.
PCA feature reconstruction = same idea on feature vectors (e.g., deep features, spectral–temporal features); reconstruction error can be an anomaly/change score.
For your research, PCA is:

A canonical way to build subspaces (pre/post) for simple change detection.
A dimensionality-reduction step before SSC or other methods.
A baseline / interpretability tool to relate your fancy subspace + U-Net methods to something classical and easy to explain.
- Yes, so that we can deduplicate the repeated links or references. Nonetheless, as I was saying, we need to properly reference where we got our information from. And, yes, that's very important. Moreover... So, I know this prompt is long, and it's all over the place, so I'm going to have to ask you, if you want, if you think that it's better to process the tasks one by one, then we can do that. You can process the first task for this prompt, and then we can process the second task in the second prompt, if you think that's going to yield a better output from you. And keep that into mind. And for the next prompt, if you want, you can give me the next task in the next prompt, so that I can put it there, and you can give me the best results. I don't want to overwhelm you with multiple requests at once, because I want you to give me the best output. So, I'm really stressed out, but I really need to transform my project, my research project, from doing research notes and planning, to doing implementation and experiments. Because so far, all of what I've been doing is taking notes and researching and reading, which takes a long time, and making plans, but there hasn't been implementation done so far. So, please, please, please, make absolutely sure that you think really hard, and you choose the best possible way you can go around this. I want a very specific, very detailed, very literate, very practical, and very well path approach. A specific path that we can go from this, from the beginning of the pipeline, from the objective statement, all the way until the conclusion. No, actually, from the beginning of the pipeline, from abstract, all the way until the end, which is the conclusion. And there's a lot of steps in there. Because we will need to get a spec sheet somehow, like we need a spec sheet for the model. I'm going to need to generate a spec sheet based on the plan that you will give me. Don't give me the spec sheet yet. This will come later on when we decide absolutely on our plan, on what to do. After we specifically pick the proper plan and we plan out everything, the implementation, to the point when we plan out the implementation, we go through with the spec sheet and then with the proper coding, like experiment implementation. So, for that case, let's work properly. Tell me if you need information somehow to give me the best possible output. And for the research notes repository, if you're suggesting to add any edits, then tell me about it. Because I still want to keep the research notes repository as the place where I keep my notes in. So, when we decide on a proper path, on a proper detailed plan, like a proper detailed implementation plan that is well verified and checked. Also, I remembered a very important point. Now I remember it. Regarding the links and the references, I told you I have a lot of papers. I'm going to need you to organize the links by importance. Also, I'm going to have to need the papers that are needed for the implementation, because we need to understand the proper implementation of the methods for our project. Like, for example, the first order difference subspace and the second order difference subspace. The mathematical formulas are well defined in the papers, so we might need them. Like, for example, maybe I can pass them along the spec sheet to the AI model, like Codex, for example, Codex 5, that will implement the project and experimentation. For that case, I might need to, for example, download all of the relevant papers and put them on the implementation project as reference points. So, in addition to that, also, when it comes to things like datasets, for example, they could be huge, like hundreds of gigabytes. For that purpose, tell me, for example, if you need me to download the datasets from some place and put them somewhere in the file structure, then inform me about that, so that I can do it manually, because I think it's too much for you. Now, take in all of what I've said, process it from the beginning to the end, and discuss things with me. Organize my thoughts, too, because I think I'm a bit scattered all over the place.
- What I'm about to give you is a Git repository that contains all of my research notes, all of my research thoughts, all of my research main points that any typical research paper or presentation or thesis would have. The notes were taken along a very long span of time, from multiple resources, all relating to the same idea, more or less, which is damage or disaster resilience. Lots of ideas, lots of ways we can go around this. It's not just one thing, for example, damage segmentation, it's like damage change detection, damage levels, damage classification, so on and so forth. So, I'm about to give you the repository. I'm going to ingest it with a tool and give you the whole repo as text here in the chat. And I'm going to ask you for a very important, extremely important tasks to do. Not just one task, but multiple tasks. First of all, I'm going to need you to understand what are the paths that we can go from here. What I mean by that is, as you already noticed from the research notes, there are multiple paths that we can go about this. Like change detection, which is the main idea of damage progression, for example. There's predicting damage based on a year-long satellite imagery, for example. There's a lot of methods, SSC, sub-sparse subspace clustering. There's first and second order different subspace. There's UNIT, there's MCBA, there's geodesic difference, there's Grassmanian manifolds, there are manifolds. There are a lot of relevant, extremely relevant methodologies. But the main problem with you right now is, even though I showed in the master files that I want the research to go in this way. You can see the master skeleton, the master outline, that the research is written in a certain way. But it does not necessarily have to be done that way. You can look at all of the variables that we have here, examine all of them, and think of the best optimal research path that we can do and practically pursue at the moment. After you think really hard, after you take in the whole repository, you take in all the information available within your grasp. And you verify it, you check it, you check the applicability, the dataset, the method, the results, or the expected results, the objective. This is very important, by the way. It's very, very, very important to define the objective, the problem, and the solution, or the proposed solution. After we've taken all of that in hand, I'm going to have to ask you to help me with formalizing my thoughts and choosing one path to go on to for now, just for now. I think that this repository has a lot of potential to go on multiple path routes, all related to more or less the same idea, which is damage or disasters and satellite images. They're all related, but I think there's a big chance that we can go on through multiple paths, multiple papers, multiple researchers, not just one, which is really good. But I need to start with one at least, especially because I don't have much time, so I'm going to need to choose one path to focus on for now. And after we choose the one path, all the way from the beginning to the start, from the beginning, from the objective statement, from the problem, the solution, the methodology, the expected results, and the specification of the methodology. Very important, the data set, or data sets. After we do all of that, because I'm going to have to, first of all, do experiments and get some sort of results, be it figures or be it results or be it metrics that I can compare in a benchmark with other methods. I'm going to have to compare my method to other methods. I'm going to also have to do a presentation where I present my research, my progress, and how can I have progress if I don't have experimentation yet. So, this is very long, I understand, but I'm going to need you to think really hard and verify locally and online, everywhere, about the optimal research path that we can go on from now. In addition to that, what I'm going to give you in the Git repository, it's all of my research thoughts, all of my research notes and thoughts on brainstorming, and also my sensei's notes, my senpai's notes, which are very important too. In addition to the repository, I'm going to give you a long list of references. They're really long, they're almost about 300 references. Not just references, by the way, they're links, to be specific. About 300 links that I saved on my Google Chrome, on my bookmarks, as bookmarks. There's a lot, a lot, a lot, a lot of links. Most of them are research papers, of course, some of them are websites, or projects, or codes. They're all relevant. I'm going to have to ask you to take in all these links that I'm going to give you, along with the Git repository, and do with them multiple things. First of all, I need you to order them from the utmost importance for our research, to the least importance, and subcategorize them in other subfolders inside the research folder. Yes, because I'm going to export the bookmarks from Google Chrome, and give you the code for the bookmarks file, which I forgot what it was, I think it was HTML or XML, I'm not sure. But I'm going to give you it, and you're going to give me as an output the same file format, which I'm going to also import back into Chrome, but this time it's going to be categorized in subfolders. Everything will be in subfolders, also cataloged by importance to our research, ordered by importance, and subject to. You know what, I'm going to let you decide the best way to sort them out, but I need something that indicates importance, so that I can read the papers in depth that matter the most to me. And this is very important, listen up. I'm going to need to connect the papers that matter the most to our methodology that we are going to choose, and choose to focus on. The path in the research that we're going to focus on, that we're going to discuss together and focus on it, from our different paths, but we're going to choose one path, and we're going to focus on it. And for that path, I'm going to need my references ordered, and we're going to have to connect the links, we're going to have to connect the references to the related points in the research repository. So, it's really hard to explain, I'm sorry that I gave you, I'm trying to explain this, but it's explained in such a bad way, but the point is yes, we need to organize the links, most of them are important, some of them could be non-important or by accident, or irrelevant. If that's the case, don't delete them, or remove them, you can organize them in a folder called irrelevant, or just say they're, for example, duplicated. Oh, also, any duplication, tell me about it so that I can remove it.
- Thank you. One important thing that I want you to keep in mind and note down, really important, I think that having the project as damage detection and land use, it feels a bit widespread and not practical for me. So I think from now on, we might need to consider orientating the research towards damage specifically. Stuff like doing first order difference subspace and second order difference subspace to produce something like feature difference or different feature vectors or maps, and then using such idea or such output, then putting it in something like SSC or sparse subspace clustering. Then trying somehow to understand the progression of damage or trying to build specific damage maps, which is by the way, the idea of building damage maps was really highlighted by my senpai. So I think it's a really good direction for us to consider. Generally speaking, what I'm trying to say is we need to focus on one task at a time. If we are able to implement the damage part properly, maybe in the future, we can incorporate land use with the research. But you don't have to do exactly what I'm saying right now. Again, think, discuss with me, act as a full on professor who has immense knowledge in this, be practical, understand the practicality, the pipeline, the math, the possibility, and the scope, most importantly. All right, I just wanted to give you a heads up. And then right now, we're going to continue distilling the next chats. We will go back to discussing the research overall, or discussing the research from the bigger picture after distilling the research. I have a lot of notes that I want to tell you about that could uh seriously elaborate and upgrade our research and we can finally get into implementation. Because so far, what we were doing is planning, which I understand is really important. But I don't have the privilege to plan that much. But at the same time, yes, it's important. Let's continue planning. In the future, I will go on the implementation when I feel like the planning is sufficient enough. Again, I'm not telling you to delete anything. We need to be able to contain all of our ideas. Because who knows, if something doesn't work out for now, maybe it could work out in the future. If one idea is not applicable, maybe we can try applying a different idea that we noted down and then incorporating a different idea we also noted down. The point is to note down everything we can, but at the same time, we need to understand what is our research approach. We need to specifically pick one approach, but at the same time, understand that we have a plethora of approaches, a plethora of paths, which means the very good opportunity of having multiple research papers, multiple researches, not just one, which is really good for us.
some old notes as justifications:
- 1. Efficiency in Handling High-Dimensional Data

Challenge: Satellite images, especially multispectral or hyperspectral ones, often have high-dimensional data. Directly feeding this into CNNs can be computationally expensive and require significant resources.
Subspace Solution: Classical subspace methods like PCA, NMF, or LDA reduce dimensionality, retaining only the most critical features while removing redundant data.
Benefit: This preprocessing reduces CNN complexity, speeds up training, and minimizes the risk of overfitting, particularly when datasets are limited.


2. Improved Interpretability

Challenge: Deep learning models like CNNs often operate as black boxes, making it difficult to interpret intermediate results or understand the "why" behind their predictions.
Subspace Solution: Subspace methods are inherently interpretable, offering clear insights into feature relationships and contributing to transparent decision-making.
Benefit: Combining the two approaches ensures the system is both performant (thanks to CNNs) and interpretable (via subspaces), fostering trust in its outputs, especially in critical applications like urban reconstruction or disaster planning.


3. Complementary Strengths

CNN Strengths: Excels at extracting hierarchical spatial features and identifying complex patterns in image data.
Subspace Strengths: Efficiently models relationships between features (e.g., spectral correlations) and handles noise robustly.
Benefit: A hybrid model leverages these complementary strengths, providing robust feature extraction and reliable pattern detection even in noisy or incomplete datasets.


4. Addressing Imbalanced and Sparse Datasets

Challenge: In satellite imagery, data is often imbalanced (e.g., many examples of urban areas but few of damaged structures) or sparse (limited labeled data for specific regions).
Subspace Solution: Subspace clustering or sampling can generate balanced subsets or identify critical regions for labeling.
Benefit: CNNs trained on preprocessed, balanced data perform better, reducing bias and improving generalization.


5. Resource-Efficiency for Lightweight Applications

Challenge: CNNs are resource-intensive and may not be feasible for deployment on edge devices or in resource-constrained environments.
Subspace Solution: Classical ML methods are lightweight and computationally efficient, making them suitable for preprocessing or feature filtering.
Benefit: A hybrid system can preprocess with subspaces on edge devices and offload deep learning tasks to more powerful systems, enabling flexible deployment.


6. Enhanced Generalization Across Tasks

Challenge: Single-method approaches may overfit or lack adaptability when applied to diverse tasks like land-use classification and damage detection.
Hybrid Benefit: Combining methods allows for task-specific optimization:
Subspaces identify general feature patterns (e.g., urban vs. rural areas).
CNNs fine-tune these patterns for nuanced distinctions (e.g., types of damage or specific land-use categories).



7. Alignment with Current Research Trends

Emerging Trend: Researchers are increasingly exploring hybrid models, as evidenced by studies integrating feature extraction methods like PCA with deep learning for satellite imagery.
Relevance: This hybrid approach aligns with cutting-edge research while addressing gaps in scalability, resource efficiency, and interpretability.
- https://earthi.space/blog/future-urban-planning/
- https://www.euspaceimaging.com/blog/2021/02/18/from-urban-to-rural-enabling-sustainable-urban-planning-and-development-using-satellite-imagery/
- https://spaceknow.com/blog/satellites-ai-urbanization-a-modern-approach-to-urban-planning-and-monitoring/
- reserech perspectives:
1. Urban Infrastructure Planning Using AI

Challenge: Urban planners often lack real-time, high-resolution data to effectively design infrastructure that balances urban sprawl, resource distribution, and environmental impact. Traditional methods like ground surveys and static maps are time-consuming and outdated.
Research Opportunity: Develop a machine learning model to analyze satellite images for key urban features (e.g., roads, buildings, vegetation). The model could predict optimal placements for new infrastructure like hospitals, schools, and evacuation centers, accounting for terrain, climate, and population densityspaceknoweuspaceimaging.
2. Dynamic Urban Growth Monitoring

Challenge: Rapid urbanization, especially in developing regions, leads to issues like overcrowding, poor sanitation, and inefficient resource use. Monitoring and predicting these changes using traditional tools is limited in scope and accuracy.
Research Opportunity: Create generative models that simulate future urban growth from satellite imagery. These models could help city planners identify potential problems before they arise, such as areas prone to congestion or insufficient green spacesearthieuspaceimaging.
3. AI-Enhanced Rural-Urban Transition Analysis

Challenge: Many rural areas transitioning to urban centers face uncoordinated development. Determining optimal land use for such regions is critical for sustainable growth.
Research Opportunity: Use satellite imagery and computer vision techniques to map and classify land use dynamically, identifying areas suitable for residential, commercial, and industrial purposeseuspaceimaging.
4. Subspace Methods for Multi-Spectral Satellite Data

Challenge: High-resolution multi-spectral and radar data can capture detailed environmental and structural features but require sophisticated processing to extract actionable insights.
Research Opportunity: Leverage subspace techniques to integrate spectral data and generate thematic maps. For example, mapping flood-prone zones for evacuation centers could be a practical applicationearthi.
5. Generative City Perspectives

Challenge: Urban designers often struggle to visualize the impact of new infrastructure on existing cityscapes from different viewpoints.
Research Opportunity: Build a generative AI model that transforms 2D satellite imagery into realistic city perspectives. This could assist in visualizing proposed developments and their integration into the existing urban fabricspaceknow.
- for presentation maybe: U-Net is a type of deep learning architecture primarily used for image segmentation tasks. It was originally designed for biomedical image segmentation, but it has since been applied to other fields, including satellite image analysis.
- https://www.mdpi.com/sensors/sensors-25-05965/article_deploy/html/images/sensors-25-05965-g007.png
- https://www.mdpi.com/1424-8220/25/19/5965
- 11. OpenCV and Generative Models for Data Augmentation

Relevance: Mentioned for generating synthetic datasets and improving model generalization, particularly when training data is limited.
12. Tokyo Metro Line Optimization Research

Relevance: Inspired your idea of using subspaces and AI for urban infrastructure planning. Parallels drawn with your proposed integration of SSC and U-Net for land-use and damage classification.
Abood/アブド  [8:15 PM]
Challenges such as spectral mixing, sensor resolution, and image registration timing were identified as contributors to classification errors
- per-pixel spectral difference vector
- grassmannian manifold
- Siamese fully convolutional network
- damage progression via subspaces
- geodesic: Instead of comparing raw pixel values or Euclidean distances, the method measures differences along the intrinsic geometry of the data manifold.
- learn PCA, LDA, SSC, LRR, Kernel PCA, and Subspace Alignment.
- Lack of Temporal Subspace Methods for Multi-Date Satellite Image Analysis in Rapid Disaster Assessment
[2:55 PM]There is limited research on integrating temporal dynamics into subspace learning frameworks to identify and track damage progression from multi-temporal satellite images (pre- and post-disaster).
[2:55 PM]In other words, current subspace models don’t:

Capture changes over time efficiently
Leverage temporal correlations for faster and more accurate damage detection
Adapt subspace structures as disasters unfold (e.g., earthquake → aftershock damage over days)


:test_tube: Why This Matters:

Disaster resilience planning depends heavily on identifying when and where damage escalates.
Relief efforts could be more effective if subspace models predicted recovery trajectories or secondary risks using evolving satellite data.


:blue_book: Suggested Research Direction:
Develop a Temporal Subspace Clustering model that:
Uses sequences of satellite images
Learns evolving low-dimensional representations (e.g., Dynamic SSC or Online Subspace Learning)
Classifies and predicts damage severity changes over time
[2:56 PM]some ideas:
Hybrid models combining U-Nets for spatial segmentation and subspace tracking for temporal damage evolution
Real-time deployment of subspace-based models on edge devices for early warning systems
Integrating subspace models with graph-based representations of infrastructure networks
- SAM (Segment Anything Model)
- Geometric machine learning https://www.instagram.com/reel/DPYnkeZjCO5/?igsh=MW8xb2E2bnZidmhvdQ==
- stuff to research: grassmannian pca: http://arxiv.org/abs/2407.07421
- high spatial resolution satellite imagery https://rslab.gr/hiresimagery.html
- https://dl.acm.org/doi/pdf/10.1145/3477145.3477162
- do this https://www.sciencedirect.com/science/article/pii/S0031320324007349 but with SM 
- Anti-Aliasing baed SM for RS imagery (or whatnot), DS for Anti-Aliasing
- I forgot to add one very important detail, which is the online ChatGPT, I did discuss with it the local ideas, right? So I sort of want to give it away to recognize the ideas that were already distilled in our local repo And for that, I can use a service like git-ingest, which is like a service that ingests git repository to a single text file, like a long text block I can use it to ingest the repository that we worked with because it's all text files or markdown files and then give it along the prompt so that the online ChatGPT recognizes what is different what is new, and what we already have worked with What I'm trying to say is a lot of the ideas that we've discussed and distilled locally are present online and I don't want to use those ideas again It would be useless, so I'm only focusing on the stuff that we don't know Or, if there are some updates for the old ideas that we've already distilled in the local repository that are online, right? If there are new updates or methods or approaches that we discussed in our online chats then maybe we can add them But yeah, just adjust the prompt so that it fits this new criteria
- I have some thoughts about your prompt that I need you to look into again and try to fix. Here's the thing. The format that we've been working with so far on this local repository is not finalized. The project title, or the project abstract, if you will, where we use SSC, or Sparse Subspace Clustering, and UNIT, and UAV, and so forth, it's not finalized. Yes, all of these elements are within our scope of research. But remember the original idea when I first started the chat with you of we don't want to limit ourselves to a certain core idea, or a certain approach, or a certain path. Because the thing about AI models, and the thing about you, ChatGPT, is if I give you the smallest hint of context, you might have a bias that you will use to give your next answers. And I don't want that bias. So, you said that you want to use the tag system of prob, obj, data, pre, myth, SSC, and the tag system that we've been using locally. We don't really have to use it. The most important thing right now is for us to collect all the ideas from the ChatGPT online. When I collect all of these ideas in the format that you specify me to, I can give you all of the data, to you locally, so that you can categorize them into their respective files, and category systems, and sort and whatnot. So you don't have to limit yourself within a certain structure. Be flexible. And you mentioned temporal change, and segmentation, and UNIT, and MCTA. That's all within our scope of research, but since AI is easily influenced by certain words, I don't want you to be biased. I want you to be very flexible. Atomicity is good. De-duplication is much needed. We don't want to have repeated ideas. Other things for the structure of the title, short, summary, final section, provenance. Again, I am not limiting you to a certain structure. Be as flexible as you can. Why do you even have a title in the first place? On-device SSC preprocessing. We didn't decide a title yet, because the approaches are finalized. The path is not finalized. Imagine we have multiple paths, right? So don't do that. Remove that. Novelty is prioritized. It's good. We need novelty. And gaps to watch is very important too. So yeah, don't get limited to the structure that we used to work with locally. Because here's the thing. Online, when I chat with Shadib2 online, we discuss thoughts that traverse the local thoughts. We've been talking about Grassmanian manifolds, Fourier series, Lie groups. Grassmanian methods, Grassmanian subspaces. We talk about different concepts that were not discussed locally. And if I specify in the prompt SSC or unit, then Shadib2 online might give the wrong idea about us being fixed toward a certain approach. So there are other ideas, other discussion ideas, other methods, other approaches that I discussed with Shadib2 online that I want to collect that you don't know about. When I collect all of these ideas, I can give them to you and you can structure them correctly. That's the intended result of the prompt that I'm hoping to get.
- Thank you for all your efforts. Now, we have finished distilling our drafts locally. That's a really great step. The thing is, I sort of have two places where I take notes, or not notes, where I have my thoughts that I discuss with AI models, specifically ChatGPT. So, for the local research notes, we just finished distilling all of them, which is good. Now, I sort of want to, not distill, but collect, first of all, collect the different, the plethora of research-related ideas and concepts, problem statements, objectives, solutions, abstracts, the gaps, and the novelty, especially the novelty. And I have multiple chats on ChatGPT on the website, in the web version, where I created a project. And under this project, I have multiple chats. I think it's around 15 chats or more. In this, about 15 chats or more, I chat with ChatGPT a lot, and I discuss concepts, thoughts, and different ways we can do the research. And I previously even made a project in our research that deploys first-order differential space, which I'm still validating. What I wanted to do is, I want you to create a prompt that I can use on ChatGPT online version, so that it understands what I want it to do. Similar to you, I don't want to specifically distill this time, but I want to sort of collect the ideas, because I have 15 chats, and some of them are really long, where I don't have the time to read all of them. So I want to do something similar to what I did here locally, and collect all the different contexts, all the different ideas, problems, solutions, gaps, novelties, etc. So that it's all in one place. And after I give the prompt from you, I give it to ChatGPT online, and it generates an answer for me, or a file. It just gives me an answer. Right? And then I get this answer, and I integrate it with the chats here on this repo, in this repository locally, so that we can finally integrate all of our thoughts that are scattered locally and online. Do you understand what I'm trying to say? Create a prompt that best suits my need here. I need to be able to have my thoughts all in one place, so that I can use these thoughts with ChatGPT to help me build the project. Like, right now, the thing that I need the most is to actually actualize and build the project, so that I can have results, right? And methods that I can compare to, like other methods I can compare to, and make a presentation by the end of next month. In order for the presentations, I need to have proper structure. I forgot the proper presentation or research structure, but I think it goes along the lines of problem statements, solution, method, solution, objective, future work, etc. I hope you understand all of this. Perform the tasks that I assigned you to do properly. Use your maximum thinking power. Be as smart as you can, because this is a very critical task. And give me suggestions, if you want to, that could help me better understand what I want to do.
- answet this question: "Describe the project you intend to investigate with Planet data. What questions do you hope to answer? Please include, when possible, some background on your research problem and a brief description of the methods you intend to use. Project description must be a minimum of 100 words.*" based on this doc
[2:46 PM]I intend to investigate the temporal dynamics of disaster-induced damage and land-use changes using high-resolution satellite imagery from Planet. My research focuses on developing a computationally efficient framework for post-disaster assessment by combining lightweight subspace methods and deep learning. Specifically, I aim to answer the following questions: How does structural damage evolve over time after a disaster? How can combining damage classification with land-use analysis support recovery prioritization and infrastructure planning?
This project is motivated by the increasing frequency of natural and man-made disasters globally, which has made rapid, accurate damage assessment essential for effective recovery. Traditional deep learning approaches for such tasks are computationally intensive, limiting their feasibility in resource-constrained environments such as disaster zones or developing regions.
Using Planet data, I plan to analyze pre- and post-disaster imagery at multiple time points. First, I will apply Sparse Subspace Clustering (SSC) for dimensionality reduction while preserving key structural features. Then, I will use U-Net to perform pixel-wise classification of land-use categories and damage levels. By incorporating temporal subspace analysis (e.g., first and second difference subspaces), I aim to track damage progression over time and integrate the results into a Multi-Criteria Decision Analysis (MCDA) framework to support recovery strategies.
The outcome will be a hybrid, lightweight, and scalable model deployable in real-world scenarios, offering actionable insights for disaster resilience, particularly in contexts like Japan’s earthquake-prone regions or post-conflict recovery zones. Planet’s high-frequency, high-resolution imagery is critical for capturing these rapid changes and validating the temporal evolution of damage in affected areas.[2:50 PM]Describe the geography you plan to investigate.*
[2:51 PM]I plan to investigate disaster-prone urban regions in Japan, such as coastal areas affected by earthquakes and tsunamis. These locations offer rich temporal satellite data and present real-world challenges in damage assessment and urban resilience, making them ideal for testing and applying the proposed framework.

Is this helpful so far?
- 