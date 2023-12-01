<div align="center">

  # My Project Portfolio

</div>

Welcome to my project repository with a collection of my recent work on artificial intelligence, data science, machine learning, deep learning, computer vision, medical imaging, etc.
Links to my relevant academic products (paper, code, poster, slides, etc.) are provided for each project.

## Projects

### Evidential Uncertainty Quantification for Active Domain Adaptation
[paper](https://arxiv.org/abs/2311.11367)
| [code](https://github.com/KerryDRX/EvidentialADA)
| [abstract](https://github.com/KerryDRX/EvidentialADA/blob/main/docs/abstract.pdf)
| [poster](https://github.com/KerryDRX/EvidentialADA/blob/main/docs/poster.pdf)
- An active domain adaptation framework based on evidential deep learning (EDL) implemented with
    - two sampling strategies: uncertainty sampling and certainty sampling
    - two uncertainty quantification methods: entropy-based and variance-based
    - three EDL loss functions: negative log-likelihood, cross-entropy, and sum-of-squares

### Continual Learning with Prompt-Based Exemplar Super-Compression and Regeneration
[paper](https://arxiv.org/abs/2311.18266)
| [code](https://github.com/KerryDRX/ESCORT)
- Class-incremental learning (CIL) with _**E**xemplar **S**uper-**CO**mpression and **R**egeneration based on promp**T**s_ (ESCORT), a diffusion-based approach that boosts CIL performance by storing exemplars with increased quantity and enhanced diversity under limited memory budget. ESCORT works by
    - extracting visual and textual prompts from selected images and saving prompts instead of images
    - regenerating exemplars from prompts with ControlNet for CIL model training in subsequent phases

### Evidential Deep Learning Loss Functions

[code](EvidentialLossFunctions)
| [derivation (focal)](EvidentialLossFunctions/docs/evidential_focal.pdf)

- Implementation of evidential deep learning (EDL) loss functions based on Bayes risk of traditional deep learning loss functions. Tasks include classification, 2D segmentation, 3D segmentation, and regression. A unified loss is proposed for all image segmentation losses in EDL.

### Evidential Uncertainty Quantification of Deformation Fields

[paper](DeformationFieldUncertainty/paper.pdf)

- Application of Deep Evidential Regression on a joint MRI and Cone-Beam CT image Synthesis and Registration model, providing aleatoric and epistemic uncertainty maps of the deformation fields and improving the registration performance.

### Statistical Shape Modeling

[code](StatisticalShapeModeling)

- Statistical shape modeling of brain ventricles, including
    - computation of brain ventricle surface correspondences by point distribution model of ShapeWorks
    - DeepSSM to predict shape PCA scores from brain MRI
    - aleatoric and epistemic uncertainty estimation based on Deep Evidential Regression

### Unsupervised Anomaly Detection of Medical Images

[code](AnomalyDetection)
| [slides](AnomalyDetection/docs/slides.pdf)

- Unsupervised anomaly detection of head (CT) and fundus (ultra-widefield) images, with
    - implementation of convolutional autoencoder, scale-space autoencoder, DAGMM, f-AnoGAN, etc.
    - anomaly segmentation based on image reconstruction error
    - visualization of anomaly region

### Cervical Spine Fracture Detection and Visualization

[code](CervicalSpineFractureDetection)
| [slides](CervicalSpineFractureDetection/docs/slides.pdf)
| [demo](CervicalSpineFractureDetection)
| [report](CervicalSpineFractureDetection/docs/report.pdf)
- Fracture detection in human cervical spine CT images based on deep convolutional neural networks, involving
    - detection of C1-C7 cervical spine on each 2D slice
    - detection of cervical spine fracture on the slice-level, vertebra-level, and patient-level
    - fracture localization based on object detection model

### Facial Expression Recognition with Label Distribution Learning

[code](FacialExpressionRecognition)
| [poster](FacialExpressionRecognition/docs/poster.png)
| [slides](FacialExpressionRecognition/docs/slides.pdf)
| [report](FacialExpressionRecognition/docs/report.pdf)
| [website](https://wp.cs.hku.hk/2021/fyp21022/)

- Facial expression recognition based on multi-label learning, with
    - ERT-based alignment technique to detect and align all faces
    - implementation of VGG, region attention network, multi-task EfficientNet, etc.
    - a detailed study of the effect of model pre-training, loss function, multi-emotion learning, etc.

### Kubernetes Operator Maturity Levels

[code](https://github.com/KerryDRX/visitors-operator)
| [paper](https://ieeexplore.ieee.org/abstract/document/9658981)

- Implementation of five maturity levels of a Kubernetes operator based on Go and Operator SDK, including
    - basic install: straightforward installation achieved by Kubernetes/Helm
    - seamless upgrades: simple upgrade by modification of yaml configuration file
    - full lifecycle: data backup and recovery functions implemented with MySQL Operator and Google Cloud
    - deep insights: monitoring and alert systems implemented with Prometheus
    - autopilot: frontend, backend, and database autoscaling functions implemented with Horizontal Pod Autoscaler

### Later-Life Health Prediction

[poster](HealthAnalysis/poster.pdf)
| [report](HealthAnalysis/report.pdf)

- Study of early-life socioeconomic status, lifestyle, and personality on later-life health with statistical learning by
    - considering 50 predictors collected from the 1957~2011 Wisconsin Longitudinal Study
    - implementing random forest, smoothing splines, LASSO, bidirectional stepwise regression with RStudio
    - identifying factors with the most impact on senior health based on coefficients and feature importance

### City Crime Data Analysis

[code](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2139130357244218/1874002467160109/8678045098065098/latest.html)

- Crime data analysis of a target city with
    - investigation of time, locations, crime types, weapons, and criminal information by Apache Spark
    - visualization of crime locations and types within any specified time period and region using GeoPandas
    - crime location clustering and hot zone identification based on bisection K-means

### Image-Text Matching and Image Search Engine

[code](ImageTextMatching)

- An image search engine is constructed by matching images and texts, based on
    - image and text feature extraction by pre-trained models
    - training of Siamese network with contrastive loss
    - implementation of an image search application with Flask

### Product Review Topic Modeling

[code](TopicModeling)

- Unsupervised clustering and topic modeling of e-commerce product reviews for customer sentiment analysis, by
    - tokenization and stemming of raw data
    - feature engineering with TF-IDF model
    - clustering and topic identification by K Means Clustering and Latent Dirichlet Allocation

### Movie Recommendation System

[code](RecommendationSystem)

- A movie recommendation system with
    - explorative data analysis of the movie rating dataset by Spark SQL
    - implementation of Alternating Least Square (ALS) Matrix Factorization with Spark ML
    - recommendation system constructed based on ALS rating predictions

### Bank Customer Churn Prediction

[code](ChurnPrediction)

- Detection of bank customers who are likely to churn in the future, with
    - four implemented supervised machine learning models: logistic regression, K nearest neighbors, random forest, and XGBoost
    - identification of factors with the most influence on user retention based on feature importance and SHAP values
