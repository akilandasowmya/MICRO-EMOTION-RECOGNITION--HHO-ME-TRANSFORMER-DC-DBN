# MICRO-EMOTION-RECOGNITION--HHO-ME-TRANSFORMER-DC-DBN
Description

This repository provides the implementation used in the paper:

“ Facial micro-emotion recognition using harris hawk optimization-based micro-expression transformer and deep learning classifier.”

This project aimed to develop a comprehensive pipeline for micro-expression analysis, involving hybrid feature extraction (HOG, LBP, SIFT), dimension reduction, deep feature learning with a dual-input transformer, and multi-class emotion classification using a Deep Convolutional Deep Belief Network (DC-DBN).
Key Stages and Achievements:
1. Image Data Loading and Preprocessing:
•	Images from the dataset were successfully loaded.
•	A corresponding annotation file was loaded to retrieve multi-class emotion labels: "Tense," "Happiness," "Repression," "Disgust," "Surprise," and "Contempt."
•	A crucial data alignment step was performed, filtering out images without valid emotion labels, resulting in a perfectly aligned dataset and their corresponding labels.
•	Raw pixel data for these images was preprocessed (resized to 64x64, converted to grayscale, and flattened) for use as a second input to the transformer model.
2. Traditional Feature Extraction (HOG, LBP, SIFT):
•	HOG Features: Extracted from all images, resulting in a hog_features_multiclass.csv.
•	LBP Features: Extracted from all images, producing lbp_features_multiclass.csv.
•	SIFT Features: Extracted from all images, saved to sift_features_multiclass.csv.
•	All three feature sets were combined into final_combined_df_multiclass and saved to combined_features_multiclass.csv.
3. Dimension Reduction and Feature Selection:
•	PCA for Dimension Reduction: Principal Component Analysis (PCA) was applied to the combined_features_multiclass.csv dataset. These pca_reduced_features_multiclass.csv  were specifically prepared as an input stream for the dual-input transformer.
4. Dual-Input Transformer Model Tuning and Deep Feature Extraction:
•	Dual-Input Architecture: A Keras Functional API-based transformer model (create_dual_input_transformer_model) was developed to process two distinct input streams: PCA-reduced features and raw pixel data. These streams were processed and then concatenated before passing through common dense layers to extract 'deep features.'
•	HHO for Hyperparameter Tuning: The HHO algorithm was re-purposed to tune the hyperparameters of this dual-input transformer model for multi-class classification. Evaluating the model using 5-fold stratified cross-validation on the actual emotion labels (y_multi_class), HHO successfully identified optimal hyperparameters:
o	num_layers: 1
o	hidden_units: 32
o	dropout_rate: 0.1
o	learning_rate: ~0.0008 The best fitness (1 - mean accuracy) achieved during this tuning was 0.5227, corresponding to a mean accuracy of ~47.73% for multi-class classification during the tuning phase.
•	Deep Feature Extraction: Using the HHO-tuned parameters, the dual-input transformer model extracted 32 deep features for each of the images.
•	These deep features were saved to deep_features_pca.csv.
5. Deep Convolutional Deep Belief Network (DC-DBN) for Multi-Class Classification:
•	DC-DBN Architecture: A conceptual DC-DBN model was defined using Keras Sequential API, consisting of two RBM-like Dense layers, followed by a classification block, and a final Dense layer with softmax activation for 6-class output.
•	Training and Evaluation: The DC-DBN was trained using the extracted deep_features_pca and the one-hot encoded multi-class emotion labels (y_multi_class). It was trained for 100 epochs, with 80% of the data used for training and 20% for testing (stratified split).
•	Outstanding Performance: The DC-DBN achieved impressive results on the test set:
o	Test Loss: 0.1629
o	Test Accuracy: 0.9160 (91.6%)
o	Precision (weighted): 0.9210
o	Recall (weighted): 0.9160
o	F1-Score (weighted): 0.9126
•	Performance for individual classes was strong, with perfect F1-scores for 'surprise,' 'contempt,' and 'happiness,' while 'disgust' showed comparatively lower recall (0.50), suggesting it's a more challenging class.

