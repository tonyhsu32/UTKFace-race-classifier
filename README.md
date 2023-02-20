## UTKFace race classifier ##

1. **UTKFace data analysis:**  

   UTKFace dataset source -> (https://susanqq.github.io/UTKFace/)  
   UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity.  
   
   * UTKFace-analysis.ipynb  
   * UTKFace-anomaly-detector.ipynb
   * UTKFace-train_test_split-to-tfrecords.ipynb  
 
 
2. **UTKFace race data training and predict:**  
   * UTKFace-CNN-race-classifier.ipynb  
   * UTKFace-all_model_predict_curves.ipynb 
   
   ***model history***
   * **VGG16**_history.json 
   
   Use **ResNet50** -> val accuracy is ≈ **0.81 ~ 0.82**.  
   * **ResNet50-freeze**_history.json  
   * **ResNet50-unfreeze**_history.json  
   ![ResNet50](https://github.com/tonyhsu32/UTKFace-race-classifier/blob/main/ResNet50.png)  
   
   Use **Xception** -> val accuracy is ≈ **0.84**.
   * **Xception-freeze**_history.json  
   * **Xception-unfreeze**_history.json  
   ![Xception](https://github.com/tonyhsu32/UTKFace-race-classifier/blob/main/Xception.png)
   
   ***Self-Supervised learning weights (Noisy-student, ImageNet21K, ImageNet21K-ft1K)***  
   
   Use **EfficientnetV2-l-21k-ft1k** -> val accuracy is ≈ **0.85**.
     * **EfficientnetV2-l-21k-ft1k**_history.json  
     ![EfficientnetV2-l-21k-ft1k](https://github.com/tonyhsu32/UTKFace-race-classifier/blob/main/EfficientnetV2-l-21k-ft1k.png)  
   
   **Total model test view**  
   * model_predict_curves_3.png
   ![Total model](https://github.com/tonyhsu32/UTKFace-race-classifier/blob/main/model_predict_curves_3.png)
   
 3. **Tools:**  
    * efficientnet_weight_update_util.py  
    * Convert_h5_to_pb_script.ipynb
   
 4. Use **Confusion Matrix:**
    * EffV2_l_21k_tf1k_confusion_matrix_batch(128).png  
    ![confusion_matrix](https://github.com/tonyhsu32/UTKFace-race-classifier/blob/main/EffV2_l_21k_tf1k_confusion_matrix_batch(128).png)  
   
 5. Use **T-SNE Visualization:**  
    * UTKFace(T-SNE).png  
    ![T-SNE](https://github.com/tonyhsu32/UTKFace-race-classifier/blob/main/UTKFace(T-SNE).png)
     
     
     
   
