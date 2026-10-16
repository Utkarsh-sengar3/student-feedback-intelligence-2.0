
 # Info on Dataset File Splitting

The full dataset was annotated in 65 batches of 100 entries each, which are available in [Annotation_raw_output > ASQE_DPT_export_ZIP/](Annotation_raw_output/ASQE_DPT_export_ZIP)

To enable more diverse sampling criteria, we pre-split the 65 batches into two groups:
* Train_validation:  to be further sampled for training and in-training validation
* DEV_TEST: to be further sampled for development (hyper-parameter tuning) or hold-out test

### Batch mapping for both files

 Train_validation_file = 40 batches

 Dev_test_file = 25 batches

 Train_Val batches = [43, 16, 44, 4, 14, 12, 48, 62, 58, 45, 33, 2, 47, 19, 26, 56, 21, 50, 27, 22, 46, 8, 54, 17, 5, 25, 37, 18, 3, 36, 10, 24, 9, 13, 39, 41, 53, 42, 35, 51]

 Dev_Test batches = [55, 38, 23, 20, 60, 30, 7, 34, 64, 49, 59, 40, 15, 52, 28, 61, 32, 6, 1, 65, 29, 63, 57, 11, 31]