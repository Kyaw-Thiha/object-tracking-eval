
# --- Uncertainty Tracking Evaluation Pipeline ---
# uncertainty_tracker, identity covariance
# python evaluation_pipeline.py \
#     --dataloader_factory mot17_factory.py \
#     --dataset_dir /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/ \
#     --example_image_path /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/img1/000001.jpg \
#     --model_factory opencv_yolox_factory.py \
#     --tracker uncertainty_tracker \
#     --device cpu \
#     --output_dir /home/allynbao/project/UncertaintyTrack/src/outputs/test_pipeline_identity_uncertainty/ \
#     --eval_result_dir /home/allynbao/project/UncertaintyTrack/src/evaluation_results/test_pipeline_identity_uncertainty/ \
#     --plot_save_path /home/allynbao/project/UncertaintyTrack/src/plots/test_pipeline_identity_uncertainty/

# uncertainty_tracker, image noise covariance
# python evaluation_pipeline.py \
#     --dataloader_factory mot17_factory.py \
#     --dataset_dir /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/ \
#     --example_image_path /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/img1/000001.jpg \
#     --model_factory opencv_yolox_factory_image_noise.py \
#     --tracker uncertainty_tracker \
#     --device cpu \
#     --output_dir /home/allynbao/project/UncertaintyTrack/src/outputs/test_pipeline_image_noise_uncertainty/ \
#     --eval_result_dir /home/allynbao/project/UncertaintyTrack/src/evaluation_results/test_pipeline_image_noise_uncertainty/ \
#     --plot_save_path /home/allynbao/project/UncertaintyTrack/src/plots/test_pipeline_image_noise_uncertainty/

# uncertainty_tracker, prob yolox covariance
python evaluation_pipeline.py \
    --dataloader_factory mot17_factory.py \
    --dataset_dir /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/ \
    --example_image_path /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/img1/000001.jpg \
    --model_factory prob_yolox_x_es_mot17_half_factory.py \
    --tracker uncertainty_tracker \
    --device cpu \
    --output_dir /home/allynbao/project/UncertaintyTrack/src/outputs/test_pipeline_prob_yolox_x_uncertainty/ \
    --eval_result_dir /home/allynbao/project/UncertaintyTrack/src/evaluation_results/test_pipeline_prob_yolox_x_uncertainty/ \
    --plot_save_path /home/allynbao/project/UncertaintyTrack/src/plots/test_pipeline_prob_yolox_x_uncertainty/

# --- Probablistic Byte Track Evaluation Pipeline ---
# probablistic byte track, identity covariance
# python evaluation_pipeline.py \
#     --dataloader_factory mot17_factory.py \
#     --dataset_dir /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/ \
#     --example_image_path /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/img1/000001.jpg \
#     --model_factory opencv_yolox_factory.py \
#     --tracker probabilistic_byte_tracker \
#     --device cpu \
#     --output_dir /home/allynbao/project/UncertaintyTrack/src/outputs/test_pipeline_identity_prob_byte_track/ \
#     --eval_result_dir /home/allynbao/project/UncertaintyTrack/src/evaluation_results/test_pipeline_identity_prob_byte_track/ \
#     --plot_save_path /home/allynbao/project/UncertaintyTrack/src/plots/test_pipeline_identity_prob_byte_track/

# probablistic byte track, image noise covariance
# python evaluation_pipeline.py \
#     --dataloader_factory mot17_factory.py \
#     --dataset_dir /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/ \
#     --example_image_path /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/img1/000001.jpg \
#     --model_factory opencv_yolox_factory_image_noise.py \
#     --tracker probabilistic_byte_tracker \
#     --device cpu \
#     --output_dir /home/allynbao/project/UncertaintyTrack/src/outputs/test_pipeline_image_noise_0.0005_prob_byte_track/ \
#     --eval_result_dir /home/allynbao/project/UncertaintyTrack/src/evaluation_results/test_pipeline_image_noise_0.0005_prob_byte_track/ \
#     --plot_save_path /home/allynbao/project/UncertaintyTrack/src/plots/test_pipeline_image_noise_0.0005_prob_byte_track/


# --- Proablistic OCSORT Tracking Evaluation Pipeline ---
# probablistic OCSORT, identity covariance
# python evaluation_pipeline.py \
#     --dataloader_factory mot17_factory.py \
#     --dataset_dir /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/ \
#     --example_image_path /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/img1/000001.jpg \
#     --model_factory opencv_yolox_factory.py \
#     --tracker prob_ocsort_tracker \
#     --device cpu \
#     --output_dir /home/allynbao/project/UncertaintyTrack/src/outputs/test_pipeline_identity_prob_ocsort_track/ \
#     --eval_result_dir /home/allynbao/project/UncertaintyTrack/src/evaluation_results/test_pipeline_identity_prob_ocsort_track/ \
#     --plot_save_path /home/allynbao/project/UncertaintyTrack/src/plots/test_pipeline_identity_prob_ocsort_track/

# probablistic OCSORT, image noise covariance
# python evaluation_pipeline.py \
#     --dataloader_factory mot17_factory.py \
#     --dataset_dir /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/ \
#     --example_image_path /home/allynbao/project/UncertaintyTrack/src/data/MOT17/train/MOT17-02-DPM/img1/000001.jpg \
#     --model_factory opencv_yolox_factory_image_noise.py \
#     --tracker prob_ocsort_tracker \
#     --device cpu \
#     --output_dir /home/allynbao/project/UncertaintyTrack/src/outputs/test_pipeline_image_noise_prob_ocsort_track/ \
#     --eval_result_dir /home/allynbao/project/UncertaintyTrack/src/evaluation_results/test_pipeline_image_noise_prob_ocsort_track/ \
#     --plot_save_path /home/allynbao/project/UncertaintyTrack/src/plots/test_pipeline_image_noise_prob_ocsort_track/