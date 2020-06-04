#!/bin/sh
# How to use:
# sh head_swap_pipeline.sh {data_source_path} {preprocess_save_path} {gpu_id}  {model_name} {result_dir} {fake_face_folder}  {mask_segmentation_path}
# e.g. # sh head_swap_pipeline.sh /home/mo/experiments/masterthesis/pipeline_results/source_data /home/mo/experiments/masterthesis/pipeline_results/preprocessing 1

# exit script on error
set -e


#
echo "Running Face Detection"
cd ../head_swap/
poetry run python preprocess/face_detector.py --data_path "$1" --file_type '.png' --save_path "$2"/cutout_face --margin_multiplier 0.33

echo "Running Segmentation."
cd ../segmentation/
poetry run python scripts/eval.py --model deeplabv3 --save_path "$2"/segmentation --save_pred --no_metric

echo "Running Preprocessing."
cd ../head_swap/
poetry run python preprocess/create_blending_dataset.py --data_path "$2"/cutout_face --save_path "$2"/test --do_not_transform --segmentation_path "$2"/segmentation --with_segmentation --use_fake_face --fake_face_folder  "$6" --mask_segmentation_path  "$7"

echo "Running Blending/Inpainting Model"
poetry run python network/test.py --dataroot "$2" --model pix2pix_vgg_loss --gpu_ids "$3" --name "$4"  --direction BtoA --num_test 20000 --results_dir "$5"

echo "Reinserting face"
poetry run python preprocess/reinsert_face.py
echo "finished"
