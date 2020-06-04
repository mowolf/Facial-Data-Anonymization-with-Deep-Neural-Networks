import json

import cv2

if __name__ == '__main__':

    scale_down = True
    # load keypoint data
    with open('/home/mo/experiments/masterthesis/pipeline_results/preprocessing/cutout_face/coordinates.txt', 'r') as f:
        data = json.load(f)

    for key, item in data.items():
        x, y, w, h = item["rect_cv"]

        original = cv2.imread(f'/home/mo/experiments/masterthesis/pipeline_results/source_data/{key[2:]}.png')
        new = cv2.imread(
            f'/home/mo/experiments/masterthesis/pipeline_results/results/pix2pix_l1_vgg_weighted_ff_face_seg_v3/Apr-21_test_latest_aligned/images/{key}_fake_B.png')

        if h > 255 or w > 255 and scale_down:
            # scale original down
            # w should be h
            assert w == h
            scale = 256 / w
            original = cv2.resize(original, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            x = int(x * scale)
            y = int(y * scale)
            w = int(w * scale)
            h = int(h * scale)
        else:
            # scale new to fit original resolution
            shape = original.shape
            h_adjust = max(0, y + h - shape[0])
            w_adjust = max(0, x + w - shape[1])
            new = cv2.resize(new, (w - w_adjust, h - h_adjust))

        print(f"Reinserting {key}")
        original[y:y + h, x:x + w, :] = new

        cv2.imwrite(f"/home/mo/experiments/masterthesis/pipeline_results/results/merged/{key}.png", original)
