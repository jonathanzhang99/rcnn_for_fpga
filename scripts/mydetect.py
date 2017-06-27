#!/usr/bin/env python
"""
detector.py is an out-of-the-box windowed detector
callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
Note that this model was trained for image classification and not detection,
and finetuning for detection can be expected to improve results.

The selective_search_ijcv_with_python code required for the selective search
proposal mode is available at
    https://github.com/sergeyk/selective_search_ijcv_with_python

The original detect.py has been modified to
1.  Run selective search natively with python using the selectivesearch module
    from pip and with crop_mode set to pycrop
2.  Displays bounding boxes for prominent detections.
    a.  Same labels share the same color. Bounding Boxes, category, and
        and confidence are all displayed.
    b.  NMS is performed by default to eliminate redundancy.
    c.  Two plots are shown. The first plot shows all bounding boxes that have
        significant confidence levels while the second plot only shows the top
        bounding box in the 3 highest confidence categories across all boxes.
3.  Accepts individual images to parse as an input file
4   Accepts hdf5 binaries saved from a previous mydetect.py run and displays
    bounding boxes for that image

Edited by Jonathan Zhang @jonathanzhang99 

TODO: Currently does not work with multiple image names in crop_mode list form.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time

import selectivesearch
import caffe

CROP_MODES = ['list', 'selective_search', 'pycrop']
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']
IMG_TYPES = ('jpeg', 'jpg', 'png')


def main(argv):
    # Change default locations to match personal system setup.
    parser = argparse.ArgumentParser()
    # Required arguments: input and output.
    parser.add_argument(
        "input_file",
        help="Input txt/csv/h5/image filename. If .txt, must be list of \
        filenames. If .csv, must be comma-separated file with header \
        'filename, xmin, ymin, xmax, ymax'. If image file, must have extension \
        JPEG/jpg/png. If h5 file, must be the result of RCNN."
    )
    parser.add_argument(
        "output_file",
        help="Output h5/csv filename. Format depends on extension."
    )
    # Optional Arguments
    parser.add_argument(
        "--model",
        default="/home/jonathanzhang/workspace/models/fpga_vgg16/"
                "VGG_ILSVRC_16_layers_deploy.prototxt",
        help="Model definition file."
    )
    parser.add_argument(
        "--weights",
        default="/home/jonathanzhang/workspace/models/fpga_vgg16/finetune_fpga_2/"
        "ilsvrc13_det_finetune_fpga_2_iter_60000.caffemodel",
        help="Trained model weights file."
    )

    parser.add_argument(
        "--crop_mode",
        default="pycrop",
        choices=CROP_MODES,
        help="How to generate windows for detection."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--mean_file",
        default=None,
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=None,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )
    parser.add_argument(
        "--context_pad",
        type=int,
        default='16',
        help="Amount of surrounding context to collect in input window."
    )
    parser.add_argument(
        "--image_test",
        default=True,
        action='store_true',
        help="Show bounding box plots of input images. Default is True")
    parser.add_argument(
        "--caffe_root",
        default="/opt/caffe/",
        help="Default location of caffe files")
    parser.add_argument(
        "--workspace_root",
        default="/home/jonathanzhang/workspace/",
        help="Default location to search for images and synset_list")

    args = parser.parse_args()

    if args.input_file.strip().endswith('h5'):
        print "Loading hdf5 file..."
        process_results(pd.read_hdf(args.input_file.strip()),
                        args.workspace_root,
                        args.caffe_root)
        return

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
        if mean.shape[1:] != (1, 1):
            mean = mean.mean(1).mean(1)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    # Make detector.
    detector = caffe.Detector(args.model, args.weights,
                              mean=mean,
                              input_scale=args.input_scale,
                              raw_scale=args.raw_scale,
                              channel_swap=channel_swap,
                              context_pad=args.context_pad)

    # Load input.
    print("Loading input...")
    if args.input_file.lower().endswith('txt'):
        with open(args.input_file) as f:
            inputs = [_.strip() for _ in f.readlines()]
    elif args.input_file.lower().endswith('csv'):
        inputs = pd.read_csv(args.input_file, sep=',', dtype={'filename': str})
        inputs.set_index('filename', inplace=True)
    elif args.input_file.lower().endswith(IMG_TYPES):
        inputs = [args.input_file]
    else:
        raise Exception("Unknown input file type: not in txt, csv, jpg, jpeg,"
                        "or png.")

    # Detect.

    if args.crop_mode == 'list':
        # Unpack sequence of (image filename, windows).
        images_windows = [
            (ix, inputs.iloc[np.where(inputs.index == ix)][COORD_COLS].values)
            for ix in inputs.index.unique()
        ]

    elif args.crop_mode == 'pycrop':
        # Thresholds for Felzenszwalb and Huttenlocher segmentation algorithm
        # By default, we set minSize = k, and sigma = 0.8
        # k = scale
        t1 = time.time()
        print "Calculating region propsals..."

        ks = [20, 50, 100, 150, 200, 250, 300, 350, 500, 1000]
        sigma = 0.8

        ssearch_regions = [[] for i in range(len(inputs))]
        region_candidates = []
        # Process all images
        for imname in inputs:
            img = caffe.io.load_image(imname).astype(np.float32)

            # k is the Segmentation Threshold
            for k in ks:
                _, regions = selectivesearch.selective_search(
                    img, scale=k, sigma=sigma, min_size=k)

                ssearch_regions[i] += regions

            for image_regions in ssearch_regions:

                # Extract all unique boxes
                candidates = set()

                for region in image_regions:

                    # caffe.Detector expects coordinates to be in
                    # (ymin, xmin, ymax, xmax) format
                    x, y, w, h = region['rect']
                    region['rect'] = (y, x, y + h, x + w)

                    if region['rect'] in candidates or h < 20 or w < 20:
                        continue

                    candidates.add(region['rect'])

                region_candidates.append(np.array(list(candidates)))

        #  print(zip(inputs, region_candidates))
        images_windows = zip(inputs, region_candidates)

        print "Found {} region propsals in {:.3f} seconds.".format(
            np.size(region_candidates[0], 0), time.time() - t1)

    t = time.time()
    detections = detector.detect_windows(images_windows)
    print("Processed {} windows in {:.3f} s.".format(len(detections),
                                                     time.time() - t))

    # Collect into dataframe with labeled fields.
    df = pd.DataFrame(detections)
    df.set_index('filename', inplace=True)
    df[COORD_COLS] = pd.DataFrame(
        data=np.vstack(df['window']), index=df.index, columns=COORD_COLS)
    del(df['window'])

    # Save results.
    t = time.time()
    if args.output_file.lower().endswith('h5'):
        df.to_hdf(args.output_file, 'df', mode='w')
        print("Saved to {} in {:.3f} s.".format(args.output_file,
                                                time.time() - t))
    else:
        print "Invalid file type"

    if args.image_test:
        process_results(df, args.workspace_root, args.caffe_root)


def process_results(df, workspace_root, caffe_root):
    prob_thr, overlap_thr = 0.5, 0.2
    print "Processing results of image detection..."
    with open(workspace_root + "_temp/det_synset_words.txt") as f:
        labels_df = pd.DataFrame([
            {
                'synset_id': l.strip().split(' ')[0],
                'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
            }
            for l in f.readlines()
        ])

    predictions_df = pd.DataFrame(np.vstack(df.prediction.values),
                                  columns=labels_df['name'])

    # max_all: maximum values wrt all windows
    max_all = predictions_df.max(0)
    max_all.sort_values(inplace=True, ascending=False)
    print "Top 10 classifications"
    print max_all[:10]

    # I asssume that the background (max_all[0]) will have the highest
    # percentage classification. Better method would be to remove the background
    # node entirely from the output layer but that adds loads of complexity.
    # This if statement allows something to be shown even if the probabilities
    # are extremely low.
    if max_all[1] < overlap_thr:
        overlap_thr = max(0, max_all[1] - 0.1)

    # max_b: maximum values wrt each individual bounding box
    max_b = pd.DataFrame([predictions_df.max(1).values, predictions_df.index],
                         index=["scores", "df_idx"],
                         columns=predictions_df.idxmax(1).values).transpose()
    filt_b = (max_b.index.values != "background") & (max_b.scores > prob_thr)
    max_b = max_b[filt_b]
    top_classes_b = max_b.index.unique()
    if len(top_classes_b) > 5:
        print "Many objects detected... limiting to 5"
    img = plt.imread(df.index[0])
    plt.imshow(img)
    current_ax = plt.gca()
    colors = ['r', 'b', 'y', 'g', 'm', 'c']
    for color, c in zip(colors, top_classes_b[:5]):

        c_slice = max_b[max_b.index.values == c]
        nms_windows = df.iloc[c_slice.df_idx][['xmin', 'ymin', 'xmax', 'ymax']]\
            .values
        dets = np.hstack(
            (nms_windows, np.vstack(c_slice[c_slice.index == c].scores.values)))
        nms_dets = nms_detections(dets, overlap_thr)
        for det in nms_dets:
            current_ax.add_patch(
                plt.Rectangle(
                    (det[0], det[1]), det[2] - det[0], det[3] - det[1],
                    fill=False, edgecolor=color, linewidth=3))
            current_ax.text(det[0], det[1], "{}: {:.3f}".format(c, det[4]),
                            color=color)

    print max_b
    print "_temp/{}.png".format(df.index[0].split('/')[-1].split('.')[0])
    plt.savefig("_temp/{}.png".format(df.index[0].split('/')[-1].split('.')[0]),
                bbox_inches='tight')
    plt.show()

    # plt.gray()
    # plt.matshow(predictions_df.values)
    # plt.xlabel('Classes')
    # plt.xlabel('Windows')

    # Top 3 classification excluding the background
    top_class = [c for c in max_all.index[:4] if c != 'background'][:3]

    idx = [predictions_df[c].argmax() for c in top_class]

    plt.imshow(img)
    currentAxis = plt.gca()

    det = [df.iloc[i] for i in idx]
    coords = [((d['xmin'], d['ymin']),
              d['xmax'] - d['xmin'],
              d['ymax'] - d['ymin']) for d in det]
    colors = ['r', 'b', 'y', 'g', 'm', 'c']
    for i in range(len(coords)):
        currentAxis.add_patch(plt.Rectangle(
            *coords[i], fill=False, edgecolor=colors[i % 3], linewidth=5))
        currentAxis.text(coords[i][0][0], coords[i][0][1],
                         "{}: {:.3f}".format(
                             top_class[i], max_all[top_class[i]]))
    plt.show()


def nms_detections(dets, overlap=0.3):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    dets: ndarray
        each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
    overlap: float
        minimum overlap ratio (0.3 default)

    Output
    ------
    dets: ndarray
        remaining after suppression.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    ind = np.argsort(dets[:, 4])

    w = x2 - x1
    h = y2 - y1
    area = (w * h).astype(float)

    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]

        xx1 = np.maximum(x1[i], x1[ind])
        yy1 = np.maximum(y1[i], y1[ind])
        xx2 = np.minimum(x2[i], x2[ind])
        yy2 = np.minimum(y2[i], y2[ind])

        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)

        wh = w * h
        o = wh / (area[i] + area[ind] - wh)

        ind = ind[np.nonzero(o <= overlap)[0]]

    return dets[pick, :]


if __name__ == "__main__":
    import sys
    main(sys.argv)
