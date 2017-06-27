#!/usr/bin/env python
'''
Script to create the validation window file used for training RCNN. Currently,
only supports the valdiation data set which was the only one used to train the
FPGA.
'''
import os
import sys
import numpy as np
import scipy.io as sio
import h5py
import xml.etree.ElementTree as ET
import argparse

# Directory holding all precomputed selective_search mat files
# Defaults to selective_search_data if running the
# fetch_selective_search_data.sh script
SELECTIVE_SEARCH_DIR = "selective_search_data/"

# specify the mat mat window files that you want processed
MAT_FILE_BOXES = [
    "ilsvrc13_val1.mat",
    "ilsvrc13_val2.mat"]

IMG_ROOT = "ILSVRC14_DET/ILSVRC2013_DET_val/",
BBOX_XML_ROOT = "ILSVRC14_DET/ILSVRC2013_DET_bbox_val/"
SYNSET_IDS = "/opt/caffe/data/ilsvrc12/det_synset_words.txt"


def create_window_file(args):

    synset_to_id = {}
    channels = 3        # assumed RGB three channels
    thr = 1e-5          # threshold for IoU choice

    with open(args.synset_file) as fid:
        for i, line in enumerate(fid):
            line = line.strip().split()
            synset_to_id[line[0]] = i + 1

    for fmat in args.input_files:
        images, boxes = process_mat(fmat)
        fname = "{}/window_data_{}.txt". \
                format(args.outdir, fmat[fmat.rfind('/') + 1:fmat.rfind('.')])

        if os.path.isfile(fname):
            print "Skipping file... {} already exists".format(fname)
            continue

        assert len(images) == len(boxes)

        with open(fname, "w+") as fout:

            for i in xrange(len(images)):

                print_status(fname, i + 1, len(images))

                # parse XML data for ground truth boxes
                tree = ET.parse("{}{}.xml".format(args.bbox_file, images[i]))
                root = tree.getroot()
                size = root.find("size")

                # regions of interest to hold all bounding boxes for current
                # image
                rois = boxes[i].astype(np.int32)

                # arrange in correct format (xmin, ymin, xmax, ymax)
                rois[:, [0, 1]] = rois[:, [1, 0]]
                rois[:, [2, 3]] = rois[:, [3, 2]]

                # make sure that bounding boxes fit within image otherwise
                # opencv will throw an error when training
                assert rois[rois[:, 0] < 0].sum() == 0 and \
                    rois[rois[:, 1] < 0].sum() == 0 and \
                    rois[rois[:, 2] > int(size[0].text)].sum() == 0 and \
                    rois[rois[:, 3] > int(size[1].text)].sum() == 0

                # XML stored as:
                #   object[0] = name (wnid)
                #   object[1] = bndbox
                gt_boxes_labels = [([float(o[1].find("xmin").text),
                                     float(o[1].find("ymin").text),
                                     float(o[1].find("xmax").text),
                                     float(o[1].find("ymax").text)],
                                   synset_to_id[o[0].text])
                                   for o in root.findall("object")]

                num_gt_boxes = len(gt_boxes_labels)
                num_boxes = np.size(rois, 0)

                overlap = np.zeros((num_boxes, 2))

                for gt_box, label in gt_boxes_labels:

                    iou = bbox_intersection(rois, gt_box)
                    max_select = overlap[:, 0] < iou

                    # [IoU_overlap, label]
                    overlap[max_select] = np.c_[
                        iou[max_select],
                        label * np.ones(max_select.sum(), dtype=np.int32)]

                im_paths = ["{}{}.{}".format(args.image_dir, img, args.ext)
                            for img in images]

                fout.write("# {}\n".format(i))
                fout.write("{}\n".format(im_paths[i]))
                fout.write("{}\n{}\n{}\n".format(
                    channels,
                    size[1].text,  # height
                    size[0].text   # width                             )
                ))

                fout.write("{}\n".format(num_boxes + num_gt_boxes))

                # write all bounding boxes to file in format
                # label, overlap, xmin, ymin, xmax, ymax
                for i in range(num_boxes):
                    ov, label = overlap[i, 0], overlap[i, 1]

                    # change all overlapping areas that fail to meet threshold
                    # limit into background
                    if ov < thr:
                        ov = 0
                        label = 0

                    bbox = rois[i]
                    bbox = map(int, bbox)

                    fout.write("{} {:.3f} {} {} {} {}\n"
                               .format(int(label), ov, *bbox))

                # write all ground truth boxes to file
                for gt_box, label in gt_boxes_labels:
                    gt_box = map(int, bbox)
                    fout.write("{} {:.3f} {} {} {} {}\n"
                               .format(int(label), 1, *gt_box))

            # clears a new line
            print


def process_mat(fname):
    if fname.endswith(".mat"):
        f = sio.loadmat(fname)

        # breaks image names from nested numpy ndarrays into a simple list
        # nummpy arrays not necessary for string manipulation
        images = np.concatenate(f["images"].tolist()).ravel().tolist()

        boxes = f["boxes"][:, 0].tolist()

        return images, boxes

    # if hdf5 format, build numpy array in same format
    elif fname.endswith(".h5"):
        f = h5py.File(fname)

        # convert images and boxes to lists from hdf5 format
        # transpose the boxes to match the shape of sio.loadmat shape
        boxes = [np.array(f[ref]).T for ref in f["boxes"][0]]

        # hdf5 stores strings in vstacked arrays by their ascii value... yup
        images = [''.join([chr(c) for c in f[ref]]) for ref in f["images"][0]]

        return images, boxes
    else:
        raise TypeError("File extension not supported.")


def bbox_intersection(boxes, gt):
    '''Assume values stored as (xmin, ymin, xmax, ymax)
    '''

    x1 = np.maximum(boxes[:, 0], gt[0])
    y1 = np.maximum(boxes[:, 1], gt[1])
    x2 = np.minimum(boxes[:, 2], gt[2])
    y2 = np.minimum(boxes[:, 3], gt[3])

    v_pos_area = np.vectorize(pos_area)

    area_intersect = v_pos_area(x1, y1, x2, y2)
    area_union = pos_area(*gt) + v_pos_area(*boxes.T) - area_intersect

    # check that no division by zero can occur
    assert area_union.all()

    return area_intersect / area_union


def pos_area(x1, y1, x2, y2):
    '''standard area calculation but negative values become zero
    '''
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def print_status(f, i, l):

    sys.stdout.write("\rMaking {}: {}/{} = {}%".format(f, i, l, 100 * i / l)),
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "outdir",
        help="Directory to store WindowData files"
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        default=[SELECTIVE_SEARCH_DIR + n for n in MAT_FILE_BOXES],
        help="Space separated list of input .mat or .h5 selective search data"
             "files")
    parser.add_argument(
        "--synset_file",
        default=SYNSET_IDS,
        help="List of synset_ids and wordnet IDS. Format: <synset_id> <WNID>. "
             "Only one synset id per line")
    parser.add_argument(
        "--bbox_file",
        default=BBOX_XML_ROOT,
        help="XML file with all the bounding boxes and labels for validation"
             "images. Can be downloaded with script.")
    parser.add_argument(
        "--image_dir",
        default=IMG_ROOT,
        nargs=2,
        help="Directory of validation images")
    parser.add_argument(
        "--ext",
        default="JPEG",
        help="File extension. (Default is JPEG)"
    )
    args = parser.parse_args()

    create_window_file(args)
