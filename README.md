# rcnn_for_fpga

Region-based Convolutional Neural Network (RCNN) finetuned for an FPGA CNN-accelerated chip. Since the FPGA chips requires a specific convolutional layer, we finetune a VGG-Net pretrained from ImageNet and optimized for the FPGA by freezing all convolutional layers.

Closely modeled from the [original RCNN paper](https://github.com/rbgirshick/rcnn). <b>NOTE:</b> My version of RCNN has removed the SVM classifier and Bounding Box regressor to better demonstrate the finetuning process at the expense of accuracy.


## Installation and Usage
1.	Install Caffe. Refer to the [Caffe website](http://caffe.berkeleyvision.org/) for more information
2.	Download the Image sets (JPEG) and Annotation data (XML). In my finetuning, I used the [ILSVRC14 DET set](http://image-net.org/challenges/LSVRC/2014/download-images-5jj5.php). Store these files in `$WORKSPACE/ILSVRC14_DET` where `$WORKSPACE` is your current working directory.
3.	Create the necessary WindowData Layer by running 
	```shell
	./scripts/fetch_selective_search_data.sh
	./scripts/create_window_data.py window_data
	```
	val1 should be used for training and val2 for validation.
	(You may need to put your own settings in `create_window_data.py`. Use `-h` flag for help)
4.	Put your caffemodel into `$WORKSPACE/models/fpga_vgg16`. I used a variant of the VGG16 model with fixed point precision Convolutional Layers.
5.	Train your network. Make sure that the source parameter of the data layer `vgg16_freeze_finetune_trainval_test.prototxt` points to the correct directory of your window_data file(s). Depending on your GPU, batch size may also need to be decreased. Change the `net` and `snapshot` parameter of the `vgg16_freeze_finetune_solver.prototxt` to properly reflect your own working directory.
	```shell
	cd $WORKSPACE
	caffe train solver=./models/fpga_vgg16/vgg16_freeze_finetune_solver.prototxt weights=./models/fpga_vgg15/<YOUR_PRETRAINED_MODEL>.caffemodel --gpu all
	```
	Training should take roughly 10-13 hours on GPU. (Tested with GTX 1070 ~3GB Memory)
6.	Demo your network by running `mydetect.py`. Use `VGG_ILSVRC_16_layers_deploy.prototxt` as the model when demoing. Refer to `mydetect.py -h` for usage.

Training data can be graphed using the included CaffePlot.py, which has been slightly modified from [the original](https://gist.github.com/Coderx7/03f46cb24dcf4127d6fa66d08126fa3b).