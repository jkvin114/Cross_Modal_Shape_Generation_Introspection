ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
	detected_OS := Windows
else
	detected_OS := $(shell uname 2>/dev/null || echo Unknown)
endif

setup_demo:
ifeq ($(detected_OS),Linux)
	conda env list | grep edit3d || conda env create -f environment.yml
else
	echo "Setup is only configured for linux."
endif

edit_via_scribble:
	python edit3d/edit_via_scribble.py ./config/chair_demo.yaml --imagenum 1 --partid 1

edit_via_sketch:
	python edit3d/edit_via_sketch.py ./config/chair_demo.yaml --pretrained=data/models/airplanes_epoch_2799_iters_156800.pth --outdir output/edit_via_sketch --source_dir examples/edit_via_sketch/chairs/2e235eafe787ad029a6e43b878d5b335 --epoch 5 --trial 1 --category airplane

reconstruct_sketch:
	python edit3d/reconstruct_from_sketch.py config/chair_demo.yaml --pretrained=data/models/airplanes_epoch_2799_iters_156800.pth --outdir output/recon_sketch --impath examples/recon_sketch/chairs/ce10e4e0d04c33e7322ed2ef5fc90e25/sketch-F-2.png

reconstruct_rgb:
	python edit3d/reconstruct_from_rgb.py config/chair_demo.yaml --pretrained=None/chair_train_2023-May-25-11-39-52/checkpoints/epoch_19_iters_1214.pth --outdir output/recon_rgb --impath examples/recon_rgb/chairs/3c786ff99885e95c685d4893e4ba8951/sketch-F-2.png

train_chair:
	python edit3d/train.py config/chair_train.yaml
