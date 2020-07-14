Yet another implement of Attention-Guided Hierarchical Structure Aggregation for Image Matting(HAttMatting).

Training code is done and tested. I do not know why the network cost lots of VRAM (the paper does not give some details). This project needs pytorch, opencv and some other things.
I do not use resnext101 and batchsize 4, because I do not have Highend GPU. So I use a resnet50 with GN and WS to train the network with batchsize 1. I also use a W-PatchGan instead of the orginal PatchGan Hope to have good performance.

Enjoy.
