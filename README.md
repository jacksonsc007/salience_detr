Branch: reppoints

The concept of sampling locations of deformable attention share the same spirit with reppoints. Experiments are conducted to validate if the idea of reppoints could be embedded into the deformable attention process of decoder for detrs.

# v1.0
Following <RepPoints>, we group sampling locations to form pseu-boxes on whice box losses are imposed. 

# v1.1
- [x] pay attention to the initialization of box embed
- [x] get rid of extra normalization in the decoder
- [x] disable look forward twice


> TODO: 1. iterative refinement of reppoints across different decoder layers.  2. class supervison on reppoints.

# v1.2
In this version, we experiment with the likelihood of refining reppoints across different decoder layers.

> TODO
- [] sampling locations could be out of box bounds
- [] reppoints from multiple feature maps