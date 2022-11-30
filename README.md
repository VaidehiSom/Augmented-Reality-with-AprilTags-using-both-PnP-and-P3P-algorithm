# Augmented Reality with AprilTags using both PnP and P3P algorithm

This is an implementtaion of a simple augmented reality application. The deliverable is a video that contains several virtual object models as if they exist in the real
world.

The algorithms have been implemented instead of simply using OpenCV libraries.

The implementation is as follows-
1. est Pw.py \
This function is responsible for finding world coordinates of a, b, c and d given tag size

2. solve pnp.py \
This function is responsible for recovering R and t from 2D-3D correspondence with coplanar assumption

3. est pixel world.py \
This function is responsible for solving 3D locations of a pixel on the table.

4. solve p3p.py \
This file has two functions P3P and Procrustes. P3P solves the polynomial and calculates the distances of the 3 points from the camera and then uses the corresponding coordinates in the camera frame and the world frame.

5. VR res.gif \
This file is generated automatically by executing main.py

## Output

A gif file of a sequence that contains a set of static virtual objects with real
backgrounds, as if the virtual objects were placed in the original scene.
![Output gif](results/VR_res.gif)

## Running
To run this program:

```
python main.py
```
Visualizations have been generated using both PnP and P3P algorithm

## Debugging

The program can be run with `--debug` mode