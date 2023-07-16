#!/bin/bash
mkdir tasks;
echo "downloading pose landmark";
curl -o ./tasks/pose-landmark.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task;
echo "downloading hand landmark";
curl -o ./tasks/hand-landmark.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task;
