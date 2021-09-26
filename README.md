# Harris_Corner_Detector_RBE595_Fusion
Access code and from github at this link: https://github.com/nastallings/Harris_Corner_Detector_RBE595_Fusion

The custom build functions for the assignment can be found in Harris_Corner_Detector.pym RANSAC.py, and match_features.py,

The crazy_video_maker.py performs the harris corner detector on a 30 fps video and outputs a 10 fps video. The final video can be found at crazy_video_output.avi

The image_stitcher.py creates a panorama of 4 scene photos (scene_1.jpeg ... scene_4.jpeg). Lines 13-26 are commented out as it takes a long time to run. I ran the code and saved the output to 4 files to prevent it from being run again. You can see the output of the harris corner detector code by running Image_Stitcher.py with those lines commented out. You can also uncomment those lines to run the code on new photos if you desire. The final output can be found in the final_output.png. There is still a bug in the feature matcher I spent multiple hours trying to figure out but could not. 




