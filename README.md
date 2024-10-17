# GoalPro-demo
The aim of this project is to give casual football league games a tool to analyse the game and players.

### Current Task: 
To create player's reel of the game where they interact with the ball

### Progress updates
- Using yolov8 to detect and track the ball and players
- player id changes when they go out of the frame
- detect player's movements with yolov8 pose estimation model

![demo](https://github.com/naomichoy/GoalPro-demo-yolov8/demo-yolov8-480p.mp4)

### Remarks: 
ball detection and pose detection has to run on different model

### TODO: 
- Re-ID for tracking same player out of frame [reference: https://github.com/mikel-brostrom/yolo_tracking]
- Fine-tune model for better ball detection from occlusions

