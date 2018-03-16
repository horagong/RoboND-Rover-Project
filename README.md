[//]: # (Image References)
[image_0]: ./misc/rover_image.jpg
[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)
# Search and Sample Return Project

## Analysis in notebook 
### obstacle and rock sample identification
In this project I used color threshold to distinguish between obstacles and navigable terrian. Because obstacles have darker color than navigable terrian, this method could be possible. 

For this I first made perspective transform. In perspect_transform() I added mask to cut off area outside angle of view.

Obstacles and navigable area are distinguished by RGB (160, 160, 160) in color_threshold(). Rock samples are identified by RGB (110, 110, 50) in find_rocks()

### process_image()
process_image() is the process pipe line. 
1. apply perspective transform
2. Apply color threshold to identify navigable terrain/obstacles/rock samples

    Obastacles are region except navigable region.
    ```
    obs_map = np.absolute(np.float32(navi_map) - 1) * mask
    ```
    navi_map is uint8. If pixel in navi_map is 0, then (pixel - 1) becomes 255. So it needs to convert to float32.
    
3. Convert thresholded image pixel values to rover-centric coords
    
4. Convert rover-centric pixel values to world coords
    
5. Update worldmap (to be displayed on right side of screen)
    ```
    data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    data.worldmap[rock_y_world, rock_x_world, 1] += 1
    data.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    ```
    The pixel intensity increases by 1 each time it's detected.
    
6. Make a mosaic image, below is some example code
    R, G, B layer of output image are filled by obstacles, rock sample, navigable terrain.
    ```
    output_image[0:img.shape[0], img.shape[1]:, 2] = navi_map * 255
    output_image[0:img.shape[0], img.shape[1]:, 1] = rock_map * 255
    output_image[0:img.shape[0], img.shape[1]:, 0] = obs_map * 255
    ```
    and then overlay worldmap with ground truth map.
    ```
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
    ```
    Worldmap needs to flip upside down to be displayed as image.
    ```
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)
    ```

7. Show the navigable terrian from worldmap into local coords.
    I wrote world_to_rover() and rover_coords_to_image() to convert the terrian from world coords into local coords.
    ```
    y_visit, x_visit = data.worldmap[:,:,2].nonzero()
    x_rover, y_rover = world_to_rover(x_visit, y_visit, xpos, ypos, yaw, world_size, scale)

    local_image[:,:,:] = 0
    rover_coords_to_image(x_rover, y_rover, local_image[:,:,0])

    vision_y, vision_x = local_image[:,:,0].nonzero()
    local_image[vision_y, vision_x, :] = 255
    ```


The result image are following.
<img src=output/test_mapping2.gif>
I 

## Autonomous Navigation and Mapping
Perception_step() in perception.py is similar with process_image() in the notebook but it do more. It have three reverse functions to convert world coords to local coords.

    ```
    # Define a reverse function of rover_coords()
    def rover_coords_to_image(x_rover, y_rover, binary_img):
        ypos = (binary_img.shape[0] - x_rover).astype(np.int)
        xpos = (binary_img.shape[1]/2 - y_rover).astype(np.int)

        ypos = np.clip(ypos, 0, binary_img.shape[0] - 1)
        xpos = np.clip(xpos, 0, binary_img.shape[1] - 1)
        binary_img[ypos, xpos] = 1
        return binary_img

    # Define a reverse function of translate_pix()
    def translate_to_rover(xpix_trans, ypix_trans, xpos, ypos, scale):
        xpix_rot = (xpix_trans - xpos) * scale
        ypix_rot = (ypix_trans - ypos) * scale
        return xpix_rot, ypix_rot

    # Define a reverse function of rover_to_world()
    def world_to_rover(x_world, y_world, x_pos, y_pos, yaw, world_size, scale):
        x_world_tran, y_world_tran = translate_to_rover(x_world, y_world, x_pos, y_pos, scale)
        x_rover, y_rover = rotate_pix(x_world_tran, y_world_tran, -yaw)
        return x_rover, y_rover

In the perception_step, it makes an image for showing start_pos/current_pos of the rover.
```
    Rover.worldmap3[:,:,1] = 0
    Rover.worldmap3[int(round(ypos, 3)), int(round(xpos, 3)), 1] = 255
    Rover.worldmap3[:,:,1] = resize_point(Rover.worldmap3[:,:,1], size=3, intensity=255)
    if Rover.start_pos is None:
       Rover.start_pos = Rover.pos
       start_ypos = int(round(Rover.start_pos[1], 3))
       start_xpos = int(round(Rover.start_pos[0], 3))
       print('start_pos', Rover.start_pos, start_ypos, start_xpos)
       Rover.worldmap3[start_ypos, start_xpos, 0] = 255
       Rover.worldmap3[start_ypos-3:start_ypos+3, start_xpos-3:start_xpos+3, 0] = 255
```

The rover sometimes runs in the loop. 
<img src=output/loop.png>
So I subtract visited paths from the navigable terrian to make the rover go the path unvisited with high probability. The visted paths are shown in gray over the blue navigalbe terrian in the left local map. It should not be applied in return time.
```
    if Rover.samples_collected <= Rover.samples_to_find - 1:
    ...
        nav_vis_map = (np.float32(navi_map) - Rover.vision_visit) > 0
        nav_vis_xpix, nav_vis_ypix = rover_coords(nav_vis_map)
        Rover.nav_dists, Rover.nav_angles = to_polar_coords(navi_xpix, navi_ypix)
    ...
```
Actually it should be ```if Rover.samples_collected <= Rover.samples_to_find```. But I realized that one of the six rocks was sometimes not visible so the rover could not find it. I think it is a bug of the simulator.

I added another worldmap which is for showing all the visited paths(the white line), the positions of the rocks(x) and whether the rock is colleted(circled x). The worldmap shows the return path which is made by A* search algorithm when the rover collected rocks and started to return.

```
    Rover.axisImage.set_data(worldmap.astype(np.uint8))
    plt.draw()
    plt.pause(1e-17)
```
<img src=output/worldmap.png>

Decision_step() in decision.py is for motion planning and control. When the rover finds the rocks, it 
picks up the rocks.
```
# For PICKING UP
if Rover.picking_up:
    rock_x = int(round(Rover.pos[0]))
    rock_y = int(round(Rover.pos[1]))
    print('picking_up:', rock_x, rock_y)
    cv2.circle(Rover.worldmap3, (rock_x, rock_y), 4, (255, 0, 255))
    return Rover
# If in a state where want to pickup a rock send pickup command
elif Rover.near_sample:
    if Rover.vel != 0:
        print('near_sample: will stop', Rover.vel)
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
    else:
        print('near_sample: will pickup')
        Rover.send_pickup = True
        Rover.mode = 'forward'
    return Rover
```
When the rover finds all of the rocks, (it's Rover.samples_to_find - 1, because of the reason said above), it starts to return by the way which is generated by A* algorithm. a_star_search() is in supporting_funcions.py.
```
# For RETURN
if Rover.samples_collected >= Rover.samples_to_find - 1:
    ret_dist = np.linalg.norm(np.int_(Rover.pos) - np.int_(Rover.start_pos))
    if ret_dist < 2:
        print('return: arrived, dist={:.1f}'.format(ret_dist))
    Rover.throttle = 0
    Rover.brake = Rover.brake_set
    Rover.arrived = True
    return Rover

init = [int(Rover.pos[1]), int(Rover.pos[0])] 
goal = [int(Rover.start_pos[1]), int(Rover.start_pos[0])]
world_vis = resize_point(Rover.world_vis, size=4, intensity=1)
grid = Rover.worldmap[:,:,2] + world_vis
Rover.world_ret = a_star_search(grid > 0, init, goal)
```
The red line is the generated return path from the current position of the rover(the green box) to the start position(the small red box).
<img src=output/return.png>

The rover sometimes get stuck in the mud or among the rocks. 
<img src=output/stuck1.png>
<img src=output/stuck2.png>
It tried to get out of it by turning -15.
```
if Rover.mode == 'forward': 
    # Check the extent of navigable terrain
    if len(Rover.go_dirs) >= stop_forward:
        if Rover.vel < 0.001:
            if Rover.prev_yaw and abs(Rover.prev_yaw - Rover.yaw) < 5:
                print('forward: stuck, yaw={}<>{}'.format(Rover.prev_yaw, Rover.yaw))
                Rover.steer = -15
                Rover.throttle = 0
                Rover.brake = 0
            else:
                print('forward: not moving, yaw=', Rover.yaw)
                Rover.steer = 0
                Rover.throttle = Rover.throttle_set
                Rover.brake = 0
                Rover.prev_yaw = Rover.yaw
            return Rover
```

The rover tries to go along the left wall not to go the visited way again. Rover.steer_bias is 10 for that case.
```
Rover.steer = np.clip(np.mean(Rover.go_dirs * 180/np.pi) + Rover.steer_bias, -15, 15)
```

When the rover go forward, if nav_angles are enough then it speeds up forward within rover's max velocity. Steering should be in between -15 and 15. Direction is selected by mean of nav_angles.

If the rover is stop mode, it should turn to find other ways and see sufficient navigable terrain in front then go.

### Autonomous mode
The rover run in autonomous mode with '1024x768 resolution' and 'fantastic' graphics quality and about 14 frames per second. 

It mapped at least 84.4% of the environment with 73.4% fidelity (accuracy) against the ground truth. It took 662.7s to collect the five rocks and come back to the start position.
<img src=output/result.png>

---

![alt text][image_0] 

This project is modeled after the [NASA sample return challenge](https://www.nasa.gov/directorates/spacetech/centennial_challenges/sample_return_robot/index.html) and it will give you first hand experience with the three essential elements of robotics, which are perception, decision making and actuation.  You will carry out this project in a simulator environment built with the Unity game engine.  

## The Simulator
The first step is to download the simulator build that's appropriate for your operating system.  Here are the links for [Linux](https://s3-us-west-1.amazonaws.com/udacity-robotics/Rover+Unity+Sims/Linux_Roversim.zip), [Mac](	https://s3-us-west-1.amazonaws.com/udacity-robotics/Rover+Unity+Sims/Mac_Roversim.zip), or [Windows](https://s3-us-west-1.amazonaws.com/udacity-robotics/Rover+Unity+Sims/Windows_Roversim.zip).  

You can test out the simulator by opening it up and choosing "Training Mode".  Use the mouse or keyboard to navigate around the environment and see how it looks.

## Dependencies
You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/ryan-keenan/RoboND-Python-Starterkit). 


Here is a great link for learning more about [Anaconda and Jupyter Notebooks](https://classroom.udacity.com/courses/ud1111)

## Recording Data

I've saved some test data for you in the folder called `test_dataset`.  In that folder you'll find a csv file with the output data for steering, throttle position etc. and the pathnames to the images recorded in each run.  I've also saved a few images in the folder called `calibration_images` to do some of the initial calibration steps with.  

The first step of this project is to record data on your own.  To do this, you should first create a new folder to store the image data in.  Then launch the simulator and choose "Training Mode" then hit "r".  Navigate to the directory you want to store data in, select it, and then drive around collecting data.  Hit "r" again to stop data collection.

## Data Analysis
Included in the IPython notebook called `Rover_Project_Test_Notebook.ipynb` are the functions from the lesson for performing the various steps of this project.  The notebook should function as is without need for modification at this point.  To see what's in the notebook and execute the code there, start the jupyter notebook server at the command line like this:

```sh
jupyter notebook
```

This command will bring up a browser window in the current directory where you can navigate to wherever `Rover_Project_Test_Notebook.ipynb` is and select it.  Run the cells in the notebook from top to bottom to see the various data analysis steps.  

The last two cells in the notebook are for running the analysis on a folder of test images to create a map of the simulator environment and write the output to a video.  These cells should run as-is and save a video called `test_mapping.mp4` to the `output` folder.  This should give you an idea of how to go about modifying the `process_image()` function to perform mapping on your data.  

## Navigating Autonomously
The file called `drive_rover.py` is what you will use to navigate the environment in autonomous mode.  This script calls functions from within `perception.py` and `decision.py`.  The functions defined in the IPython notebook are all included in`perception.py` and it's your job to fill in the function called `perception_step()` with the appropriate processing steps and update the rover map. `decision.py` includes another function called `decision_step()`, which includes an example of a conditional statement you could use to navigate autonomously.  Here you should implement other conditionals to make driving decisions based on the rover's state and the results of the `perception_step()` analysis.

`drive_rover.py` should work as is if you have all the required Python packages installed. Call it at the command line like this: 

```sh
python drive_rover.py
```  

Then launch the simulator and choose "Autonomous Mode".  The rover should drive itself now!  It doesn't drive that well yet, but it's your job to make it better!  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results!  Make a note of your simulator settings in your writeup when you submit the project.**

### Project Walkthrough
If you're struggling to get started on this project, or just want some help getting your code up to the minimum standards for a passing submission, we've recorded a walkthrough of the basic implementation for you but **spoiler alert: this [Project Walkthrough Video](https://www.youtube.com/watch?v=oJA6QHDPdQw) contains a basic solution to the project!**.


