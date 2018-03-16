import numpy as np
import cv2
from supporting_functions import a_star_search
from perception import world_to_rover, to_polar_coords, rover_coords_to_image, rover_coords, resize_point

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!


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
        y_ret, x_ret = Rover.world_ret.nonzero()
        if len(y_ret) == 0:
            print('return: no path to', Rover.start_pos, 'from', Rover.pos)
            return Rover

        xr_rover, yr_rover = world_to_rover(x_ret, y_ret
            , Rover.pos[0], Rover.pos[1], Rover.yaw, Rover.worldmap.shape[0], 10)
        vision_ret = np.zeros_like(Rover.vision_image[:,:,2], dtype=np.uint8)
        rover_coords_to_image(xr_rover, yr_rover, vision_ret)
        vision_ret = resize_point(vision_ret, size=30, intensity=255)

        vision_ret[:,:] = ((vision_ret - Rover.vision_image[:,:,0]) > 0) * Rover.vision_mask
        yy, xx = vision_ret.nonzero()
        Rover.vision_image2[:,:,:] = 0
        Rover.vision_image2[yy, xx, 1:] = 255

        xr_rover, yr_rover = rover_coords(vision_ret)
        Rover.ret_dists, Rover.ret_angles = to_polar_coords(xr_rover, yr_rover)

        Rover.steer_bias = 0
        go_forward = 0
        if len(Rover.ret_dists) < 5:
            print('return: zero ret_dists')
            Rover.brake = 0
            Rover.throttle = 0
            Rover.steer = -15
            Rover.mode = 'forward'
            stop_forward = len(Rover.go_dirs)
        else:
            Rover.go_dirs = Rover.ret_angles
            stop_forward = len(Rover.go_dirs)
            Rover.mode = 'forward'

    # For SEARCH
    else: 
        if Rover.rock_angles is not None and len(Rover.rock_angles) != 0:
            print('search: for the rock_dir')
            Rover.throttle = 0.05
            Rover.steer_bias = 0
            Rover.go_dirs = Rover.rock_angles
            go_forward = 0
            stop_forward = len(Rover.go_dirs)
        else:
            Rover.steer_bias = 10
            Rover.go_dirs = Rover.nav_angles
            stop_forward = Rover.stop_forward
            go_forward = Rover.go_forward

    # Check if we have vision data to make decisions with
    if Rover.go_dirs is not None:
        # Check for Rover.mode status
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

                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0

                Rover.brake = 0
                Rover.steer = np.clip(np.mean(Rover.go_dirs * 180/np.pi) + Rover.steer_bias, -15, 15)
                # Set steering to average angle clipped to the range +/- 15
                print('forward: left steer_bias={}, steer={:.1f}, throttle={:.1f}, vel={:.1f}'.format(
                        Rover.steer_bias, Rover.steer, Rover.throttle, Rover.vel))

            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.go_dirs) < stop_forward:
                print('forward: will stop, for angles:', len(Rover.go_dirs), '< stop_forward:', stop_forward)
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'

        # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop':

            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                print('stop: will brake, for vel:', Rover.vel, '> 0.2')
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0

            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:

                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.go_dirs) < go_forward:
                    print('stop: turn, for short nav_dirs={} at yaw={}'.format(
                        len(Rover.go_dirs), Rover.yaw))
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 # Could be more clever here about which way to turn

                # If we're stopped but see sufficient navigable terrain in front then go!
                elif len(Rover.go_dirs) >= go_forward:
                    print('stop: now go, for large nav_dirs{} at trottle={}'.format(
                        len(Rover.go_dirs), Rover.throttle))
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.go_dirs * 180/np.pi) + Rover.steer_bias, -15, 15)
                    Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        print('Rover.go_dirs zero')
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0

    return Rover

