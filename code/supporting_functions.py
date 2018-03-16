import numpy as np
import cv2
from PIL import Image
from io import BytesIO, StringIO
import base64
import time

# Define a function to convert telemetry strings to float independent of decimal convention
def convert_to_float(string_to_convert):
      if ',' in string_to_convert:
            float_value = np.float(string_to_convert.replace(',','.'))
      else: 
            float_value = np.float(string_to_convert)
      return float_value

def update_rover(Rover, data):
      # Initialize start time and sample positions
      if Rover.start_time == None:
            Rover.start_time = time.time()
            Rover.total_time = 0
            samples_xpos = np.int_([convert_to_float(pos.strip()) for pos in data["samples_x"].split(';')])
            samples_ypos = np.int_([convert_to_float(pos.strip()) for pos in data["samples_y"].split(';')])
            Rover.samples_pos = (samples_xpos, samples_ypos)
            Rover.samples_to_find = np.int(data["sample_count"])
            Rover.worldmap3[samples_ypos, samples_xpos, 0] = 255
            Rover.worldmap3[samples_ypos-1, samples_xpos-1, 0] = 255
            Rover.worldmap3[samples_ypos+1, samples_xpos+1, 0] = 255
            Rover.worldmap3[samples_ypos-1, samples_xpos+1, 0] = 255
            Rover.worldmap3[samples_ypos+1, samples_xpos-1, 0] = 255
      # Or just update elapsed time
      else:
            tot_time = time.time() - Rover.start_time
            if np.isfinite(tot_time) and not Rover.arrived:
                  Rover.total_time = tot_time
      # Print out the fields in the telemetry data dictionary
      #XXX print(data.keys())
      # The current speed of the rover in m/s
      Rover.vel = convert_to_float(data["speed"])
      # The current position of the rover
      Rover.pos = [convert_to_float(pos.strip()) for pos in data["position"].split(';')]
      # The current yaw angle of the rover
      Rover.yaw = convert_to_float(data["yaw"])
      # The current yaw angle of the rover
      Rover.pitch = convert_to_float(data["pitch"])
      # The current yaw angle of the rover
      Rover.roll = convert_to_float(data["roll"])
      # The current throttle setting
      Rover.throttle = convert_to_float(data["throttle"])
      # The current steering angle
      Rover.steer = convert_to_float(data["steering_angle"])
      # Near sample flag
      Rover.near_sample = np.int(data["near_sample"])
      # Picking up flag
      Rover.picking_up = np.int(data["picking_up"])
      # Update number of rocks collected
      Rover.samples_collected = Rover.samples_to_find - np.int(data["sample_count"])

      """ XXX
      print('speed =',Rover.vel, 'position =', Rover.pos, 'throttle =', 
      Rover.throttle, 'steer_angle =', Rover.steer, 'near_sample:', Rover.near_sample, 
      'picking_up:', data["picking_up"], 'sending pickup:', Rover.send_pickup, 
      'total time:', Rover.total_time, 'samples remaining:', data["sample_count"], 
      'samples collected:', Rover.samples_collected)
      """
      # Get the current image from the center camera of the rover
      imgString = data["image"]
      image = Image.open(BytesIO(base64.b64decode(imgString)))
      Rover.img = np.asarray(image)

      # Return updated Rover and separate image for optional saving
      return Rover, image

# Define a function to create display output given worldmap results
def create_output_images(Rover):

      # Create a scaled map for plotting and clean up obs/nav pixels a bit
      if np.max(Rover.worldmap[:,:,2]) > 0:
            nav_pix = Rover.worldmap[:,:,2] > 0
            navigable = Rover.worldmap[:,:,2] * (255 / np.mean(Rover.worldmap[nav_pix, 2]))
      else: 
            navigable = Rover.worldmap[:,:,2]

      if np.max(Rover.worldmap[:,:,0]) > 0:
            obs_pix = Rover.worldmap[:,:,0] > 0
            obstacle = Rover.worldmap[:,:,0] * (255 / np.mean(Rover.worldmap[obs_pix, 0]))
      else:
            obstacle = Rover.worldmap[:,:,0]

      likely_nav = navigable >= obstacle
      obstacle[likely_nav] = 0
      plotmap = np.zeros_like(Rover.worldmap)
      plotmap[:, :, 0] = obstacle
      plotmap[:, :, 2] = navigable
      plotmap = plotmap.clip(0, 255)
      # Overlay obstacle and navigable terrain map with ground truth map
      map_add = cv2.addWeighted(plotmap, 1, Rover.ground_truth, 0.5, 0)

      # Check whether any rock detections are present in worldmap
      rock_world_pos = Rover.worldmap[:,:,1].nonzero()
      # If there are, we'll step through the known sample positions
      # to confirm whether detections are real
      samples_located = 0
      if rock_world_pos[0].any():
            
            rock_size = 2
            for idx in range(len(Rover.samples_pos[0])):
                  test_rock_x = Rover.samples_pos[0][idx]
                  test_rock_y = Rover.samples_pos[1][idx]
                  rock_sample_dists = np.sqrt((test_rock_x - rock_world_pos[1])**2 + \
                                        (test_rock_y - rock_world_pos[0])**2)
                  # If rocks were detected within 3 meters of known sample positions
                  # consider it a success and plot the location of the known
                  # sample on the map
                  if np.min(rock_sample_dists) < 3:
                        samples_located += 1
                        map_add[test_rock_y-rock_size:test_rock_y+rock_size, 
                        test_rock_x-rock_size:test_rock_x+rock_size, :] = 255

      # Calculate some statistics on the map results
      # First get the total number of pixels in the navigable terrain map
      tot_nav_pix = np.float(len((plotmap[:,:,2].nonzero()[0])))
      # Next figure out how many of those correspond to ground truth pixels
      good_nav_pix = np.float(len(((plotmap[:,:,2] > 0) & (Rover.ground_truth[:,:,1] > 0)).nonzero()[0]))
      # Next find how many do not correspond to ground truth pixels
      bad_nav_pix = np.float(len(((plotmap[:,:,2] > 0) & (Rover.ground_truth[:,:,1] == 0)).nonzero()[0]))
      # Grab the total number of map pixels
      tot_map_pix = np.float(len((Rover.ground_truth[:,:,1].nonzero()[0])))
      # Calculate the percentage of ground truth map that has been successfully found
      perc_mapped = round(100*good_nav_pix/tot_map_pix, 1)
      # Calculate the number of good map pixel detections divided by total pixels 
      # found to be navigable terrain
      if tot_nav_pix > 0:
            fidelity = round(100*good_nav_pix/(tot_nav_pix), 1)
      else:
            fidelity = 0
      # Flip the map for plotting so that the y-axis points upward in the display
      map_add = np.flipud(map_add).astype(np.float32)
      # Add some text about map and rock sample detection results
      cv2.putText(map_add,"Time: "+str(np.round(Rover.total_time, 1))+' s', (0, 10), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add,"Mapped: "+str(perc_mapped)+'%', (0, 25), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add,"Fidelity: "+str(fidelity)+'%', (0, 40), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add,"Rocks", (0, 55), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add,"  Located: "+str(samples_located), (0, 70), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      cv2.putText(map_add,"  Collected: "+str(Rover.samples_collected), (0, 85), 
                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
      # Convert map and vision image to base64 strings for sending to server
      pil_img = Image.fromarray(map_add.astype(np.uint8))
      buff = BytesIO()
      pil_img.save(buff, format="JPEG")
      encoded_string1 = base64.b64encode(buff.getvalue()).decode("utf-8")


      vision_add = cv2.addWeighted(Rover.vision_image, 1, Rover.vision_image2, 0.5, 0)
      pil_img = Image.fromarray(vision_add.astype(np.uint8))
      buff = BytesIO()
      pil_img.save(buff, format="JPEG")
      encoded_string2 = base64.b64encode(buff.getvalue()).decode("utf-8")

      return encoded_string1, encoded_string2

def a_star_search(grid, init, goal, cost=1, heuristic=None):
    """
    grid: 1 - path, 0 - occupied
    return: 1 - path
    """
    delta = np.array(
        [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]]) # go right

    y, x = init
    g = 0
    if heuristic is not None:
        f = g + heuristic[y, x]
    else:
        f = g
    open = [[f, g, y, x]]
    
    closed = np.zeros_like(grid, dtype=np.int)
    closed[y, x] = 1
    expand = np.zeros_like(grid, dtype=np.int)
    expand[:,:] = -1
    action = np.zeros_like(grid, dtype=np.int)
    action[:,:] = -1
    policy = np.zeros_like(grid, dtype=np.int)
    
    found = False  # flag that is set when search is complete
    resign = False # flag set if we can't find expand
    count = 0
    
    while not found and not resign:
        if len(open) == 0:
            resign = True
            print('resign')
            policy[:,:] = 0
            return policy
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            f, g, y, x = next
            
            expand[y, x] = count
            count += 1

            if y == goal[0] and x == goal[1]:
                found = True
            else:
                for i in range(len(delta)):
                    y2 = y + delta[i][0]
                    x2 = x + delta[i][1]
                    if 0 <= x2 < grid.shape[1] and 0 <= y2 < grid.shape[0]:
                        if closed[y2, x2] == 0 and grid[y2, x2] == 1:
                            g2 = g + cost
                            if heuristic is not None:
                                f2 = g2 + heuristic[y2, x2]
                            else:
                                f2 = g2
                            open.append([f2, g2, y2, x2])
                            closed[y2, x2] = 1
                            # action to come here
                            action[y2, x2] = i
    y, x = goal
    policy[y, x] = 1 
    # reverse from goal
    while y != init[0] or x != init[1]:
        i = action[y, x]
        y = y - delta[i, 0]
        x = x - delta[i, 1]
        policy[y, x] = 1
         
    return policy
