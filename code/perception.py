import numpy as np
import cv2
import matplotlib.pyplot as plt

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float)
    return x_pixel, y_pixel

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated

# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def rover_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    return warped, mask

def find_rocks(img, levels=(110, 110, 50)):
    rockpix = (img[:,:,0] > levels[0]) & (img[:,:,1] > levels[1]) & (img[:,:,2] < levels[2])
    color_select = np.zeros_like(img[:,:,0])
    color_select[rockpix] = 1
    return color_select


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

# for display
def resize_point(img, size=1, intensity=1):
    yp, xp = img.nonzero()
    uint8_img = img.astype(np.uint8)
    for y, x in zip(yp, xp):
        pts = [[x-size,y+size], [x-size, y-size], [x+size, y-size], [x+size, y+size]]
        cv2.fillPoly(uint8_img, np.array([pts]), intensity)
    return uint8_img.astype(img.dtype)

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # DONE: 
    # NOTE: camera image is coming to you in Rover.img
    image = Rover.img
    # 1) Define source and destination points for perspective transform
    # Define calibration box in source (actual) and destination (desired) coordinates
    # These source and destination points are defined to warp the image
    # to a grid where each 10x10 pixel square represents 1 square meter
    # The destination box will be 2*dst_size on each side
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                      [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                      [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                      ])

    # 2) Apply perspective transform
    warped, Rover.vision_mask = perspect_transform(image, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navi_map = color_thresh(warped)
    obs_map = np.absolute(np.float32(navi_map) - 1) * Rover.vision_mask
    rock_map = find_rocks(image)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = obs_map * 255
    Rover.vision_image[:,:,1] = rock_map * 255
    Rover.vision_image[:,:,2] = navi_map * 255

    # 5) Convert map image pixel values to rover-centric coords
    navi_xpix, navi_ypix = rover_coords(navi_map)
    obs_xpix, obs_ypix = rover_coords(obs_map)
    rock_xpix, rock_ypix = rover_coords(rock_map)

    # 6) Convert rover-centric pixel values to world coordinates
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    navigable_x_world, navigable_y_world = rover_to_world(navi_xpix, navi_ypix, xpos, ypos, yaw, world_size, scale)
    obstacle_x_world, obstacle_y_world = rover_to_world(obs_xpix, obs_ypix, xpos, ypos, yaw, world_size, scale)
    rock_x_world, rock_y_world = rover_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    pitch = min(abs(Rover.pitch - 360), Rover.pitch)
    roll = min(abs(Rover.roll - 360), Rover.roll)
    if pitch < 0.4 and roll < 0.4:
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[rock_y_world, rock_x_world, 1] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(navi_xpix, navi_ypix)
    Rover.rock_dists, Rover.rock_angles = to_polar_coords(rock_xpix, rock_ypix)


    # worldmap3: start_pos/current_pos
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


    Rover.world_vis[int(round(ypos, 3)), int(round(xpos, 3))] = 255
    if Rover.samples_collected <= Rover.samples_to_find - 1:
        # for subtracting the visited path from the navigable terrian
        y_wvis, x_wvis = Rover.world_vis.nonzero()
        x_rvis, y_rvis = world_to_rover(x_wvis, y_wvis, xpos, ypos, yaw, world_size, scale)

        Rover.vision_visit[:,:] = 0
        rover_coords_to_image(x_rvis, y_rvis, Rover.vision_visit)
        vision_visit = resize_point(Rover.vision_visit, size=6)
        '''
        vision_visit = Rover.vision_visit.astype(np.uint8)
        yy, xx = vision_visit.nonzero()
        vis_points = np.array([p for p in zip(xx, yy)])
        hull = cv2.convexHull(vis_points)
        cv2.fillConvexPoly(vision_visit, hull, 1)
        '''
        Rover.vision_image2[:,:,:] = 0
        Rover.vision_image2[:,:,0] = vision_visit * 255 * Rover.vision_mask
        Rover.vision_image2[:,:,1] = vision_visit * 255 * Rover.vision_mask
        nav_vis_map = (np.float32(navi_map) - Rover.vision_visit) > 0
        nav_vis_xpix, nav_vis_ypix = rover_coords(nav_vis_map)
        Rover.nav_dists, Rover.nav_angles = to_polar_coords(navi_xpix, navi_ypix)
    #'''
    near_dists = (Rover.nav_dists < 30).nonzero()
    Rover.nav_dists = Rover.nav_dists[near_dists[0]]
    Rover.nav_angles = Rover.nav_angles[near_dists[0]]
    #'''


    Rover.worldmap2[:,:,0] = Rover.world_ret * 255
    Rover.worldmap2[:,:,1] = Rover.world_vis
    ypath, xpath = Rover.worldmap[:,:,2].nonzero()
    Rover.worldmap2[ypath, xpath, 2] = 255

    worldmap2 = np.flipud(Rover.worldmap2)
    worldmap3 = np.flipud(Rover.worldmap3)
    worldmap = cv2.addWeighted(worldmap2, 1, worldmap3, 1, 0)
    Rover.axisImage.set_data(worldmap.astype(np.uint8))
    plt.draw()
    plt.pause(1e-17)

    return Rover
