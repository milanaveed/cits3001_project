from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import cv2 as cv
import numpy as np
import string
import time



# code for locating objects on the screen in super mario bros
# by Lauren Gee

# Template matching is based on this tutorial:
# https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

################################################################################

# change these values if you want more/less printing
PRINT_GRID      = False
PRINT_LOCATIONS = False

# If printing the grid doesn't display in an understandable way, change the
# settings of your terminal (or anaconda prompt) to have a smaller font size,
# so that everything fits on the screen. Also, use a large terminal window /
# whole screen.

# other constants (don't change these)
SCREEN_HEIGHT   = 240
SCREEN_WIDTH    = 256
MATCH_THRESHOLD = 0.9

################################################################################
# TEMPLATES FOR LOCATING OBJECTS

# ignore sky blue colour when matching templates
MASK_COLOUR = np.array([252, 136, 104])
# (these numbers are [BLUE, GREEN, RED] because opencv uses BGR colour format by default)

# You can add more images to improve the object locator, so that it can locate
# more things. For best results, paint around the object with the exact shade of
# blue as the sky colour. (see the given images as examples)
#
# Put your image filenames in image_files below, following the same format, and
# it should work fine.

# filenames for object templates
image_files = {
    "mario": {
        "small": ["images/marioA.png", "images/marioB.png", "images/marioC.png", "images/marioD.png",
                  "images/marioE.png", "images/marioF.png", "images/marioG.png"],
        "tall": ["images/tall_marioA.png", "images/tall_marioB.png", "images/tall_marioC.png"],
        # Note: Many images are missing from tall mario, and I don't have any
        # images for fireball mario.
    },
    "enemy": {
        "goomba": ["images/goomba.png"],
        "koopa": ["images/koopaA.png", "images/koopaB.png"],
        # "fish": ["images/fishA.png", "images/fishB.png"],
        "plant": ["images/plantA.png", "images/plantB.png"],
    },
    "block": {
        "block": ["images/block1.png", "images/block2.png", "images/block3.png", "images/block4.png", "images/block5.png", "images/block6.png", "images/block7.png", "images/block8.png"],
        "question_block": ["images/questionA.png", "images/questionB.png", "images/questionC.png"],
        "pipe": ["images/pipe_upper_section.png", "images/pipe_lower_section.png"],
    },
    "item": {
        # Note: The template matcher is colourblind (it's using greyscale),
        # so it can't tell the difference between red and green mushrooms.
        "mushroom": ["images/mushroom_red.png"],
        # There are also other items in the game that I haven't included,
        # such as star.

        # There's probably a way to change the matching to work with colour,
        # but that would slow things down considerably. Also, given that the
        # red and green mushroom sprites are so similar, it might think they're
        # the same even if there is colour.
    }
}

def _get_template(filename):
    image = cv.imread(filename)
    assert image is not None, f"File {filename} does not exist."
    template = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.uint8(np.where(np.all(image == MASK_COLOUR, axis=2), 0, 1))
    num_pixels = image.shape[0]*image.shape[1]
    if num_pixels - np.sum(mask) < 10:
        mask = None # this is important for avoiding a problem where some things match everything
    dimensions = tuple(template.shape[::-1])
    # print(filename)
    # print(template)
    return template, mask, dimensions

def get_template(filenames):
    results = []
    for filename in filenames:
        results.append(_get_template(filename))
    return results

def get_template_and_flipped(filenames):
    results = []
    for filename in filenames:
        template, mask, dimensions = _get_template(filename)
        results.append((template, mask, dimensions))
        results.append((cv.flip(template, 1), cv.flip(mask, 1), dimensions))
    return results

# Mario and enemies can face both right and left, so I'll also include
# horizontally flipped versions of those templates.
include_flipped = {"mario", "enemy"}

# generate all templatees
templates = {}
for category in image_files:
    category_items = image_files[category]
    category_templates = {}
    for object_name in category_items:
        filenames = category_items[object_name]
        if category in include_flipped or object_name in include_flipped:
            category_templates[object_name] = get_template_and_flipped(filenames)
        else:
            category_templates[object_name] = get_template(filenames)
    templates[category] = category_templates

################################################################################
# PRINTING THE GRID (for debug purposes)

colour_map = {
    (104, 136, 252): " ", # sky blue colour
    (0,     0,   0): " ", # black
    (252, 252, 252): "'", # white / cloud colour
    (248,  56,   0): "M", # red / mario colour
    (228,  92,  16): "%", # brown enemy / block colour
}
unused_letters = sorted(set(string.ascii_uppercase) - set(colour_map.values()),reverse=True)
DEFAULT_LETTER = "?"

def _get_colour(colour): # colour must be 3 ints
    colour = tuple(colour)
    if colour in colour_map:
        return colour_map[colour]
    
    # if we haven't seen this colour before, pick a letter to represent it
    if unused_letters:
        letter = unused_letters.pop()
        colour_map[colour] = letter
        return letter
    else:
        return DEFAULT_LETTER

def print_grid(obs, object_locations):
    pixels = {}
    # build the outlines of located objects
    for category in object_locations:
        for location, dimensions, object_name in object_locations[category]:
            x, y = location
            width, height = dimensions
            name_str = object_name.replace("_", "-") + "-"
            for i in range(width):
                pixels[(x+i, y)] = name_str[i%len(name_str)]
                pixels[(x+i, y+height-1)] = name_str[(i+height-1)%len(name_str)]
            for i in range(1, height-1):
                pixels[(x, y+i)] = name_str[i%len(name_str)]
                pixels[(x+width-1, y+i)] = name_str[(i+width-1)%len(name_str)]

    # print the screen to terminal
    print("-"*SCREEN_WIDTH)
    for y in range(SCREEN_HEIGHT):
        line = []
        for x in range(SCREEN_WIDTH):
            coords = (x, y)
            if coords in pixels:
                # this pixel is part of an outline of an object,
                # so use that instead of the normal colour symbol
                colour = pixels[coords]
            else:
                # get the colour symbol for this colour
                colour = _get_colour(obs[y][x])
            line.append(colour)
        print("".join(line))

################################################################################
# LOCATING OBJECTS

def _locate_object(screen, templates, stop_early=False, threshold=MATCH_THRESHOLD):
    locations = {}
    for template, mask, dimensions in templates:
        results = cv.matchTemplate(screen, template, cv.TM_CCOEFF_NORMED, mask=mask)
        locs = np.where(results >= threshold)
        for y, x in zip(*locs):
            locations[(x, y)] = dimensions

        # stop early if you found mario (don't need to look for other animation frames of mario)
        if stop_early and locations:
            break
    
    #      [((x,y), (width,height))]
    return [( loc,  locations[loc]) for loc in locations]

def _locate_pipe(screen, threshold=MATCH_THRESHOLD):
    upper_template, upper_mask, upper_dimensions = templates["block"]["pipe"][0]
    lower_template, lower_mask, lower_dimensions = templates["block"]["pipe"][1]

    # find the upper part of the pipe
    upper_results = cv.matchTemplate(screen, upper_template, cv.TM_CCOEFF_NORMED, mask=upper_mask)
    upper_locs = list(zip(*np.where(upper_results >= threshold)))
    
    # stop early if there are no pipes
    if not upper_locs:
        return []
    
    # find the lower part of the pipe
    lower_results = cv.matchTemplate(screen, lower_template, cv.TM_CCOEFF_NORMED, mask=lower_mask)
    lower_locs = set(zip(*np.where(lower_results >= threshold)))

    # put the pieces together
    upper_width, upper_height = upper_dimensions
    lower_width, lower_height = lower_dimensions
    locations = []
    for y, x in upper_locs:
        for h in range(upper_height, SCREEN_HEIGHT, lower_height):
            if (y+h, x+2) not in lower_locs:
                locations.append(((x, y), (upper_width, h), "pipe"))
                break
    return locations

def locate_objects(screen, mario_status):
    # convert to greyscale
    screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)

    # iterate through our templates data structure
    object_locations = {}
    for category in templates:
        category_templates = templates[category]
        category_items = []
        stop_early = False
        for object_name in category_templates:
            # use mario_status to determine which type of mario to look for
            if category == "mario":
                if object_name != mario_status:
                    continue
                else:
                    stop_early = True
            # pipe has special logic, so skip it for now
            if object_name == "pipe":
                continue
            
            # find locations of objects
            results = _locate_object(screen, category_templates[object_name], stop_early)
            for location, dimensions in results:
                category_items.append((location, dimensions, object_name))

        object_locations[category] = category_items

    # locate pipes
    object_locations["block"] += _locate_pipe(screen)

    return object_locations

################################################################################
# GETTING INFORMATION AND CHOOSING AN ACTION

def make_action(screen, info, step, env, prev_action, counter, previous_jump, highest_x):
    mario_status = info["status"]
    object_locations = locate_objects(screen, mario_status)

    # You probably don't want to print everything I am printing when you run
    # your code, because printing slows things down, and it puts a LOT of
    # information in your terminal.

    # Printing the whole grid is slow, so I am only printing it occasionally,
    # and I'm only printing it for debug purposes, to see if I'm locating objects
    # correctly.
    if PRINT_GRID:
        print_grid(screen, object_locations)
        # If printing the grid doesn't display in an understandable way, change
        # the settings of your terminal (or anaconda prompt) to have a smaller
        # font size, so that everything fits on the screen. Also, use a large
        # terminal window / whole screen.

        # object_locations contains the locations of all the objects we found
        #print(object_locations)

    # List of locations of Mario:
    mario_locations = object_locations["mario"]
    # (There's usually 1 item in mario_locations, but there could be 0 if we
    # couldn't find Mario. There might even be more than one item in the list,
    # but if that happens they are probably approximately the same location.)

    # List of locations of enemies, such as goombas and koopas:
    enemy_locations = object_locations["enemy"]

    # List of locations of blocks, pipes, etc:
    block_locations = object_locations["block"]

    # List of locations of items: (so far, it only finds mushrooms)
    item_locations = object_locations["item"]

    # This is the format of the lists of locations:
    # ((x_coordinate, y_coordinate), (object_width, object_height), object_name)
    #
    # x_coordinate and y_coordinate are the top left corner of the object
    #
    # For example, the enemy_locations list might look like this:
    # [((161, 193), (16, 16), 'goomba'), ((175, 193), (16, 16), 'goomba')]
    
    #---------------------------------------------------------------------------
    # PRINTING LOCATIONS
    #---------------------------------------------------------------------------
    if PRINT_LOCATIONS:
        # To get the information out of a list:
        for enemy in enemy_locations:
            enemy_location, enemy_dimensions, enemy_name = enemy
            x, y = enemy_location
            width, height = enemy_dimensions
            print("enemy:", x, y, width, height, enemy_name)

        # Or you could do it this way:
        for block in block_locations:
            block_x = block[0][0]
            block_y = block[0][1]
            block_width = block[1][0]
            block_height = block[1][1]
            block_name = block[2]
            print(f"{block_name}: {(block_x, block_y)}), {(block_width, block_height)}")

        # Or you could do it this way:
        for item_location, item_dimensions, item_name in item_locations:
            x, y = item_location
            print(item_name, x, y)

        # gym-super-mario-bros also gives us some info that might be useful
        print(info)
        # see https://pypi.org/project/gym-super-mario-bros/ for explanations

        # The x and y coordinates in object_locations are screen coordinates.
        # Top left corner of screen is (0, 0), top right corner is (255, 0).
        # Here's how you can get Mario's screen coordinates:
        if mario_locations:
            location, dimensions, object_name = mario_locations[0]
            mario_x, mario_y = location
            print("Mario's location on screen:",
                  mario_x, mario_y, f"({object_name} mario)")
        
        # The x and y coordinates in info are world coordinates.
        # They tell you where Mario is in the game, not his screen position.
        mario_world_x = info["x_pos"]
        mario_world_y = info["y_pos"]
        # Also, you can get Mario's status (small, tall, fireball) from info too.
        mario_status = info["status"]
        print("Mario's location in world:",
              mario_world_x, mario_world_y, f"({mario_status} mario)")
    #---------------------------------------------------------------------------
    # METHOD
    #---------------------------------------------------------------------------
    # TODO: Write code for a strategy, such as a rule based agent.

    # Choose an action from the list of available actions.
    # For example, action = 0 means do nothing
    #              action = 1 means press 'right' button
    #              action = 2 means press 'right' and 'A' buttons at the same time
    
    #Boolean values to determine whether the agent should perform any of these types of jumps
    enemy_jump = False
    block_jump = False
    pit_jump = False
    
    #Attributes for previous_jump variable, each integer determining what type of jump Mario was previously doing
    NO_JUMP = -1
    IS_ENEMY_JUMP = 0
    IS_BLOCK_JUMP = 1
    IS_PIT_JUMP = 2
    
    #If the agent wasn't in the process of performing a jump prior to this step
    if(previous_jump == NO_JUMP):
        if mario_locations:
            location, dimensions, object_name = mario_locations[0]
            mario_x, mario_y = location
            mario_width, mario_height = dimensions
            
            #CHECKING IF WE NEED TO AVOID ENEMIES   
            for enemy in enemy_locations:
                enemy_location, enemy_dimensions, enemy_name = enemy
                enemy_x, enemy_y = enemy_location
                enemy_width, enemy_height = enemy_dimensions
                #if the enemy is on the same y-axis level and about within one block away from Mario, the agent will perform an enemy_jump
                if enemy_x <= mario_x + mario_width + 16 and enemy_x - mario_x > -1 and enemy_y + enemy_height == mario_y + mario_height:
                    enemy_jump = True
                    previous_jump = IS_ENEMY_JUMP
                    break
                    
            #CHECKING IF WE NEED TO JUMP OVER A BLOCK
            for block in block_locations:
                block_location, block_dimensions, block_name = block
                block_x, block_y = block_location
                block_width, block_height = block_dimensions
                
                #Will perform block_jump if the block is within 50 pixels ahead of Mario and if the block is obstructing Mario, path
                if block_y <= mario_y and block_y + block_height + 1 == mario_y + mario_height and block_x - mario_x < 50  and block_x - mario_x > -1:
                    #If the block is a pipe, the agent will choose to perform a block_jump only if there is no Piranha Plant on the pipe
                    if block_name == 'pipe':
                        should_break = False
                        for enemy in enemy_locations:
                            enemy_location, enemy_dimensions, enemy_name = enemy
                            enemy_x, enemy_y = enemy_location
                            enemy_width, enemy_height = enemy_dimensions
                            
                            if(enemy_name == 'plant' and block_x + 8 == enemy_x):
                                should_break = True
                                break
                        
                        if(should_break):
                            break       
                    block_jump = True
                    previous_jump = IS_BLOCK_JUMP
                    break
                
            #CHECKING FOR PIT JUMP
            #We should find which block mario is on
            #We should then check if there is a block next to that
            
            #Determines the y coordinate of the block which Mario could be standing on
            mario_block_y = mario_y + mario_height - 1
            
            #Determines the x coordinate of the block which Mario could be standing on
            mario_block_x = 16*(mario_x//16)
            
            block_below_mario = False
            
            #Checks if there is a block below Mario
            for block in block_locations:
                block_location, block_dimensions, block_name = block
                block_x, block_y = block_location
                block_width, block_height = block_dimensions
                
                if block_x == mario_block_x and block_y == mario_block_y:
                    block_below_mario = True
                    break
            
            #Set to True, will be changed to False if no block is found at the pit_block coordinates (where the pit should be)
            pit_next_to_mario = True
            pit_block_y = mario_block_y
            pit_block_x = mario_block_x + 32
            
            #If Mario is standing on a block, the Agent will check if there is a block 2 blocks ahead of the one he is standing on.
            #If there is no block two blocks ahead, it means there is a pit (empty block), so the Agent will choose to jump over it.
            if block_below_mario:
                for block in block_locations:
                    block_location, block_dimensions, block_name = block
                    block_x, block_y = block_location
                    block_width, block_height = block_dimensions
                  
                    if (block_x == pit_block_x and block_y == pit_block_y):
                        pit_next_to_mario = False
                        break
                        
                if pit_next_to_mario:
                    pit_jump = True
                    previous_jump = IS_PIT_JUMP
            
    #Used this to find the furthest distance the agent can reach in a level
    mario_world_x = info["x_pos"]
    if mario_world_x > highest_x:
        highest_x = mario_world_x
    
    #If the agent is performing an enemy_jump, it will only do so for 8 steps as it only needs to be a brief jump to avoid an enemy ahead.
    #If the agent is performing a block_jump or a pit_jump, it will do one for 17 steps, as both of these types of jumps require a full jump and 17 steps of holding the jump button is the closest we found to being a full jump.
    if ((enemy_jump and counter <= 8) or (prev_action == 4 and counter <= 8) or (prev_action == 0 and counter <= 8)) and (previous_jump == 0):
        if(counter == 0):
            print("Enemy Jump 0")
            action = 4
            counter += 1
            previous_jump = IS_ENEMY_JUMP
            return action, counter, previous_jump, highest_x
        elif(counter == 8):
            print("Enemy Jump", counter)
            counter += 1
            action = 3
            previous_jump = NO_JUMP
            return action, counter, previous_jump, highest_x
        else:
            print("Enemy Jump", counter)
            action = 4
            counter += 1
            return action, counter, previous_jump, highest_x
    elif ((block_jump and counter <= 17) or (prev_action == 4 and counter <= 17) or (prev_action == 0 and counter <= 17)) and (previous_jump == 1):
        if(counter == 0):
            print("Block Jump 0")
            action = 4
            counter += 1
            previous_jump = IS_BLOCK_JUMP
            return action, counter, previous_jump, highest_x
        elif(counter == 17):
            print("Block Jump", counter)
            counter += 1
            action = 3
            previous_jump = NO_JUMP
            return action, counter, previous_jump, highest_x
        else:
            print("Block Jump", counter)
            action = 4
            counter += 1
            return action, counter, previous_jump, highest_x
    elif (((pit_jump and counter <= 17) or (prev_action == 4 and counter <= 17) or (prev_action == 0 and counter <= 17)) and (previous_jump == 2)):
        if(counter == 0):
            print("Pit Jump 0")
            action = 4
            counter += 1
            previous_jump = IS_PIT_JUMP
            return action, counter, previous_jump, highest_x
        elif(counter == 17):
            print("Pit Jump", counter)
            counter += 1
            action = 3
            previous_jump = NO_JUMP
            return action, counter, previous_jump, highest_x 
        else:
            print("Pit Jump", counter)
            action = 4
            counter += 1
            return action, counter, previous_jump, highest_x
    else:
        action = 3
        counter = 0
        previous_jump = NO_JUMP
        return action, counter, previous_jump, highest_x

################################################################################
env = gym.make("SuperMarioBros-v0", apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs = None
done = True
env.reset()
counter = 0
previous_jump = -1
highest_x = 0
for step in range(100000):
    if obs is not None:
        action, counter, previous_jump, highest_x = make_action(obs, info, step, env, action, counter, previous_jump, highest_x)
        #print(action)
    else:
        action = 3
    obs, reward, terminated, truncated, info = env.step(action)
    #print(action)
    #counter += 1
    done = terminated or truncated
    if done:
        env.reset()
env.close()