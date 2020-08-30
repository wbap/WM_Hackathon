#########################################################################################
## Delayed Match to Sample Test
##   In Observation phase + Actual trial phase:
##      Show a sample figure for a while
##      Pause
##      Show a pair of figures, one of which matches to the sample
##      Wait for arrow key response (Left or Right) [automatic in the observation phase]
## Created by the Whole Brain Architecture Initiative and Riken
## under Apache 2.0 License (2020 Riken) 
#########################################################################################
import pygame
import os
import sys
import csv
import random
from pygame.locals import *

# define colors
BLACK = (0,0,0)
WHITE = (255, 255, 255)
YELLOW = (255,255,0)
BLUE = (0,0,255)
GRAY = (128,128,128)

# define global variables
gameKinds = 3	# 0:Shape, 1:Color, 2: Bar-Position
shapeKinds = 5
colorKinds = 4
barPositions = 4
transfigurationKinds = 3
screen = None
gParams = {}	# parameter dictionary
gVideoWidth = 800
gVideoHeight = 800
gColors =  ["LB","LG","YELLOW","PINK"]
gShapes =  ["Barred_Ring","Triangle","Crescent","Cross","Heart","Pentagon","Square","Circle","Ring"]
gDelay  =  ["Non-Delayed", "Delayed"]
gTransfigurations = ["None", "Reduce", "+45°"]
gBarPositions = ["Bottom","Right","Left","Top"]
gGameTypes = ["Shape","Color","Bar-Position"]
gButton1 = None
gButton1F = None
gButton2 = None
gButton2F = None
gCorrectFB = Rect(0, gVideoHeight - 100, gVideoWidth, 100)
gObservBar = Rect(0, 0, gVideoWidth, 80)

def combination_generator():
    games = []
    for i in range(0, gameKinds):
        for j in range(0, 2):	# 0:Non-Delayed vs 1:Delayed
            games.append([i,j])
    
    random.shuffle(games)
    
    game_sessions = []
    for i in games:
       if i[0]!=2:	# Not Bar-Position
           sessions = []
           for j in range(0, shapeKinds):
               for k in range(0, colorKinds):
                   for l in range(0, transfigurationKinds):
                       sessions.append([j,k,l,0])
           random.shuffle(sessions)
           for m in sessions:
              game_sessions.append(i+m)
       else:	# Bar-Position
            sessions = []
            for j in range(0, shapeKinds):
               for k in range(0, barPositions):
                sessions.append([j,0,0,k])
            random.shuffle(sessions)
            for m in sessions:
                game_sessions.append(i+m)
    return(game_sessions)

# ===  define procedure that runs the experiment === #
def run_experiment(argv):
    """runs the experiment."""
    # initialization
    init_exp(argv)
    # main part
    run_games()
    # exit experiment
    quit_pygame()

def init_exp(argv):
    # initialize variables for button icons
    global gButton1
    global gButton1F
    global gButton2
    global gButton2F
    global screen

    # initialize pygame modules
    pygame.init()

    # obtain parameters from a file
    with open(argv[1]) as f:
        for line in f:
            buf = line.strip().split(",")
            gParams[buf[0]] = buf[1]

    # define screen settings
    size = (800, 800)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Matching-to-Sample")

    # load button images
    gButton1 = pygame.image.load('png/Button_Green.png')
    gButton1F = pygame.image.load('png/Button_LG.png')
    gButton2 = pygame.image.load('png/Button_Red.png')
    gButton2F = pygame.image.load('png/Button_Pink.png')

def run_games():
    # random game setting
    games = []
    for i in range(0, gameKinds):
        for j in range(0, 2):	# 0:Non-Delayed vs 1:Delayed
            games.append([i,j])
    random.shuffle(games)
    # run games
    for game in games:
        run_a_game(game)

def run_a_game(game):
    print("Game: " + gDelay[game[1]] + "," + gGameTypes[game[0]])
    # Observation sessions
    game_sessions(True, game)

    # flash screen
    screen.fill(BLACK)
    pygame.display.update()
    pygame.time.wait(400)
    screen.fill(WHITE)
    pygame.display.update()
    pygame.time.wait(400)
    screen.fill(BLACK)
    pygame.display.update()
    pygame.time.wait(400)

    # Actual sessions
    game_sessions(False, game)

    # black out for 2 seconds to end a game
    screen.fill(BLACK)
    pygame.display.update()
    pygame.time.wait(2000)

def game_sessions(observation, game):
    count = 0
    if game[0]!=2:	# Not Bar-Position
        sessions = []
        for j in range(0, shapeKinds):
            for k in range(0, colorKinds):
                if observation or game[0]==1:	# Color
                    sessions.append([j,k,0,0])
                else:
                    for l in range(0, transfigurationKinds):
                        sessions.append([j,k,2,0])
                        # sessions.append([j,k,l,0])
        random.shuffle(sessions)
        for session in sessions:
            mainTask(observation, game, session)
            count += 1
            if observation:
                if count > int(gParams["observationRepeat"]):
                    return
            else:
                if count > int(gParams["mainTaskRepeat"]):
                    return
    else:	# Bar-Position
        sessions = []
        for k in range(0, barPositions):
            if observation:
                sessions.append([0,0,0,k])
            else:
                for j in range(0, shapeKinds):
                    sessions.append([j,0,0,k])
        random.shuffle(sessions)
        for session in sessions:
            mainTask(observation, game, session)
            count += 1
            if observation:
                if count > int(gParams["observationRepeat"]):
                    return
            else:
                if count > int(gParams["mainTaskRepeat"]):
                    return

def mainTask(observation, game, session):
    print("Session: " + gShapes[session[0]] + ", " + gColors[session[1]]  + ", " + gTransfigurations[session[2]] + ", " + gBarPositions[session[3]])
    # draw buttons
    drawScreen(observation)

    # draw the sample image
    shape = session[0]
    color = session[1]
    sample = getImage(color,shape)
    screen.blit(sample, (int(gVideoWidth/2 - sample.get_width()/2), 140))
    if game[0]==2:	# Bar-Position game
        draw_bar("Sample", gBarPositions[session[3]])
    pygame.display.update()
    pygame.time.wait(2000)

    if game[1]==1:	# Delayed task
        # redraw the screen
        drawScreen(observation)
        pygame.time.wait(2000)

    # draw target images
    lr = random.randint(1, 2)  # Left:1 & Right:2
    if game[0]==2:	# Bar-Position game
        if lr==1:
            target1 = getImage(getAnother(color,colorKinds),shape)	# another color
            target2 = getImage(color,getAnother(shape,shapeKinds))	# another shape
            draw_bar("Left", gBarPositions[session[3]])
            draw_bar("Right", gBarPositions[getAnother(session[3],4)])
        else:
            target1 = getImage(color,getAnother(shape,shapeKinds))	# another shape
            target2 = getImage(getAnother(color,colorKinds),shape)	# another color
            draw_bar("Right", gBarPositions[session[3]])
            draw_bar("Left", gBarPositions[getAnother(session[3],4)])
    else:
        if lr==1:
            if game[0]==0:	# Shape
                target1 = getImage(getAnother(color,colorKinds),shape)	# another color
                target2 = getImage(color,getAnother(shape,shapeKinds))	# another shape
            else:		# Color
                target1 = getImage(color,getAnother(shape,shapeKinds))	# another shape
                target2 = getImage(getAnother(color,colorKinds),shape)	# another color
        else:
            if game[0]==0:	# Shape
                target2 = getImage(getAnother(color,colorKinds),shape)	# another color
                target1 = getImage(color,getAnother(shape,shapeKinds))	# another shape
            else:		# Color
                target2 = getImage(color,getAnother(shape,shapeKinds))	# another shape
                target1 = getImage(getAnother(color,colorKinds),shape)	# another color
    if game[0]==0:	# Shape
        target1 = transfiguration(target1, session[2])
        target2 = transfiguration(target2, session[2])
    screen.blit(target1, (int(gVideoWidth/2 - target1.get_width()/2) - 160, 410))
    screen.blit(target2, (int(gVideoWidth/2 - target2.get_width()/2) + 160, 410))
    pygame.display.update()

    response = None
    if observation:
        pygame.time.wait(1000)
        if lr==1:
            response = pygame.K_LEFT
        else:
            response = pygame.K_RIGHT
    else:
        response = getKeyResponse()

    buttonFlashes(screen, response)
    pygame.time.wait(500)

    correct = (lr==1 and response == pygame.K_LEFT) or  (lr==2 and response == pygame.K_RIGHT)

    if correct:
        # print("lr:"+str(lr)+",key"+str(response))
        # draw Observation Bar
        pygame.draw.rect(screen, YELLOW, gCorrectFB)
        pygame.display.update()
        pygame.time.wait(1000)

    if not observation:
        if response == None:
            res = "None"
        else:
            res = pygame.key.name(response)
        print("Response:"+res+",Correct:"+str(correct))

def drawScreen(observation):
    # fill screen
    screen.fill(WHITE)
    
    if observation:
        # draw Observation Bar
        pygame.draw.rect(screen, BLUE, gObservBar)

    # draw button images
    screen.blit(gButton1, (int(gVideoWidth/2 - gButton1.get_width()/2) - 160, 610))
    screen.blit(gButton2, (int(gVideoWidth/2 - gButton2.get_width()/2) + 160, 610))

    pygame.display.update()

def getImage(color,shape):
   return pygame.image.load("png/"+gShapes[shape]+"_"+gColors[color]+".png")

def getAnother(pos,size):
    p = random.randint(0, size-1)
    while p==pos:
        p = random.randint(0, size-1)
    return p

def draw_bar(pos1, pos2):
    # pos1: "Sample", "Left", "Right"
    # pos2: "Bottom","Right","Left","Top"
    length = 150
    height = 20
    if pos1=="Sample":
        voffset = 140
        if pos2 == "Bottom":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - length/2), voffset+160, length, height))
        if pos2 == "Right":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - height/2) + 100, voffset-10, height, length))
        if pos2 == "Left":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - height/2) - 100, voffset-10, height, length))
        if pos2 == "Top":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - length/2), voffset-40, length, height))
    if pos1=="Left":
        voffset = 410
        if pos2 == "Bottom":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - length/2) - 160, voffset+160, length, height))
        if pos2 == "Right":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - height/2) + 100 - 160, voffset-10, height, length))
        if pos2 == "Left":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - height/2) - 100 - 160, voffset-10, height, length))
        if pos2 == "Top":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - length/2) - 160, voffset-40, length, height))
    if pos1=="Right":
        voffset = 410
        if pos2 == "Bottom":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - length/2) + 160, voffset+160, length, height))
        if pos2 == "Right":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - height/2) + 100 + 160, voffset-10, height, length))
        if pos2 == "Left":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - height/2) - 100 + 160, voffset-10, height, length))
        if pos2 == "Top":
            pygame.draw.rect(screen,GRAY,(int(gVideoWidth/2 - length/2) + 160, voffset-40, length, height))

def getKeyResponse():
    pygame.event.clear()
    init = pygame.time.get_ticks()
    time = pygame.time.get_ticks()
    while time - init < 5000:    # timeout
       events = pygame.event.get()
       for event in events:
           if event.type == pygame.KEYDOWN:
               return event.key
       time = pygame.time.get_ticks()

def buttonFlashes(screen, response):
    if response==pygame.K_LEFT:
        screen.blit(gButton1F, (int(gVideoWidth/2 - gButton1F.get_width()/2) - 160, 610))
        pygame.display.update()
        pygame.time.wait(400)
        screen.blit(gButton1, (int(gVideoWidth/2 - gButton1.get_width()/2) - 160, 610))
    elif response==pygame.K_RIGHT:
        screen.blit(gButton2F, (int(gVideoWidth/2 - gButton2F.get_width()/2) + 160, 610))
        pygame.display.update()
        pygame.time.wait(400)
        screen.blit(gButton2, (int(gVideoWidth/2 - gButton2.get_width()/2) + 160, 610))
    pygame.display.update()

def transfiguration(image, kind):
    reduce = 0.7
    if   kind==1:	# Reduce
        return pygame.transform.scale(image, (int(image.get_width() * reduce), int(image.get_height() * reduce)))
    elif kind==2:	# +45
        return pygame.transform.rotate(image, 45)
    else:
        return image

def quit_pygame():
    """exits pygame explicitly."""
    # quit program
    pygame.quit()
    # exit python
    sys.exit()

if __name__=='__main__':
    if len(sys.argv)<2:
        print("USE: python M2S.py M2S.par")
        exit()

    run_experiment(sys.argv)
