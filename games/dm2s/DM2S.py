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

# define global variables
gParams = {}	# parameter dictionary
gVideoWidth = 800
gVideoHeight = 800
gColors =  ["LB"]
gShapes =  ["Barred_Ring","Triangle","Crescent","Cross","Circle","Heart","Pentagon","Ring","Square"]
gButton1 = None
gButton1F = None
gButton2 = None
gButton2F = None
gCorrectFB = Rect(0, gVideoHeight - 80, gVideoWidth, 80)
gObservBar = Rect(0, 0, gVideoWidth, 80)

# ===  define procedure that runs the experiment === #
def run_experiment(argv):
    """runs the experiment."""
    # obtain parameters from a file
    with open(argv[1]) as f:
        for line in f:
            buf = line.strip().split(",")
            gParams[buf[0]] = buf[1]

    # main part
    init_pygame_and_exp()

    # exit experiment
    quit_pygame()

def init_pygame_and_exp():
    # initialize variables for button icons
    global gButton1
    global gButton1F
    global gButton2
    global gButton2F

    # initialize pygame modules
    pygame.init()

    # define screen settings
    size = (800, 800)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Delayed Matching-to-Sample")

    # load button images
    gButton1 = pygame.image.load('png/Button_Green.png')
    gButton1F = pygame.image.load('png/Button_LG.png')
    gButton2 = pygame.image.load('png/Button_Red.png')
    gButton2F = pygame.image.load('png/Button_Pink.png')

    # Observation sessions
    for i in range(0,int(gParams["observationRepeat"])):
        mainTask(True, screen)

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
    for i in range(0,int(gParams["mainTaskRepeat"])):
        mainTask(False, screen)

    # wait for 2 seconds then end the program
    pygame.time.wait(2000)

def mainTask(observation, screen):
    # draw buttons
    drawScreen(screen, observation)

    # draw the sample image
    color = gColors[random.randint(0, len(gColors))-1]
    shape = gShapes[random.randint(0, len(gColors))-1]
    sample = pygame.image.load("png/"+shape+"_"+color+".png")
    screen.blit(sample, (int(gVideoWidth/2 - sample.get_width()/2), 140))
    pygame.display.update()
    pygame.time.wait(2000)

    # redraw the screen
    drawScreen(screen, observation)
    pygame.time.wait(2000)

    # draw target images
    lr = random.randint(1, 2)  # Left:1 & Right:2
    if lr==1:
        target1 = sample
        target2 = getAnother(color,shape)
    else:
        target1 = getAnother(color,shape)
        target2 = sample

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

def drawScreen(screen,observation):
    # fill screen
    screen.fill(WHITE)
    
    if observation:
        # draw Observation Bar
        pygame.draw.rect(screen, BLUE, gObservBar)

    # draw button images
    screen.blit(gButton1, (int(gVideoWidth/2 - gButton1.get_width()/2) - 160, 610))
    screen.blit(gButton2, (int(gVideoWidth/2 - gButton2.get_width()/2) + 160, 610))

    pygame.display.update()

def getAnother(cl,sp):
    color = gColors[random.randint(0, len(gColors))-1]
    shape = gShapes[random.randint(0, len(gColors))-1]
    while cl==color and sp==shape:
        color = gColors[random.randint(0, len(gColors))-1]
        shape = gShapes[random.randint(0, len(gColors))-1]
    return pygame.image.load("png/"+shape+"_"+color+".png")

def getKeyResponse():
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

def quit_pygame():
    """exits pygame explicitly."""
    # quit program
    pygame.quit()
    # exit python
    sys.exit()

if __name__=='__main__':
    if len(sys.argv)<2:
        print("USE: python DM2S.py DM2S.par")
        exit()

    run_experiment(sys.argv)
