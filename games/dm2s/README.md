# Python code for the Dalayed Match to Sample Task
Prerequisite: pygame

Place png files in the png directory directly under the directory in which you run the program.

USE:
    
    python3 DM2S.py DM2S.par 
    
where DM2S.par contains parameters:

    mainTaskRepeat, # of actual trial sessions
    observationRepeat, # of observation sessions
    
In Observation phase + Actual trial phase:
  - It shows a sample figure for a while
  - Pause
  - It show a pair of figures, one of which matches to the sample
  - Wait for arrow key response (Left or Right) [automatic in the observation phase]

A blue bar at the top of the screen indicates the observation phase.

A yellow bar at the bottom of the screen indicates a correct answer.

After the observation phase, the screen flashes to enter the actual trial phase.

The timeout period for the arrow key response is 5 sec. 