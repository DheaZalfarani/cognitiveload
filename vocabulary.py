#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on January 21, 2025, at 13:24
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
import time
import random
from random import random as random_number_for_boolean_check
import pandas as pd
import datetime

# import pygame
import playsound

# Standard PsychoPy Imports
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '0'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import randint, normal, shuffle, choice as randchoice     #random, 
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

from unidecode import unidecode
import string

import logging_utilities
from questionnaires import likertscale, panas, nasa, affective_slider
from eye_closing import eye_closing
from wait_some_time import wait_some_time

from gtts import gTTS

# --- Setup global variables (available in all functions) ---
# TODO: Adjust language preferences and known languages!
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'vocabulary'  # from the Builder filename that created this script

# TODO: When running new experiments, adjust the expInfo metadata as well as the storage path of language dicts as well as the participants language preferences.

# information about this experiment
expInfo = {
    'Pseudonymised ID': 'AHK9038',
    'Session': '002',
    'Language Preferences|hid': 'Spanish, Turkish, Esperanto, Nahuatl, Pinjin, Hinglish',
    'Known Languages|hid': 'Esperanto, Pinjin',
    'starting_from|hid': 2,
    'pre_randomized_order|hid': 'UNDERLOAD, LOW INTEREST, OVERLOAD, HIGH INTEREST',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}
THIS_PARTICIPANT_LANGUAGE_PREFERENCES = ['Spanish, Turkish, Hinglish, Nahuatl']
LANGUAGE_PREFERENCES_DICTS = {
    'Esperanto': 'C:\\Users\\studi\\Desktop\\E-Learning\\materials\\languages_csvs\\esperanto_english_680_words_translated.csv',
    'Hinglish': 'C:\\Users\\studi\\Desktop\\E-Learning\\materials\\languages_csvs\\hinglish_english_602_words_translated.csv',
    'Nahuatl': 'C:\\Users\\studi\\Desktop\\E-Learning\\materials\\languages_csvs\\nahuatl_english_557_words_translated.csv',
    'Pinjin': 'C:\\Users\\studi\\Desktop\\E-Learning\\materials\\languages_csvs\\pinjin_english_1000_words_translated.csv',
    'Spanish': 'C:\\Users\\studi\\Desktop\\E-Learning\\materials\\languages_csvs\\spanish_english_986_words_translated.csv',
    'Turkish': 'C:\\Users\\studi\\Desktop\\E-Learning\\materials\\languages_csvs\\turkish_english_1104_words_translated.csv'
}
TASK_TO_LANGUAGE_DICT = {
    'LOW INTEREST': -1,
    'HIGH INTEREST': 0,
    'UNDERLOAD': -2,
    'OVERLOAD': 1,
}
VOCABULARIES_AMOUNT_DICT = {
    'LOW INTEREST': 20,
    'HIGH INTEREST': 20,
    'UNDERLOAD': 10,
    'OVERLOAD': 100,
}
VOCABULARIES_DURATION_DICT = {
    'LOW INTEREST': 10,
    'HIGH INTEREST': 10,
    'UNDERLOAD': 10,
    'OVERLOAD': 6,
}
VOCABULARIES_REPETITION_DICT = {
    'LOW INTEREST': 6,
    'HIGH INTEREST': 6,
    'UNDERLOAD': 12,
    'OVERLOAD': 2,
}
TRANSLATOR = str.maketrans('', '', string.punctuation)
letter_height_gobal = 0.035
base_path = 'C:\\Users\\studi\\Desktop\\E-Learning\\materials\\'
TASKS_ORDER = None

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1920, 1080)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']


def questionnaires_after_task(continueRoutine, thisExp, routineTimer, win, eye_closing_timer, frameTolerance, endExpNow, defaultKeyboard, questionnaire_number):
    wait_some_time(continueRoutine, thisExp, routineTimer, win, frameTolerance, endExpNow, defaultKeyboard, time_in_ms=500)

    ### PANAS ###
    panas(continueRoutine, thisExp, routineTimer, win, frameTolerance, endExpNow, defaultKeyboard, questionnaire_number)
    wait_some_time(continueRoutine, thisExp, routineTimer, win, frameTolerance, endExpNow, defaultKeyboard, time_in_ms=500)

    ###   NASA  ###
    nasa(continueRoutine, thisExp, routineTimer, win, endExpNow, defaultKeyboard, frameTolerance, questionnaire_number)

    ### AFFECTIVE SLIDER ###
    affective_slider(base_path, thisExp, routineTimer, win, frameTolerance, endExpNow, defaultKeyboard, questionnaire_number)
    wait_some_time(continueRoutine, thisExp, routineTimer, win, frameTolerance, endExpNow, defaultKeyboard, time_in_ms=500)

    ###   LIKERT SCALE  ###
    likertscale(experiment_Ref=thisExp, win_ref=win, for_which='mental effort', enumerated_id=questionnaire_number)
    likertscale(experiment_Ref=thisExp, win_ref=win, for_which='stress', enumerated_id=questionnaire_number)


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['Pseudonymised ID'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\studi\\Desktop\\E-Learning\\vocabulary.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # save a log file for detail verbose info
    logging_utilities.logger_setup(filename)
    
    # Initialize the script aspects needed for the experiment
    TASKS_ORDER = ['LOW INTEREST', 'HIGH INTEREST', 'UNDERLOAD', 'OVERLOAD']
    random.shuffle(TASKS_ORDER)
    
    task_order_predefined = expInfo['pre_randomized_order'] != None
    start_of_experiment = expInfo['starting_from']
    
    if task_order_predefined:
        print('This run will start with a pre-defined TASK_ORDER from task_position %d...' % start_of_experiment)
    else:
        print('This run will start with a new TASK_ORDER from task_position %d...' % start_of_experiment)
    
    LANGUAGES_NOT_TO_CONSIDER_FOR_THIS_PARTICIPANT = [lang.strip() for lang in expInfo['Known Languages'].split(',')]
    THIS_PARTICIPANT_LANGUAGE_PREFERENCES = [lang.strip() for lang in expInfo['Language Preferences'].split(',')]
    
    # Remove the languages from the list for the test if the participant knows the language
    for lang in LANGUAGES_NOT_TO_CONSIDER_FOR_THIS_PARTICIPANT:
        if lang in THIS_PARTICIPANT_LANGUAGE_PREFERENCES:
            THIS_PARTICIPANT_LANGUAGE_PREFERENCES.remove(lang)
    
    TASKS_ORDER = TASKS_ORDER if not task_order_predefined else [task_order.strip() for task_order in expInfo['pre_randomized_order'].split(',')]
    
    print('Task Order is: %s, hence we will start with the task %s and the respective language %s.' % (TASKS_ORDER, TASKS_ORDER[start_of_experiment], THIS_PARTICIPANT_LANGUAGE_PREFERENCES[TASK_TO_LANGUAGE_DICT[TASKS_ORDER[start_of_experiment]]]))
    print('The participants language preferences are: %s, already considering the languages known which were %s and were removed...' % (THIS_PARTICIPANT_LANGUAGE_PREFERENCES, LANGUAGES_NOT_TO_CONSIDER_FOR_THIS_PARTICIPANT))
        
    # return experiment handler and necessary data for the experiment
    return thisExp, TASKS_ORDER, THIS_PARTICIPANT_LANGUAGE_PREFERENCES, start_of_experiment


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('spacebar_to_continue_to_vocabulary_learning') is None:
        # initialise spacebar_to_continue_to_vocabulary_learning
        spacebar_to_continue_to_vocabulary_learning = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='spacebar_to_continue_to_vocabulary_learning',
        )
    if deviceManager.getDevice('spacebar_to_continue_vocabulary_learning_after_test_instructions') is None:
        # initialise spacebar_to_continue_vocabulary_learning_after_test_instructions
        spacebar_to_continue_vocabulary_learning_after_test_instructions = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='spacebar_to_continue_vocabulary_learning_after_test_instructions',
        )
    if deviceManager.getDevice('vocabulary_end_after_results_reveal') is None:
        # initialise vocabulary_end_after_results_reveal
        vocabulary_end_after_results_reveal = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='vocabulary_end_after_results_reveal',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        keys = defaultKeyboard.getKeys(clear=False)
        if len(keys) > 1:
            if keys[0].name == "escape" and keys[1].name == "l" or keys[0].name == "l" and keys[1].name == "escape":
                thisExp.status = FINISHED
                endExperiment(thisExp, win=win)
            else:
                defaultKeyboard.clearEvents()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run_based_on_time(expInfo, thisExp, TASKS_ORDER, THIS_PARTICIPANT_LANGUAGE_PREFERENCES, start_of_experiment, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # Adjustment marker
    do_adjustments_flag = True # If we want to adjust the code during run or not!
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    # Eye-Closing, then the questionnaires for the first time!
    eye_closing_timer = 60 #60 # 5 # 60.0
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    continueRoutine = True
    # eye_closing(thisExp, routineTimer, win, eye_closing_timer, frameTolerance, endExpNow, defaultKeyboard, base_path)
    questionnaires_after_task(continueRoutine, thisExp, routineTimer, win, eye_closing_timer, frameTolerance, endExpNow, defaultKeyboard, 0)

    # Leaving this here... Example code to reuse the text to speech feature for adjustment
    #-#-#from gtts import gTTS
    # The text that you want to convert to audio
    #-#-#mytext = 'Welcome to geeksforgeeks!'
    
    # Language in which you want to convert
    #-#-#language = 'en'

    # Passing the text and language to the engine, 
    # here we have marked slow=False. Which tells 
    # the module that the converted audio should 
    # have a high speed
    #-#-#myobj = gTTS(text=mytext, lang=language, slow=False)
    #-#-#myobj.save("test_word.mp3")
    #-# sound_test = sound.backend_ptb.SoundPTB('test_word.mp3', stereo=True, hamming=True, name='testme')
    #-# sound_test.play()
    # Initialize the mixer module
    ## pygame.mixer.init()

    # Load the mp3 file
    ## pygame.mixer.music.load("test_word.mp3")
    ## os.system("start test_word.mp3")
    # playsound.playsound('test_word.mp3', False)
    # eye_closing(thisExp, routineTimer, win, eye_closing_timer, frameTolerance, endExpNow, defaultKeyboard, base_path)
    # Play the loaded mp3 file
    ## pygame.mixer.music.play()
    # exit(0)

    # iterate over each task
    for task_idx, task in enumerate(TASKS_ORDER):
        if (start_of_experiment is not None) and (task_idx < start_of_experiment): 
            continue
        logging_utilities.log_info('WILL NOW START THE TASK %s' % task)
        if do_adjustments_flag:
            # Reset the status_flag file to start, i.e. no adjustment necessary
            adjustment_flag_file = 'C:\\Users\\studi\\Desktop\\E-Learning\\materials\\adjustment_flag.txt'
            if os.path.exists(adjustment_flag_file):
                with open(adjustment_flag_file, 'w') as filetowrite:
                    filetowrite.write('0')  # Code 0 == No adjustment needed!; Code 1 == Speedup!; Code -1 == Slow Down!
                logging_utilities.log_status_change('Vocabulary_Presenter', 'resetted the adjustment_flag.txt file to code zero, i.e. no adjustment necessary')
        else:
            logging_utilities.log_status_change('Vocabulary_Presenter', 'Going to ignore the adjustment_flag.txt file, as this run is without adjustments!')
        amount_repetitions_to_do = VOCABULARIES_REPETITION_DICT[task]
        vocabularies_amount = VOCABULARIES_AMOUNT_DICT[task]
        vocabularies_duration = VOCABULARIES_DURATION_DICT[task] * vocabularies_amount  # in seconds for all the vocabularies
        aspired_task_duration_in_seconds = vocabularies_duration
        csv_path = LANGUAGE_PREFERENCES_DICTS[THIS_PARTICIPANT_LANGUAGE_PREFERENCES[TASK_TO_LANGUAGE_DICT[task]]]
        TO_LANG = csv_path.split('\\')[-1].split('_')[0].capitalize()
        FROM_LANG = 'English'
        word_presented_last = None
        
        # --- Initialize components for Routine "start_vocabulary" ---
        welcome_text = visual.TextStim(win=win, name='welcome_text',
            text="Next, you'll start a %s vocabulary learning task [%s -- %s].\n\nYou'll be presented %d words, each presented for %d times for %d seconds.\n\nYou will have to memorize these words.\n\nYou can ignore special characters like í or á and any punctuation signs like / or . or ¡\n\nAfterwards, you'll be asked to respond with the vocabularies you learned.\n\nTherefore, you'll have to type the respective words in the respective languages.\n\nThe application will continue automatically. There is no proceed/stop button.\n\nTo continue, please press [SPACE]." % (task, FROM_LANG, TO_LANG, vocabularies_amount, amount_repetitions_to_do, VOCABULARIES_DURATION_DICT[task]),
            font='Arial',
            pos=(0, 0), draggable=False, height=letter_height_gobal, wrapWidth=None, ori=0.0, 
            color='white', colorSpace='rgb', opacity=None, 
            languageStyle='LTR',
            depth=0.0);
        spacebar_to_continue_to_vocabulary_learning = keyboard.Keyboard(deviceName='spacebar_to_continue_to_vocabulary_learning')
        
        # --- Initialize components for Routine "vocabulary_presenter" ---
        vocabulary_learner = visual.TextStim(win=win, name='vocabulary_learner',
            text='WORD (FROM-LANGUAGE)\n\n\n\n\nTRANSLATION (TO-LANGUAGE)',
            font='Arial',
            pos=(0, 0), draggable=False, height=letter_height_gobal, wrapWidth=None, ori=0.0, 
            color='white', colorSpace='rgb', opacity=None, 
            languageStyle='LTR',
            depth=0.0);
        
        # --- Initialize components for Routine "vocabulary_instructions_for_task" ---
        text = visual.TextStim(win=win, name='text',
            text="Next, the actual vocabulary test will be performed.\n\nYou'll see a word from the language you just learned.\n\nYou'll have to type the translation of this word into the respective textbox.\n\nYou can ignore special characters like í or á and any punctuation signs like / or . or ¡\n\nThere is no submit button, and the test continues automatically after a predefined time.\n\nYou'll have %d seconds per word.\n\nTo continue, please press [SPACE].",
            font='Arial',
            pos=(0, 0), draggable=False, height=letter_height_gobal, wrapWidth=None, ori=0.0, 
            color='white', colorSpace='rgb', opacity=None, 
            languageStyle='LTR',
            depth=0.0);
        spacebar_to_continue_vocabulary_learning_after_test_instructions = keyboard.Keyboard(deviceName='spacebar_to_continue_vocabulary_learning_after_test_instructions')
        
        # --- Initialize components for Routine "vocabulary_tester" ---
        text_2 = visual.TextStim(win=win, name='text_2',
            text='Please translate WORD (FROM-LANGUAGE) TO (TO-LANGUAGE)',
            font='Arial',
            pos=(0, 0), draggable=False, height=letter_height_gobal, wrapWidth=None, ori=0.0, 
            color='white', colorSpace='rgb', opacity=None, 
            languageStyle='LTR',
            depth=0.0);
        textbox = visual.TextBox2(
             win, text='', placeholder='', font='Arial',
             ori=0.0, pos=(0.0, -0.25), draggable=False,      letterHeight=letter_height_gobal,
             size=(0.5, 0.5), borderWidth=2.0,
             color='white', colorSpace='rgb',
             opacity=None,
             bold=False, italic=False,
             lineSpacing=1.0, speechPoint=None,
             padding=0.0, alignment='center',
             anchor='center', overflow='visible',
             fillColor=None, borderColor=None,
             flipHoriz=False, flipVert=False, languageStyle='LTR',
             editable=True,
             name='textbox',
             depth=-1, autoLog=True,
        )
        
        # --- Initialize components for Routine "end_vocabulary" ---
        text_3 = visual.TextStim(win=win, name='text_3',
            text='This concludes the vocabulary test.\n\nYour correct response rate was: %s\n\nTo continue, please press [SPACE].',
            font='Arial',
            pos=(0, 0), draggable=False, height=letter_height_gobal, wrapWidth=None, ori=0.0, 
            color='white', colorSpace='rgb', opacity=None, 
            languageStyle='LTR',
            depth=0.0);
        vocabulary_end_after_results_reveal = keyboard.Keyboard(deviceName='vocabulary_end_after_results_reveal')
        
        # create some handy timers
        
        # global clock to track the time since experiment started
        if globalClock is None:
            # create a clock if not given one
            globalClock = core.Clock()
        if isinstance(globalClock, str):
            # if given a string, make a clock accoridng to it
            if globalClock == 'float':
                # get timestamps as a simple value
                globalClock = core.Clock(format='float')
            elif globalClock == 'iso':
                # get timestamps in ISO format
                globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
            else:
                # get timestamps in a custom format
                globalClock = core.Clock(format=globalClock)
        if ioServer is not None:
            ioServer.syncClock(globalClock)
        logging.setDefaultClock(globalClock)
        # routine timer to track time remaining of each (possibly non-slip) routine
        routineTimer = core.Clock()
        win.flip()  # flip window to reset last flip timer
        # store the exact time the global clock started
        expInfo['expStart'] = data.getDateStr(
            format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
        )
        
        # --- Prepare to start Routine "start_vocabulary" ---
        # create an object to store info about Routine start_vocabulary
        start_vocabulary = data.Routine(
            name='start_vocabulary',
            components=[welcome_text, spacebar_to_continue_to_vocabulary_learning],
        )
        start_vocabulary.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for spacebar_to_continue_to_vocabulary_learning
        spacebar_to_continue_to_vocabulary_learning.keys = []
        spacebar_to_continue_to_vocabulary_learning.rt = []
        _spacebar_to_continue_to_vocabulary_learning_allKeys = []
        # store start times for start_vocabulary
        start_vocabulary.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        start_vocabulary.tStart = globalClock.getTime(format='float')
        start_vocabulary.status = STARTED
        thisExp.addData('start_vocabulary.started', start_vocabulary.tStart)
        start_vocabulary.maxDuration = None
        # keep track of which components have finished
        start_vocabularyComponents = start_vocabulary.components
        for thisComponent in start_vocabulary.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "start_vocabulary" ---
        start_vocabulary.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *welcome_text* updates
            
            # if welcome_text is starting this frame...
            if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                welcome_text.frameNStart = frameN  # exact frame index
                welcome_text.tStart = t  # local t and not account for scr refresh
                welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'welcome_text.started')
                # update status
                welcome_text.status = STARTED
                welcome_text.setAutoDraw(True)
            
            # if welcome_text is active this frame...
            if welcome_text.status == STARTED:
                # update params
                pass
            
            # *spacebar_to_continue_to_vocabulary_learning* updates
            waitOnFlip = False
            
            # if spacebar_to_continue_to_vocabulary_learning is starting this frame...
            if spacebar_to_continue_to_vocabulary_learning.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                spacebar_to_continue_to_vocabulary_learning.frameNStart = frameN  # exact frame index
                spacebar_to_continue_to_vocabulary_learning.tStart = t  # local t and not account for scr refresh
                spacebar_to_continue_to_vocabulary_learning.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(spacebar_to_continue_to_vocabulary_learning, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'spacebar_to_continue_to_vocabulary_learning.started')
                # update status
                spacebar_to_continue_to_vocabulary_learning.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(spacebar_to_continue_to_vocabulary_learning.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(spacebar_to_continue_to_vocabulary_learning.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if spacebar_to_continue_to_vocabulary_learning.status == STARTED and not waitOnFlip:
                theseKeys = spacebar_to_continue_to_vocabulary_learning.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _spacebar_to_continue_to_vocabulary_learning_allKeys.extend(theseKeys)
                if len(_spacebar_to_continue_to_vocabulary_learning_allKeys):
                    spacebar_to_continue_to_vocabulary_learning.keys = _spacebar_to_continue_to_vocabulary_learning_allKeys[-1].name  # just the last key pressed
                    spacebar_to_continue_to_vocabulary_learning.rt = _spacebar_to_continue_to_vocabulary_learning_allKeys[-1].rt
                    spacebar_to_continue_to_vocabulary_learning.duration = _spacebar_to_continue_to_vocabulary_learning_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            # Each Frame
            keys = defaultKeyboard.getKeys(clear=False)
            if len(keys) > 1:
                if keys[0].name == "escape" and keys[1].name == "l" or keys[0].name == "l" and keys[1].name == "escape":
                    thisExp.status = FINISHED
                else:
                    defaultKeyboard.clearEvents()
            
            #if defaultKeyboard.getKeys(keyList=["escape"]):
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                start_vocabulary.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in start_vocabulary.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "start_vocabulary" ---
        for thisComponent in start_vocabulary.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for start_vocabulary
        start_vocabulary.tStop = globalClock.getTime(format='float')
        start_vocabulary.tStopRefresh = tThisFlipGlobal
        thisExp.addData('start_vocabulary.stopped', start_vocabulary.tStop)
        # check responses
        if spacebar_to_continue_to_vocabulary_learning.keys in ['', [], None]:  # No response was made
            spacebar_to_continue_to_vocabulary_learning.keys = None
        thisExp.addData('spacebar_to_continue_to_vocabulary_learning.keys',spacebar_to_continue_to_vocabulary_learning.keys)
        if spacebar_to_continue_to_vocabulary_learning.keys != None:  # we had a response
            thisExp.addData('spacebar_to_continue_to_vocabulary_learning.rt', spacebar_to_continue_to_vocabulary_learning.rt)
            thisExp.addData('spacebar_to_continue_to_vocabulary_learning.duration', spacebar_to_continue_to_vocabulary_learning.duration)
        thisExp.nextEntry()
        # the Routine "start_vocabulary" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "vocabulary_presenter" ---
        start_time_this_task = datetime.datetime.now()
        time_last_adjustment_code_executed = datetime.datetime.now() - datetime.timedelta(minutes=2)
        aspired_end_time_this_task = start_time_this_task + datetime.timedelta(seconds=((amount_repetitions_to_do * aspired_task_duration_in_seconds) + VOCABULARIES_DURATION_DICT[task]))
        logging_utilities.log_status_change('Vocabulary Presenter', 'Is this aspired_end_time (%s) bigger than the time now (%s)? %s' % (aspired_end_time_this_task, start_time_this_task, start_time_this_task < aspired_end_time_this_task))
        vocabularies_pd = pd.read_csv(csv_path)
        duration_per_vocabulary = vocabularies_duration / vocabularies_amount
        time_per_vocabulary_for_repetition = duration_per_vocabulary
        text.text = text.text % int(time_per_vocabulary_for_repetition)
        words_repetition_counter_dictionary_english_to_repetition_ctr = {}
        translations_dictionary_from_english_to_lang = {}
        words_not_yet_presented_set = set()
        words_presented_set = set()

        ten_random_vocabularies_idxs = random.sample(range(len(vocabularies_pd)), vocabularies_amount)
        for word_idx in ten_random_vocabularies_idxs:
            translation_word, english_word = vocabularies_pd.iloc[word_idx]
            translations_dictionary_from_english_to_lang[english_word] = translation_word
            words_repetition_counter_dictionary_english_to_repetition_ctr[english_word] = 0
            words_not_yet_presented_set.add(english_word)
        
        logging_utilities.log_info('Vocabulary Presenter Setup', 'We will try to present all these words: %s for a repetition of each %d in a total time of approximately %s' % (words_not_yet_presented_set, amount_repetitions_to_do, aspired_end_time_this_task - start_time_this_task))
        
        myVocabularyPresenterClock = None

        # create an object to store info about Routine vocabulary_presenter
        vocabulary_presenter = data.Routine(
            name='vocabulary_presenter',
            components=[vocabulary_learner],
        )
        vocabulary_presenter.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for vocabulary_presenter
        vocabulary_presenter.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        vocabulary_presenter.tStart = globalClock.getTime(format='float')
        vocabulary_presenter.status = STARTED
        thisExp.addData('vocabulary_presenter.started', vocabulary_presenter.tStart)
        vocabulary_presenter.maxDuration = None
        # keep track of which components have finished
        vocabulary_presenterComponents = vocabulary_presenter.components
        for thisComponent in vocabulary_presenter.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "vocabulary_presenter" ---
        changes_made_slow_down = 0
        changes_made_speed_up = 0
        vocabulary_presenter.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and datetime.datetime.now() < aspired_end_time_this_task:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *vocabulary_learner* updates
            
            # if vocabulary_learner is starting this frame...
            if vocabulary_learner.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                vocabulary_learner.frameNStart = frameN  # exact frame index
                vocabulary_learner.tStart = t  # local t and not account for scr refresh
                vocabulary_learner.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(vocabulary_learner, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'vocabulary_learner.started')

                # Start with the first word!
                english_word = words_not_yet_presented_set.pop()
                spanish_word = translations_dictionary_from_english_to_lang[english_word]
                if do_adjustments_flag:
                    mytext = '%s is %s!' % (spanish_word, english_word)

                    # Language in which you want to convert
                    language = 'en'

                    # Passing the text and language to the engine, 
                    # here we have marked slow=False. Which tells 
                    # the module that the converted audio should 
                    # have a high speed
                    myobj = gTTS(text=mytext, lang=language, slow=False)
                    myobj.save("test_word.mp3")
                    sound_test = sound.backend_ptb.SoundPTB('test_word.mp3', stereo=True, hamming=True, name='testme')
                    sound_test.play()
                logging_utilities.log_info('%s--%s--[Repetition #%d]' % (spanish_word, english_word, words_repetition_counter_dictionary_english_to_repetition_ctr[english_word] + 1))
                vocabulary_learner.text = '%s\n\n\n--\n\n\n%s\n\n\n\n\n\n\n\n\n\n\n\n[Repetition #%d]' % (spanish_word, english_word, words_repetition_counter_dictionary_english_to_repetition_ctr[english_word] + 1)
                words_presented_set.add(english_word)
                words_repetition_counter_dictionary_english_to_repetition_ctr[english_word] += 1
                word_presented_last = english_word
                
                myVocabularyPresenterClock = clock.Clock()

                # update status
                vocabulary_learner.status = STARTED
                vocabulary_learner.setAutoDraw(True)

            
            # if vocabulary_learner is active this frame...
            if vocabulary_learner.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            keys = defaultKeyboard.getKeys(clear=False)
            if len(keys) > 1:
                if keys[0].name == "escape" and keys[1].name == "l" or keys[0].name == "l" and keys[1].name == "escape":
                    thisExp.status = FINISHED
                else:
                    defaultKeyboard.clearEvents()
            #if defaultKeyboard.getKeys(keyList=["escape"]):
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                vocabulary_presenter.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in vocabulary_presenter.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                if myVocabularyPresenterClock.getTime() >= duration_per_vocabulary:
                    # Only check for adjustments if this is a run where it shall be done, the file exists, and no adjustment in the last minute!
                    current_time = datetime.datetime.now()
                    if do_adjustments_flag and os.path.exists(adjustment_flag_file) and (current_time >= (time_last_adjustment_code_executed + datetime.timedelta(seconds=15))) and (current_time < (aspired_end_time_this_task - datetime.timedelta(minutes=2))): # and (current_time >= (time_last_adjustment_code_executed + datetime.timedelta(minutes=1))):
                        with open(adjustment_flag_file, 'r') as filetoread:
                            adjustment_code = filetoread.read().rstrip()  # Code 0 == No adjustment needed!; Code 1 == Speedup!; Code -1 == Slow Down!
                            if adjustment_code == str(0):
                                logging_utilities.log_status_change('Vocabulary_Presenter', 'read code zero in the adjustment_flag.txt file, i.e. no adjustment necessary')
                            elif adjustment_code == str(1):
                                logging_utilities.log_status_change('Vocabulary_Presenter', 'read code one in the adjustment_flag.txt file, i.e. speed up necessary')
                                #-#from gtts import gTTS
                                #-## The text that you want to convert to audio
                                mytext = 'Increasing Difficulty!'

                                # Language in which you want to convert
                                language = 'en'

                                # Passing the text and language to the engine, 
                                # here we have marked slow=False. Which tells 
                                # the module that the converted audio should 
                                # have a high speed
                                myobj = gTTS(text=mytext, lang=language, slow=False)
                                myobj.save("test_word.mp3")
                                sound_test = sound.backend_ptb.SoundPTB('test_word.mp3', stereo=True, hamming=True, name='testme')
                                sound_test.play()
                                time.sleep(2)
                                if changes_made_speed_up >= 2:
                                    changes_made_speed_up = 0
                                else:
                                    new_random_words = random.sample(range(len(vocabularies_pd)), 20) # for now, just randomly add 20 new words
                                    for word_idx in new_random_words:
                                        translation_word, english_word = vocabularies_pd.iloc[word_idx]
                                        translations_dictionary_from_english_to_lang[english_word] = translation_word
                                        words_repetition_counter_dictionary_english_to_repetition_ctr[english_word] = 0
                                        words_not_yet_presented_set.add(english_word)
                                    duration_per_vocabulary = duration_per_vocabulary - 2 if (duration_per_vocabulary > 8) else 6 # also, double the speed!
                                    changes_made_speed_up += 1
                                    changes_made_slow_down = 0
                                # not changing the amount_repetitions_to_do yet, as I can't get out of my head calculated how it would need to change!
                            elif adjustment_code == str(-1):
                                logging_utilities.log_status_change('Vocabulary_Presenter', 'read code minus one in the adjustment_flag.txt file, i.e. slow down necessary')
                                #-#from gtts import gTTS
                                #-## The text that you want to convert to audio
                                mytext = 'Reducing Difficulty!'

                                # Language in which you want to convert
                                language = 'en'

                                # Passing the text and language to the engine, 
                                # here we have marked slow=False. Which tells 
                                # the module that the converted audio should 
                                # have a high speed
                                myobj = gTTS(text=mytext, lang=language, slow=False)
                                myobj.save("test_word.mp3")
                                sound_test = sound.backend_ptb.SoundPTB('test_word.mp3', stereo=True, hamming=True, name='testme')
                                sound_test.play()
                                time.sleep(2)
                                if changes_made_slow_down >= 2:
                                     changes_made_slow_down = 0
                                else:
                                    duration_per_vocabulary = duration_per_vocabulary + 2 if (duration_per_vocabulary < 10) else 10 # double the time per vocabulary!
                                    # also, remove half of the words that were not yet presented!
                                    for i in range(int(len(words_not_yet_presented_set)/2)):
                                        words_not_yet_presented_set.pop()
                                    changes_made_slow_down +=1
                                    changes_made_speed_up = 0
                            elif adjustment_code == str(2):
                                logging_utilities.log_status_change('Vocabulary_Presenter', 'read code two in the adjustment_flag.txt file, i.e. pause internal timer and show motivational things then decide if +1 or 0 as action necessary')
                                #-#from gtts import gTTS
                                #-## The text that you want to convert to audio
                                mytext = 'Reducing Difficulty!'

                                # Language in which you want to convert
                                language = 'en'

                                # Passing the text and language to the engine, 
                                # here we have marked slow=False. Which tells 
                                # the module that the converted audio should 
                                # have a high speed
                                myobj = gTTS(text=mytext, lang=language, slow=False)
                                myobj.save("test_word.mp3")
                                sound_test = sound.backend_ptb.SoundPTB('test_word.mp3', stereo=True, hamming=True, name='testme')
                                sound_test.play()
                                time.sleep(2)
                                if changes_made_slow_down >= 2:
                                     changes_made_slow_down = 0
                                else:
                                    duration_per_vocabulary = duration_per_vocabulary + 2 if (duration_per_vocabulary < 10) else 10 # double the time per vocabulary!
                                    # also, remove half of the words that were not yet presented!
                                    for i in range(int(len(words_not_yet_presented_set)/2)):
                                        words_not_yet_presented_set.pop()
                                    changes_made_slow_down +=1
                                    changes_made_speed_up = 0
                            else:
                                logging_utilities.log_status_change('Vocabulary_Presenter', 'read miscelaneous adjustment code - not doing anything')
                        time_last_adjustment_code_executed = datetime.datetime.now()
                    
                    # First, define new word_to_present and therefore check if we can do a repetition or a new word or if we need to repeat or present a new word
                    word_to_present = None
                    # If new words can be presented and if the repetitions we want to achieve are not yet achieved for all words,
                    # then check if there is enough time to do a repetition or if we need to present the new words now
                    if (len(words_not_yet_presented_set) >= 1) and (min(words_repetition_counter_dictionary_english_to_repetition_ctr.values()) < amount_repetitions_to_do):
                        time_remaining = aspired_end_time_this_task - datetime.timedelta(seconds=duration_per_vocabulary) - current_time
                        time_needed_for_remaining_unpresented_words = len(words_not_yet_presented_set) * duration_per_vocabulary
                        # We have enough time to go for either a new word or a repetition
                        if time_remaining.total_seconds() >= (time_needed_for_remaining_unpresented_words + duration_per_vocabulary):
                            new_word = random_number_for_boolean_check() < 0.7  # Chance if there will be a new word or a repetition! Exciting! bool(random.getrandbits(1))
                        else:
                            new_word = False # Need to repeat a word!
                    elif len(words_not_yet_presented_set) == 0:     # Need to repeat a word, as we can't show a new one
                        new_word = False
                    elif len(words_repetition_counter_dictionary_english_to_repetition_ctr) == 0:   # can't repeat any word
                        new_word = len(words_not_yet_presented_set) >= 1  # check if the set has members, otherwise simply repeat another word
                    else:   # Need to present a new word, regardless that the repetitions are all achieved
                        new_word = len(words_not_yet_presented_set) >= 1  # check if the set has members, otherwise simply repeat another word
                    
                    if new_word:
                        try:
                            word_to_present = words_not_yet_presented_set.pop()
                        except KeyError:
                            continueRoutine = False
                    else:
                        words_potentially_to_repeat_set = set([k for k, v in words_repetition_counter_dictionary_english_to_repetition_ctr.items() if v < amount_repetitions_to_do])
                        try:
                            if len(words_potentially_to_repeat_set) >= 1:
                                word_to_present = words_potentially_to_repeat_set.pop()
                            elif len(set([k for k, v in words_repetition_counter_dictionary_english_to_repetition_ctr.items() if v < amount_repetitions_to_do])) >= 2:
                                while (word_to_present == word_presented_last) and len(words_potentially_to_repeat_set) >= 1:
                                    word_to_present = words_potentially_to_repeat_set.pop()
                        except KeyError:
                            continueRoutine = False

                    if word_to_present == None:
                       continueRoutine = False 

                    if continueRoutine:
                        spanish_word = translations_dictionary_from_english_to_lang[word_to_present]
                        logging_utilities.log_info('%s--%s--[Repetition #%d]' % (spanish_word, word_to_present, words_repetition_counter_dictionary_english_to_repetition_ctr[word_to_present] + 1))
                        if do_adjustments_flag:
                            mytext = '%s is %s!' % (spanish_word, word_to_present)

                            # Language in which you want to convert
                            language = 'en'

                            # Passing the text and language to the engine, 
                            # here we have marked slow=False. Which tells 
                            # the module that the converted audio should 
                            # have a high speed
                            myobj = gTTS(text=mytext, lang=language, slow=False)
                            myobj.save("test_word.mp3")
                            sound_test = sound.backend_ptb.SoundPTB('test_word.mp3', stereo=True, hamming=True, name='testme')
                            sound_test.play()
                        if not new_word:
                            vocabulary_learner.text = '%s\n\n\n--\n\n\n%s\n\n\n\n\n\n\n\n\n\n\n\n[Repetition #%d]' % (spanish_word, word_to_present, words_repetition_counter_dictionary_english_to_repetition_ctr[word_to_present] + 1)
                        else:
                            vocabulary_learner.text = '%s\n\n\n--\n\n\n%s\n\n\n\n\n\n\n\n\n\n\n\n[Repetition #%d]' % (spanish_word, word_to_present, words_repetition_counter_dictionary_english_to_repetition_ctr[word_to_present] + 1)
                        words_presented_set.add(word_to_present)
                        words_repetition_counter_dictionary_english_to_repetition_ctr[word_to_present] += 1
                        word_presented_last = word_to_present
                        myVocabularyPresenterClock.reset()
                win.flip()
        
        # --- Ending Routine "vocabulary_presenter" ---
        for thisComponent in vocabulary_presenter.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for vocabulary_presenter
        vocabulary_presenter.tStop = globalClock.getTime(format='float')
        vocabulary_presenter.tStopRefresh = tThisFlipGlobal
        thisExp.addData('vocabulary_presenter.stopped', vocabulary_presenter.tStop)
        thisExp.nextEntry()
        # the Routine "vocabulary_presenter" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "vocabulary_instructions_for_task" ---
        # create an object to store info about Routine vocabulary_instructions_for_task
        vocabulary_instructions_for_task = data.Routine(
            name='vocabulary_instructions_for_task',
            components=[text, spacebar_to_continue_vocabulary_learning_after_test_instructions],
        )
        vocabulary_instructions_for_task.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for spacebar_to_continue_vocabulary_learning_after_test_instructions
        spacebar_to_continue_vocabulary_learning_after_test_instructions.keys = []
        spacebar_to_continue_vocabulary_learning_after_test_instructions.rt = []
        _spacebar_to_continue_vocabulary_learning_after_test_instructions_allKeys = []
        # store start times for vocabulary_instructions_for_task
        vocabulary_instructions_for_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        vocabulary_instructions_for_task.tStart = globalClock.getTime(format='float')
        vocabulary_instructions_for_task.status = STARTED
        thisExp.addData('vocabulary_instructions_for_task.started', vocabulary_instructions_for_task.tStart)
        vocabulary_instructions_for_task.maxDuration = None
        # keep track of which components have finished
        vocabulary_instructions_for_taskComponents = vocabulary_instructions_for_task.components
        for thisComponent in vocabulary_instructions_for_task.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "vocabulary_instructions_for_task" ---
        vocabulary_instructions_for_task.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *spacebar_to_continue_vocabulary_learning_after_test_instructions* updates
            waitOnFlip = False
            
            # if spacebar_to_continue_vocabulary_learning_after_test_instructions is starting this frame...
            if spacebar_to_continue_vocabulary_learning_after_test_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                spacebar_to_continue_vocabulary_learning_after_test_instructions.frameNStart = frameN  # exact frame index
                spacebar_to_continue_vocabulary_learning_after_test_instructions.tStart = t  # local t and not account for scr refresh
                spacebar_to_continue_vocabulary_learning_after_test_instructions.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(spacebar_to_continue_vocabulary_learning_after_test_instructions, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'spacebar_to_continue_vocabulary_learning_after_test_instructions.started')
                # update status
                spacebar_to_continue_vocabulary_learning_after_test_instructions.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(spacebar_to_continue_vocabulary_learning_after_test_instructions.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(spacebar_to_continue_vocabulary_learning_after_test_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if spacebar_to_continue_vocabulary_learning_after_test_instructions.status == STARTED and not waitOnFlip:
                theseKeys = spacebar_to_continue_vocabulary_learning_after_test_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _spacebar_to_continue_vocabulary_learning_after_test_instructions_allKeys.extend(theseKeys)
                if len(_spacebar_to_continue_vocabulary_learning_after_test_instructions_allKeys):
                    spacebar_to_continue_vocabulary_learning_after_test_instructions.keys = _spacebar_to_continue_vocabulary_learning_after_test_instructions_allKeys[-1].name  # just the last key pressed
                    spacebar_to_continue_vocabulary_learning_after_test_instructions.rt = _spacebar_to_continue_vocabulary_learning_after_test_instructions_allKeys[-1].rt
                    spacebar_to_continue_vocabulary_learning_after_test_instructions.duration = _spacebar_to_continue_vocabulary_learning_after_test_instructions_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            keys = defaultKeyboard.getKeys(clear=False)
            if len(keys) > 1:
                if keys[0].name == "escape" and keys[1].name == "l" or keys[0].name == "l" and keys[1].name == "escape":
                    thisExp.status = FINISHED
                else:
                    defaultKeyboard.clearEvents()
            #if defaultKeyboard.getKeys(keyList=["escape"]):
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                vocabulary_instructions_for_task.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in vocabulary_instructions_for_task.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "vocabulary_instructions_for_task" ---
        for thisComponent in vocabulary_instructions_for_task.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for vocabulary_instructions_for_task
        vocabulary_instructions_for_task.tStop = globalClock.getTime(format='float')
        vocabulary_instructions_for_task.tStopRefresh = tThisFlipGlobal
        thisExp.addData('vocabulary_instructions_for_task.stopped', vocabulary_instructions_for_task.tStop)
        # check responses
        if spacebar_to_continue_vocabulary_learning_after_test_instructions.keys in ['', [], None]:  # No response was made
            spacebar_to_continue_vocabulary_learning_after_test_instructions.keys = None
        thisExp.addData('spacebar_to_continue_vocabulary_learning_after_test_instructions.keys',spacebar_to_continue_vocabulary_learning_after_test_instructions.keys)
        if spacebar_to_continue_vocabulary_learning_after_test_instructions.keys != None:  # we had a response
            thisExp.addData('spacebar_to_continue_vocabulary_learning_after_test_instructions.rt', spacebar_to_continue_vocabulary_learning_after_test_instructions.rt)
            thisExp.addData('spacebar_to_continue_vocabulary_learning_after_test_instructions.duration', spacebar_to_continue_vocabulary_learning_after_test_instructions.duration)
        thisExp.nextEntry()
        # the Routine "vocabulary_instructions_for_task" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "vocabulary_tester" ---
        # create an object to store info about Routine vocabulary_tester
        vocabulary_tester = data.Routine(
            name='vocabulary_tester',
            components=[text_2, textbox],
        )
        vocabulary_tester.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textbox.reset()
        # store start times for vocabulary_tester
        vocabulary_tester.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        vocabulary_tester.tStart = globalClock.getTime(format='float')
        vocabulary_tester.status = STARTED
        thisExp.addData('vocabulary_tester.started', vocabulary_tester.tStart)
        vocabulary_tester.maxDuration = None
        # keep track of which components have finished
        vocabulary_testerComponents = vocabulary_tester.components
        for thisComponent in vocabulary_tester.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "vocabulary_tester" ---
        # calculate seconds per word to present in time to finish in approx. 10 minutes all vocabulary presented
        time_per_vocabulary_for_repetition = (10*60) / len(words_presented_set) # TODO: Adjust this time here!
        duration_per_vocabulary = time_per_vocabulary_for_repetition
        start_time_vocabulary_presenter = datetime.datetime.now()
        aspired_end_time = start_time_vocabulary_presenter + datetime.timedelta(minutes=10, seconds=int(time_per_vocabulary_for_repetition))
        myVocabularyTestClock = clock.Clock()
        
        vocabulary_tester.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and datetime.datetime.now() <= aspired_end_time:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)

                try:
                    word_to_present = words_presented_set.pop()
                except KeyError:
                    continueRoutine = False
                
                if continueRoutine:
                    spanish_word = translations_dictionary_from_english_to_lang[word_to_present]
                    logging_utilities.log_info('%s: %s -- English: %s' % (TO_LANG, spanish_word, word_to_present))
                    from_english = random_number_for_boolean_check() < 0.7  # Chance if we start from english or spanish! Exciting!
                    starter_word = word_to_present if from_english else spanish_word
                    target_word = spanish_word if from_english else word_to_present
                    from_lang = FROM_LANG if from_english else TO_LANG
                    to_lang = TO_LANG if from_english else FROM_LANG
                    new_text = 'Please translate %s from %s to %s' % (starter_word.strip(), from_lang, to_lang)
                    text_2.text = new_text
                    logging_utilities.log_info('%s with target: %s' % (new_text, target_word))
                    num_correct = 0
                    textbox.setHasFocus(True)
                    myVocabularyTestClock.reset()
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # *textbox* updates
            
            # if textbox is starting this frame...
            if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textbox.frameNStart = frameN  # exact frame index
                textbox.tStart = t  # local t and not account for scr refresh
                textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textbox.started')
                # update status
                textbox.status = STARTED
                textbox.setAutoDraw(True)
            
            # if textbox is active this frame...
            if textbox.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            keys = defaultKeyboard.getKeys(clear=False)
            if len(keys) > 1:
                if keys[0].name == "escape" and keys[1].name == "l" or keys[0].name == "l" and keys[1].name == "escape":
                    thisExp.status = FINISHED
                else:
                    defaultKeyboard.clearEvents()
            #if defaultKeyboard.getKeys(keyList=["escape"]):
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                vocabulary_tester.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in vocabulary_tester.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                if myVocabularyTestClock.getTime() >= duration_per_vocabulary:
                    test_target = unidecode(target_word.strip()).translate(TRANSLATOR)
                    test_answer = unidecode(textbox.text.strip()).translate(TRANSLATOR)
                    test_correctly_answered = (test_target == test_answer) or (test_target.lower() in test_answer.lower()) or (any([test_target.lower() == answer_part for answer_part in test_answer.lower().split(' ')]))
                    if test_correctly_answered:
                        num_correct += 1
                    logging_utilities.log_info('The answer in the textbox was %s, which is %s, as the target was %s' % (test_answer, 'correct' if test_correctly_answered else 'incorrect', test_target))
                    textbox.text = ''
                    if len(words_presented_set) == 0:   # All the words were already presented, so stop the routine
                        continueRoutine = False
                    else:
                        # Continue with the next word!
                        # logging_utilities.log_info('The text in the textbox was: %s' % textbox.text)
                        textbox.text = ''
                        word_to_present = words_presented_set.pop()
                        spanish_word = translations_dictionary_from_english_to_lang[word_to_present]
                        logging_utilities.log_info('%s: %s -- English: %s' % (TO_LANG, spanish_word, word_to_present))
                        from_english = random_number_for_boolean_check() < 0.7  # Chance if we start from english or spanish! Exciting!
                        starter_word = word_to_present if from_english else spanish_word
                        target_word = spanish_word if from_english else word_to_present
                        from_lang = FROM_LANG if from_english else TO_LANG
                        to_lang = TO_LANG if from_english else FROM_LANG
                        new_text = 'Please translate %s from %s to %s' % (starter_word.strip(), from_lang, to_lang)
                        text_2.text = new_text
                        logging_utilities.log_info('%s with target: %s' % (new_text, target_word))
                        textbox.setHasFocus(True)
                    myVocabularyTestClock.reset()
                win.flip()
        
        # Do the grading of the vocabulary test
        test_performance = '%.2f%%, as %d out of %d words were correctly identified!' % ((num_correct / len(ten_random_vocabularies_idxs)) * 100, num_correct, len(ten_random_vocabularies_idxs))
        
        # --- Ending Routine "vocabulary_tester" ---
        for thisComponent in vocabulary_tester.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for vocabulary_tester
        vocabulary_tester.tStop = globalClock.getTime(format='float')
        vocabulary_tester.tStopRefresh = tThisFlipGlobal
        thisExp.addData('vocabulary_tester.stopped', vocabulary_tester.tStop)
        thisExp.nextEntry()
        # the Routine "vocabulary_tester" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "end_vocabulary" ---
        # create an object to store info about Routine end_vocabulary
        end_vocabulary = data.Routine(
            name='end_vocabulary',
            components=[text_3, vocabulary_end_after_results_reveal],
        )
        end_vocabulary.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for vocabulary_end_after_results_reveal
        vocabulary_end_after_results_reveal.keys = []
        vocabulary_end_after_results_reveal.rt = []
        _vocabulary_end_after_results_reveal_allKeys = []
        # store start times for end_vocabulary
        end_vocabulary.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        end_vocabulary.tStart = globalClock.getTime(format='float')
        end_vocabulary.status = STARTED
        thisExp.addData('end_vocabulary.started', end_vocabulary.tStart)
        end_vocabulary.maxDuration = None
        # keep track of which components have finished
        end_vocabularyComponents = end_vocabulary.components
        for thisComponent in end_vocabulary.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "end_vocabulary" ---
        end_vocabulary.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.started')
                # add here the performance that was just achieved
                text_3.text = text_3.text % test_performance
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # *vocabulary_end_after_results_reveal* updates
            waitOnFlip = False
            
            # if vocabulary_end_after_results_reveal is starting this frame...
            if vocabulary_end_after_results_reveal.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                vocabulary_end_after_results_reveal.frameNStart = frameN  # exact frame index
                vocabulary_end_after_results_reveal.tStart = t  # local t and not account for scr refresh
                vocabulary_end_after_results_reveal.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(vocabulary_end_after_results_reveal, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'vocabulary_end_after_results_reveal.started')
                # update status
                vocabulary_end_after_results_reveal.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(vocabulary_end_after_results_reveal.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(vocabulary_end_after_results_reveal.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if vocabulary_end_after_results_reveal.status == STARTED and not waitOnFlip:
                theseKeys = vocabulary_end_after_results_reveal.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _vocabulary_end_after_results_reveal_allKeys.extend(theseKeys)
                if len(_vocabulary_end_after_results_reveal_allKeys):
                    vocabulary_end_after_results_reveal.keys = _vocabulary_end_after_results_reveal_allKeys[-1].name  # just the last key pressed
                    vocabulary_end_after_results_reveal.rt = _vocabulary_end_after_results_reveal_allKeys[-1].rt
                    vocabulary_end_after_results_reveal.duration = _vocabulary_end_after_results_reveal_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            keys = defaultKeyboard.getKeys(clear=False)
            if len(keys) > 1:
                if keys[0].name == "escape" and keys[1].name == "l" or keys[0].name == "l" and keys[1].name == "escape":
                    thisExp.status = FINISHED
                else:
                    defaultKeyboard.clearEvents()
            #if defaultKeyboard.getKeys(keyList=["escape"]):
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                end_vocabulary.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in end_vocabulary.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "end_vocabulary" ---
        for thisComponent in end_vocabulary.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for end_vocabulary
        end_vocabulary.tStop = globalClock.getTime(format='float')
        end_vocabulary.tStopRefresh = tThisFlipGlobal
        thisExp.addData('end_vocabulary.stopped', end_vocabulary.tStop)
        # check responses
        if vocabulary_end_after_results_reveal.keys in ['', [], None]:  # No response was made
            vocabulary_end_after_results_reveal.keys = None
        thisExp.addData('vocabulary_end_after_results_reveal.keys',vocabulary_end_after_results_reveal.keys)
        if vocabulary_end_after_results_reveal.keys != None:  # we had a response
            thisExp.addData('vocabulary_end_after_results_reveal.rt', vocabulary_end_after_results_reveal.rt)
            thisExp.addData('vocabulary_end_after_results_reveal.duration', vocabulary_end_after_results_reveal.duration)
        thisExp.nextEntry()
        # the Routine "end_vocabulary" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # TODO In the future, make here the distinction if there was an adaptation that happened! If so (based on scanning a flag-file), then make sure to ask another type of questionnaires as well and play the adjustment sound!!
        questionnaires_after_task(continueRoutine, thisExp, routineTimer, win, eye_closing_timer, frameTolerance, endExpNow, defaultKeyboard, task_idx + 1)
        # the Routine "end_vocabulary" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    
    # eye_closing(thisExp, routineTimer, win, eye_closing_timer, frameTolerance, endExpNow, defaultKeyboard, base_path) ## TODO: THIS IS THE REASON FOR THE EXCEPTION!!
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':    
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp, TASKS_ORDER, THIS_PARTICIPANT_LANGUAGE_PREFERENCES, start_of_experiment = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    try:
        run_based_on_time(
            expInfo=expInfo, 
            thisExp=thisExp, 
            TASKS_ORDER=TASKS_ORDER,
            THIS_PARTICIPANT_LANGUAGE_PREFERENCES=THIS_PARTICIPANT_LANGUAGE_PREFERENCES,
            start_of_experiment = start_of_experiment,
            win=win,
            globalClock='float'
        )
    except Exception as e:
        logging_utilities.log_info(str(repr(e)))
        print(repr(e))
        exit(-1)
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
