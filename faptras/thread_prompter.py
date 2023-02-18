from threading import Thread
import cv2 as cv
import constants
import utils
import globals
import time

class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self.prompt = None
        self.value = None
        self.scheduled = False
        self.running = True
        self.start()
        
    def set_execution_config(self, prompt: str):
        self.prompt = prompt
        self.scheduled = True
        
    def run(self):
        while self.running:
            if self.scheduled:
                print(self.prompt)
                self.value = input()
                self.scheduled = False
                globals.stop_thread = True
            else:
                time.sleep(1)

    def join(self, *args):
        globals.stop_thread = False
        Thread.join(self, *args)