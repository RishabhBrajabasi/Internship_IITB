# from tkinter import *
import Tkinter as tk
from winsound import *

root = tk.Tk() # create tkinter window

play = lambda: PlaySound('Sound.wav', SND_FILENAME)
button = tk.Button(root, text = 'Play', command = play)

button.pack()
root.mainloop()