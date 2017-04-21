import Tkinter as tk

class App:
  def __init__(self, master):
    frame = tk.Frame(master)
    frame.pack()
    self.button = tk.Button(frame,
                         text="QUIT", fg="red",
                         command=quit)
    self.button.pack(side=tk.LEFT)
    self.slogan = tk.Button(frame,
                         text="Hello",
                         command=self.write_slogan)
    self.slogan.pack(side=tk.LEFT)
  def write_slogan(self):
    print("Tkinter is easy to use!")

root = tk.Tk()
app = App(root)
root.mainloop()