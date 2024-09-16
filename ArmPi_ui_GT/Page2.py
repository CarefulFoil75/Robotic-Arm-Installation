#import tkinter as tk
import customtkinter as ctk

LARGEFONT =("Verdana", 100)

class Page(ctk.CTkFrame):
	
	def __init__(self, parent, controller):
		ctk.CTkFrame.__init__(self, parent)
		
		self._set_appearance_mode("light")
		self.columnconfigure((0, 1, 2, 3, 4), weight = 1)
		self.rowconfigure((0, 1, 2, 3, 4, 5, 6), weight = 1)
		
		# button to show frame 2 with text
		# layout2
		button2 = ctk.CTkButton(self, text ="Page 1",
							command = lambda : controller.show_page("Page1"))
	
		# putting the button in its place by 
		# using grid
		button2.grid(row = 2, column = 3, padx = 10, pady = 10, sticky = "news")

	def __str__(self):
	    return "Page2"