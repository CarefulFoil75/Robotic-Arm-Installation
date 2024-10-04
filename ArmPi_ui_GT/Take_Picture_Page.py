#import tkinter as tk
import customtkinter as ctk
import sys
from PIL import Image

from ArmPi_windowsMJ import common_code_windows_MJ as cc
sys.path.append(str(cc.curr_dir))
from ArmPi_windowsMJ import take_images

LARGEFONT =("Verdana", 100)

class Page(ctk.CTkFrame):
	
	def __init__(self, parent, controller):
		ctk.CTkFrame.__init__(self, parent)
		
		self._set_appearance_mode("light")
		self.columnconfigure((0, 1, 2), weight = 1)
		self.rowconfigure((0, 2), weight = 3)
		self.rowconfigure((1, 3), weight = 1)
		self.rowconfigure((4), weight = 8)
		
		self.controller = controller
		self.image_number = 0
		self.image_type = 0
		self.image_types = ["Red", "Green", "Blue"]
		# button to show frame 2 with text
		# layout2
		self.button1 = ctk.CTkButton(self, font = ("Arial", 60), text ="Take Picture",
							command = lambda : self.take_image())
		self.button2 = ctk.CTkButton(self, font = ("Arial", 60), text ="Continue",
							command = lambda : self.next_page())
		self.label1 = ctk.CTkLabel(self, font = ("Arial", 100), text_color = "black", text = "Take 10 Pictures of " + self.image_types[self.image_type] + " Cubes")
		

		#self.my_image = ctk.CTkImage(light_image=Image.open(cc.dev_path / 'breeze.png'), dark_image=Image.open(cc.dev_path / 'breeze.png'), size=(30, 30))
		#self.image_label = ctk.CTkLabel(self, image=self.my_image._get_scaled_light_photo_image((1, 1)), text=None)  
	
		# putting the button in its place by 
		# using grid
		self.button1.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = "news")
		self.label1.grid(row = 0, column = 0, sticky = "nsew", columnspan = 3)
		#self.image_label.grid(row = 3, column = 0, sticky = "nsew", columnspan = 3)

		#for i in range(1, 4, 1):
		#	for j in range(1, 6, 1):
		#		slot = ctk.CTkLabel(self, bg_color = ("#" + str(i * 10) + str(j * 10) + str(i * 10 + j)))
		#		slot.grid(row = j - 1, column = i - 1, sticky = "news")
		
	def next_page(self):
		if(self.image_type >= 2):
			self.controller.show_page("Test_Model_Page")
		self.reset()

	def reset(self):
		self.image_number = 0
		self.image_type += 1
		self.label1.configure(text = "Take 10 Pictures of " + self.image_types[self.image_type] + " Cubes")
		self.button2.grid_forget()

	def take_image(self):
		take_images.take_image("demo", self.image_types[self.image_type], self.image_types[self.image_type] + str(self.image_number))
		self.image_number += 1
		if(self.image_number >= 3):
			self.button2.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = "news")

	def __str__(self):
	    return "Take_Picture_Page"