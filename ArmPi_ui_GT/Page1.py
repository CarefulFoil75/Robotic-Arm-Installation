import tkinter as tk
import customtkinter as ctk

class Page(ctk.CTkFrame):
    def __init__(self, parent: ctk.CTk(), controller):
        super().__init__(parent)
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller

        self._set_appearance_mode("light")
        self.columnconfigure((0, 1, 2), weight = 1)
        self.rowconfigure((0, 1, 2, 3), weight = 1)
        

        self.label1 = ctk.CTkLabel(self, font = ("Arial", 100), text_color = "black", text = "Welcome to the Robotic Arm \nAI Trainer")
        self.button1 = ctk.CTkButton(self, text = "Start", font = ("Arial", 100), text_color = "white", border_color =  "black", corner_radius = 100,
							command = lambda : controller.show_page("Page2"))

        self.label1.grid(row = 0, column = 0, sticky = "nsew", columnspan = 3)
        self.button1.grid(row = 2, column = 1, sticky = "nsew")

    def __str__(self):
	    return "Page1"