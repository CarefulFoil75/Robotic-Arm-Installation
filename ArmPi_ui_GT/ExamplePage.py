#import tkinter as tk
import customtkinter as ctk

class ExamplePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        
        self.columnconfigure((0, 1), weight = 1) #columns on the page
        self.rowconfigure(0, weight = 1)#rows on the page

        label1 = ctk.CTkLabel(self, text = "Label 1", bg_color = "green")
        label2 = ctk.CTkLabel(self, text = "Label 2", bg_color = "blue")

        label1.grid(row = 0, column = 0, sticky = "nsew")
        label2.grid(row = 0, column = 1, sticky = "nsew")
        
    def __str__(self):
	    return "ExamplePage"