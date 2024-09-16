import tkinter as tk
import customtkinter as ctk

#from ArmPi_windowsMJ import common_code_windows_MJ as cc
#from ArmPi_windowsMJ import take_images

from Page1 import Page as Page1
from Page2 import Page as Page2

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("green")

class Main:
    def __init__(self):
        #Window Boilerplate

        self.window = ctk.CTk()
        self.window.title("Grid")
        self.window.geometry("1440x1080")
        self.window.columnconfigure((0), weight = 1)
        self.window.rowconfigure(0, weight = 1)

        #Error Warning Code

        label1 = ctk.CTkLabel(self.window, text = "Error", bg_color = "red")
        label2 = ctk.CTkLabel(self.window, text = "Error 2", bg_color = "orange")

        label1.grid(row = 0, column = 0, sticky = "nsew")
        label2.grid(row = 0, column = 0, sticky = "nsew")

        #Pages Code

        self.pages = {}

        for currentPage in (Page1, Page2):
            page = currentPage(self.window, self) #each page is a frame object, this initializes the page
            self.pages[str(page)] = page #adds page to list of pages
            page.grid(column = 0, row = 0, sticky = "nsew") #places the page in cell (0, 0) and fills it to the boarders


        self.show_page("Page2") #This is the first page to display

    def show_page(self, cont): #Allows for switching pages
        page = self.pages[str(cont)] #Allows for string name (toString) and object type to be passed
        page.tkraise()
        print("Page switched")

    def get_win_height(self):
        return self.window.winfo_height()

main = Main()
main.window.mainloop()