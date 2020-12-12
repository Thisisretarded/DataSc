# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:23:31 2020

Script to create directory structure for any project.

@author: rt2
"""
import os
dirs = ["input", "src", "models", "notebooks"]

def main():
    for dirName in dirs:
        try:
            # Create target Directory
            os.makedirs(dirName)
            print("Directory " , dirName ,  " Created ") 
        except FileExistsError:
            print("Directory " , dirName ,  " already exists")  


if __name__ == "__main__":
    main()
