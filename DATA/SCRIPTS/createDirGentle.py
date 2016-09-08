import os
import sys
import shutil

argl = len(sys.argv)

if argl < 2:
	print "Please provide a directory name"
	sys.exit()

dir_name = sys.argv[1]

if os.path.exists(dir_name):
	print "Directory already exists! (dir_name)"
	sys.exit()
os.makedirs(dir_name)