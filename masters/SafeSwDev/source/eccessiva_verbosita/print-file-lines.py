#!/usr/bin/python3

import sys

for arg in sys.argv[1:]:
	try:
		f = open(arg, 'r')
	except OSError:
		print('cannot open', arg)
	else:
		l = len(f.readlines())
		print(arg, 'has', l, 'lines' if l > 1 else 'line')
		f.close()
