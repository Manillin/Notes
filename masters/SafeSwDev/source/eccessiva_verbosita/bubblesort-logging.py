#!/usr/bin/python3

import logging

def bubblesort(elements):
	swapped = False
	logging.info("bubblesort: start");
	# Looping from size of array from last index[-1] to index [0]
	for n in range(len(elements)-1, 0, -1):
		logging.info("n: {}".format((n)));
		for i in range(n):
			logging.info("i: {}".format(str(i)));
			if elements[i] > elements[i + 1]:
				logging.info("swapped = true");
				swapped = True
				# swapping data if the element is less than next element in the array
				elements[i], elements[i + 1] = elements[i + 1], elements[i]	
		if not swapped:
			logging.info("swapped = false");
			# exiting the function if we didn't make a single swap
			# meaning that the array is already sorted.
			return

elements = [39, 12, 18, 85, 72, 10, 2, 18]

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
print("Unsorted list is,")
print(elements)
bubblesort(elements)
print("Sorted Array is, ")
print(elements)
