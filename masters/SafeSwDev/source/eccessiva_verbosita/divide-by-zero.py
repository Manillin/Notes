#!/usr/bin/python3

def b(v):
	return v/0

def a(v):
	return b(v)

print(a(1))
