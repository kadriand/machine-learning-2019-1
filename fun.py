import importlib
import sys


def update():
    importlib.reload(sys.modules[__name__])


def rc(age):
    return age*2


def gato2(age):
    return age*4


def height(t):
    return -16*t**2+96*t