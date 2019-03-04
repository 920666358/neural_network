#pip install pillow
from PIL import Image
import os
import numpy

test_picture_dir = "test_samples"

def ProcessInTotal(fp) :
    with Image.open(infile) as im:
        size = (min(im.height,im.width), min(im.height,im.width))
        im.resize(size)
        size = (28, 28) #缩放为28x28像素
        im.resize(size)
        res_file_path = fp[:-3]+"{}x{}.png".format(*size) # *将tuple解开为两个单独的变量
        im.save(res_file_path)
        # 转换为黑白图片
        gray_im = im.convert("L")
        res_file_path = fp[:-3]+"{}x{}_gray.png".format(*size) # *将tuple解开为两个单独的变量
        gray_im.save(res_file_path)

def resize2Square(im):
    size = (min(im.height,im.width), min(im.height,im.width))
    res_im = im.resize(size)
    size = (28, 28) #缩放为28x28像素
    res_im = res_im.resize(size)
    return res_im

def color2gray(im):
    gray_im = im.convert("L")
    return gray_im

def image2array(im):
    pix = numpy.array(im.getdata()).reshape(im.size[0], im.size[1])
    #pix = numpy.array(im.getdata())
    return pix

for infile in os.listdir(test_picture_dir):
    infilepath = test_picture_dir + "/" + infile
    with Image.open(infilepath) as im :
        new_im = resize2Square(im)
        new_im = color2gray(new_im)
        narray = image2array(new_im)
        print("end")
        print(narray)
        break
