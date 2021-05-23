#adarsha written 05/12/2020
#updated 05/02/2021

from imutils import paths
import os
import cv2
from lxml import etree
import xml.etree.cElementTree as ET

img = None
tl_list = []
br_list = []
object_list = []

image_folder = 'output'
savedir = 'annotations'
obj = ''



class MyImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name

def write_xml(folder, img,file_name, object,savedir):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    height, width, depth = img.shape
    coordinates_file_name = file_name.split('.')

    coordinates_file_name = coordinates_file_name[0]+'.txt'
    with open(coordinates_file_name,'r') as c:
        coordinates = c.read()
        coordinates = coordinates.split(' ')
    xmin,xmax,ymin,ymax = coordinates
    #print(type(xmax)

    #print('coordinates_file_name = {}'.format(coordinates_file_name))

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = file_name
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    ob = ET.SubElement(annotation, 'object')
    ET.SubElement(ob, 'name').text = str(object)
    ET.SubElement(ob, 'pose').text = 'Unspecified'
    ET.SubElement(ob, 'truncated').text = '0'
    ET.SubElement(ob, 'difficult').text = '0'
    bbox = ET.SubElement(ob, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = str(xmin)
    ET.SubElement(bbox, 'ymin').text = str(ymin)
    ET.SubElement(bbox, 'xmax').text = str(xmax)
    ET.SubElement(bbox, 'ymax').text = str(ymax)
    #ET.SubElement(bbox, 'xmax').text = str(width)
    #ET.SubElement(bbox, 'ymax').text = str(height)

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(savedir, file_name.replace('png', 'xml'))
    #save_path = os.path.join(savedir, file_name.replace('jpg', 'xml'))
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)

def split_file(image_file):
    file_name = str(MyImage(image_file))
    b = file_name.split('\\')
    file_name = b[1]
    b1 = b[1]
    b1 = b1.split('-')
    b1 = list(b1[0])
    obj = int(b1[-2]+b1[-1])
    return file_name, obj

for image_file in sorted(paths.list_images(image_folder)):
    img = cv2.imread(image_file)
    file_name, obj = split_file(image_file)
    write_xml(image_folder, img, file_name, obj,savedir)



