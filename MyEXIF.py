import random

import exifread
from datetime import datetime
from datetime import date
import calendar
from apscheduler.schedulers.background import BackgroundScheduler
import shutil
from shutil import copyfile
import os,sys
from PIL import Image
from PIL.ExifTags import TAGS
from libxmp.files import XMPFiles
from libxmp.utils import file_to_dict
from libxmp import consts
import piexif
import piexif.helper
import pyexiv2
import wx
import Wizard
import Objects
import math
import cv2
import dlib
import openface
import sklearn
import openface.helper
import json
import time
from sklearn.svm import SVC
import numpy as np
from DBoperations import DBoperations
from Objects import Kindergarten
from Objects import ImageObject
from Objects import Event
from Objects import Album
from Objects import Rectangle
from Objects import Child
import logging
from QualityAssessment import Quality
from shutil import copy2

class MyExif():

    """
    Function converts EXIF value of 'UserComment' TAG
    and returns it as a dictionary format """
    def get_author_data_from_EXIF_as_dictionary(self, value):
        str_data = '{' + value.split("{", 1)[1]
        dict_data = eval(str_data)
        return dict_data

    def write_to_image_exif(self, dict, image_source_path):
        try:
            json_comment = json.dumps(dict)
            user_comment = piexif.helper.UserComment.dump(json_comment)
            exif_dict = piexif.load(image_source_path)
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, image_source_path)
        except Exception as ex:
            print(ex)

    def write_updated_data_to_image_exif(self, image_object, image_source_path):

        """ EXAMPLE OF OUTPUT DICTIONARY """

        d = {'final_image_tag': 6,
             'timestamp': 'Sunday  2017-09-17  10:07:57',
             'author':

                 {'author_email': 'aba.yuval@gmail.com',
                  'child_of_author':

                      {'tag_number': 6,
                       'tagged': 0,
                       'name': 'YUVAL DVIRI'},

                  'author_phone': '050-0000006'},

             'image_name': 'DSC03924.JPG',
             'detected_faces':

                 {'num_of_detected_faces': 4,
                  'faces_details': [

                      {'face_quality': 0.5,
                       'relative_size_of_face_to_image': 0.3,
                       'is_face_tagged': 1,
                       'tag_number': 6,
                       'face_importance': 1},

                      {'face_quality': 0.5,
                       'relative_size_of_face_to_image': 0.2,
                       'is_face_tagged': 0,
                       'tag_number': 0,
                       'face_importance': 2},

                      {'face_quality': 0.5,
                       'relative_size_of_face_to_image': 0.1,
                       'is_face_tagged': 0,
                       'tag_number': 0,
                       'face_importance': 3},

                      {'face_quality': 0.5,
                       'relative_size_of_face_to_image': 0.1,
                       'is_face_tagged': 0,
                       'tag_number': 0,
                       'face_importance': 4}]}}

        dict = image_object.exif
        dict['detected_faces'] = image_object.detected_faces
        dict['final_image_tag'] = image_object.image_tag
        #dict['number of detected faces'] = image_object.number_of_detected_faces

        if image_object.image_tag is 0:
            if image_object.number_of_detected_faces is not 0:
                for face in dict['detected_faces']['faces_details']:
                    if face['face_importance'] == 1 and face['is_face_tagged'] == 1:
                        dict['final_image_tag'] = face['tag_number']
            else:
                dict['final_image_tag'] = dict['author']['child_of_author']['tag_number']


        str_timestamp = dict['timestamp'].strftime('%A  %Y-%m-%d  %H:%M:%S')
        dict['timestamp'] = str_timestamp
        self.write_to_image_exif(dict, image_source_path)
        print "\nExif of processed image:\n\n" + str(dict)

    def get_image_EXIF_data(self, image_source_path):

        try:
            info = piexif.load(image_source_path)
            #x = ["Exif"][piexif.ExifIFD.UserComment]
            usercmt = info['Exif'][
                37510]  # v = 'ASCII   {"Image Name": "DSC03614.JPG", "Author": {"child of author": {"tag number": 4, "name": "TUVI BIMBA"}, "author email": "aba.tuvi@gmail.com"}}'
            timestamp = info['0th'][306]
            dict = self.get_author_data_from_EXIF_as_dictionary(usercmt)
            t_stmp = datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S')
            dict['timestamp'] = t_stmp

            return dict
        except Exception as ex:
            print(ex)




    # This function is for 'replacing' the purpose of the KidsAlbums App: the app should write into the EXIF of the image
    # into key "UserComment" details about who took the photo, who is the child of the author of the photo,
    # is the photo 'tagged' (the photo is 'tagged' when the user manually uploads a photo of their child -
    # it means the child in the photo is definitly the child of the author) ect..
    # example: {"Image Name": "IMG_20190427_165142_TAGGED.jpg" , "Author": {"child of author": {"tag number": 3,
    # "name": "GAYA FARAJ", "tagged": 1}, "author phone": "054-0000002",  "author email": "ima.gaya@gmail.com"}}
    # we use this function when we need to "functionally replace" the App - usually for Debug purposes
    def write_untagged_pseudo_image_exif_authors_to_all_images(self):

        user_comments = {}

        first_names = ['YONATAN', 'AYA', 'GAYA', 'TUVI', 'ALMA', 'YUVAL', 'ASI', 'AVITAL', 'AVIV', 'DANA', 'DANI',
                       'DOR',
                       'EVYATAR', 'MAYA', 'NERI', 'NOA', 'NOAM', 'NOGA', 'SAGIV', 'SHAKED', 'SHANI', 'SOPHIE', 'TZADOK',
                       'YASMIN', 'DIMA', 'DAVIDA', 'KAI', 'YOAV', 'IDAN', 'DIKLA']

        last_names = ['LEVI', 'FRIDMAN', 'FARAJ', 'BIMBA', 'COHEN', 'DVIRI', 'AMSALEM', 'MARJI', 'BUHMAN', 'MEIRI',
                      'ENDELMAN', 'SHARONI', 'FISHMAN', 'GEVA', 'TAVOR', 'BENI', 'FISHMAN', 'YOVEL', 'BARUH', 'NAHMANI',
                      'NIMNI', 'ALTMAN', 'MAMO', 'BEN-SHIMON', 'GRAFMAN', 'YOSEFI', 'KIPER', 'SEGULA', 'ASHKENAZI',
                      'GRANATI']

        try:
            for i in range(1, 31):

                child_name = first_names[i - 1] + " " + last_names[i - 1]
                tag = i
                if (i < 10):
                    aba_cell = '050-000000' + str(i)
                    ima_cell = '054-000000' + str(i)
                else:
                    aba_cell = '050-00000' + str(i)
                    ima_cell = '054-00000' + str(i)

                f_email = 'aba.' + first_names[i-1].lower() + '@gmail.com'
                m_email = 'ima.' + first_names[i-1].lower() + '@gmail.com'

                choice = random.choice([True, False])

                if choice:
                    cell = aba_cell
                    email = f_email
                else:
                    cell = ima_cell
                    email = m_email

                user_comment = {"image_name": "",
                                "author": {"child_of_author": {"tag_number": tag, "name": child_name, "tagged": 0},
                                           "author_phone": cell, "author_email": email}}
                user_comments[first_names[i-1]] = user_comment
        except Exception as ex:
            print ex


        PATH = '/home/victoria/original_project/PythonServer/Kids_Photos_Mapping/'
        all_children_directories = os.listdir(PATH)
        all_children_directories.remove('Z_SHARED')
        for dir in all_children_directories:
            dir_path = PATH + dir

            current_comment = user_comments[dir]
            for image in os.listdir(dir_path):
                current_comment['image_name'] = image
                current_image_path = dir_path+'/'+image
                self.write_to_image_exif(current_comment, current_image_path)
                try:
                    shutil.copy2(current_image_path, '/home/victoria/original_project/PythonServer/full_images_data_set_with_exif_for_before_images_folder/')
                except Exception as ex:
                    print ex
        print 'done'










    def write_image_author_to_exif(self, photo_author_email, image_path, image_name):
        try:
            child_of_author = self.currentKindergarten.getChildFromImage(photo_author_email)
            child = {}
            child['name'] = child_of_author.first_name + " " + child_of_author.last_name
            child['tag number'] = child_of_author.tag
            author = {}
            author['author email'] = photo_author_email
            author['child of author'] = child
            comment = {}
            comment['author'] = author
            comment['image name'] = image_name
            json_comment = json.dumps(comment)
            # comment = '{ TAKEN BY : ' + child_of_author.first_name + ' ' + child_of_author.last_name + 'S PARENT, TAG : ' + str(child_of_author.tag) + ' }'
        except Exception as ex:
            print(ex)
        user_comment = piexif.helper.UserComment.dump(json_comment)
        exif_dict = piexif.load(image_path)
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image_path)



    # Write the name of the child to exif of the image
    def writeToImageExif(self, childName, image):
        user_comment = piexif.helper.UserComment.dump(childName)
        exif_dict = piexif.load(image)
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, image)



    """ These functions were written to extract exif data from image using two specific libraries: PIL and ExifRead """


    def get_PIL_image_exif_data(self, image_source_path):

        # PIL
        print('\n<< Test of PIL >> \n')
        try:
            img = Image.open(image_source_path)
            info = img._getexif()

            for k, v in info.items():
                nice = TAGS.get(k, k)

                if type(v) is bytes:
                    try:
                        print('%s (%s) = %s' % (nice, k, v.decode("utf-8")))
                    except:
                        pass
                else:
                    print('%s (%s) = %s' % (nice, k, v))

        except Exception as ex:
            print "Couldent get exif with PIL from image\n" + ex.message

        print "\n************************************************************************************************\n"

    def get_ExifRead_image_exif_data(self, image_source_path):

        try:
            with open(image_source_path, 'rb') as f:
                exif = exifread.process_file(f)

            for k in sorted(exif.keys()):
                if type(exif[k]) is bytes:
                    try:
                        print('%s = %s' % (k, exif[k].decode("utf-8")))
                    except:
                        pass
                else:
                    print('%s = %s' % (k, exif[k]))

        except Exception as ex:
            print "Couldent get exif with EXIF READ from image\n" + ex.message

