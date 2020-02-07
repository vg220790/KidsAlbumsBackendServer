from asynchat import fifo

import shutil
from shutil import copyfile
import os, sys
from PIL import Image
from PIL.ExifTags import TAGS
from libxmp.files import XMPFiles
from libxmp.utils import file_to_dict
from libxmp import consts
import piexif
import piexif.helper
import pyexiv2
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
from Objects import ImageObject
from Objects import Rectangle
import logging
from QualityAssessment import Quality
from MyEXIF import MyExif
from FaceOperations import FaceOperations


# Logic class manages all the actions and connects the user(UI) with the system and with the database(DBoperations)
class Logic():

    def __init__(self, path, dbx, sql_pswd):
        self.PATH = path
        self.dbx = dbx
        self.quality = Quality(path)
        self.MyExif = MyExif()
        self.database = DBoperations(path, passwd=sql_pswd)
        self.network_model = path + 'PythonServer/openface.nn4.small2.v1.t7'
        self.currentKindergarten = None
        self.face_operations = None

    def init_AFEKA_kindergarten(self):
        if self.currentKindergarten == None:
            print("init afeka method")
            self.database.add_AFEKA_kindergarten_to_table()
            self.database.addChildren_Afeka_Kindergarten()

    # Load kindergarten from the database using DBoperations class
    def loadKindergartenFromDatabase(self):
        if self.currentKindergarten == None:

            try:

                self.currentKindergarten = self.database.loadKindergarten()
                if self.currentKindergarten != None:
                    self.database.loadChildren(self.currentKindergarten)
                    self.database.loadSchedule(self.currentKindergarten)
                    self.database.loadAlbumes(self.currentKindergarten)
                    print self.currentKindergarten
                    if self.currentKindergarten.children != None:
                        # print self.currentKindergarten.children
                        pass
                    if self.currentKindergarten.schedule != None:
                        # print self.currentKindergarten.schedule
                        pass
                    if self.currentKindergarten.albums != None:
                        # print self.currentKindergarten.albums
                        pass
                    self.face_operations = FaceOperations(self.currentKindergarten, self.PATH, self.database,
                                                          self.quality)

            except Exception as ex:

                logging.warning(ex.message)
                self.init_AFEKA_kindergarten()

    # Get children's names from the database
    def getChildrenNames(self):
        if self.currentKindergarten != None:
            childrenNames = self.database.getChildrenNames(self.currentKindergarten.id)
            return childrenNames
        else:
            return None

    # Delete child id from database
    def deleteChild(self, childId):
        for ch in self.currentKindergarten.getChildList():
            if ch.id == childId:
                self.currentKindergarten.getChildList().remove(ch)
        return self.database.deleteChild(childId)

    #####################################################################

    # Testing mode -> train the model with trainSet of images
    def trainTheSystem(self):
        # create list for training set
        IMAGES_PATH = self.PATH + 'PythonServer/trainSet/'
        train_images_list = os.listdir(IMAGES_PATH)
        num_of_images = len(train_images_list)
        currnt_image_number = 0

        images_list = []

        for img_name in train_images_list:
            file_name = IMAGES_PATH + img_name
            image = ImageObject(img_name, file_name)
            self.face_operations.face_detection_openface(image)
            self.face_operations.face_importance(image)
            # self.get_area(image)
            self.face_operations.face_characterization(image, self.network_model)
            faces = image.get_faces()
            images_list.append(image)
            if currnt_image_number == 9:
                childName = 'demo'
                self.MyExif.writeToImageExif(childName, image)
                break
            currnt_image_number += 1
        # create model
        self.face_operations.face_recognition(images_list, self.currentKindergarten, True)
        # self.start_menu()

    def trainSystem(self):
        PATH = self.PATH + 'PythonServer/tagged_images/'
        train_images_list = os.listdir(PATH)
        num_of_images = len(train_images_list)
        currnt_image_number = 0

        images_list = []
        for img_name in train_images_list:
            file_name = PATH + img_name
            try:
                image = ImageObject(img_name, file_name)
                self.face_operations.face_detection_openface(image)
                self.face_operations.face_importance(image)
                self.face_operations.face_characterization(image, self.network_model)
                faces = image.get_faces()
                child_name = img_name.split('.')[0]
                child = self.currentKindergarten.getChildByFirstName(child_name)
                child_tag = child.tag
                image.tag = child_tag
                child_name_and_tag = "Name : " + child_name + ", Tag : " + str(child_tag)
                image.write_exif(child_name_and_tag)
                # self.writeToImageExif(child_name_and_tag, file_name)
                currnt_image_number += 1
                images_list.append(image)
            except Exception as ex:
                print(img_name)
        self.face_operations.face_recognition(images_list, self.currentKindergarten, True)

    def perform_face_functions(self, image_object):

        detected_faces = self.face_operations.face_detection_openface(image_object)

        """ if we detected a face on a tagged image we will move it to 'tagged_images' folder """
        if image_object.author['child_of_author']['tagged'] is not 0 and detected_faces[
            'num_of_detected_faces'] is not 0:
            shutil.move(image_object.path, self.PATH + 'PythonServer/tagged_images/')
            # when we add a photo to tagged images we want to re-train the system first
            return

        detected_faces = self.face_operations.face_importance(image_object, detected_faces)
        self.face_operations.face_characterization(image_object, self.network_model)
        return

    def drop_low_quality_images(self):
        quality = self.quality
        quality.filter_low_quality_images()

    def process_images_for_statistics_measures(self):
        BEFORE_IMAGES_PATH = self.PATH + 'PythonServer/full_images_data_set_with_exif_for_before_images_folder/'
        list_of_images_in_directory = os.listdir(BEFORE_IMAGES_PATH)
        num_of_images = len(list_of_images_in_directory)
        number_of_images_with_detected_faces = 0

        if num_of_images > 0:

            for current_image_number, img_name in enumerate(list_of_images_in_directory):

                try:
                    print("Processing image " + str(current_image_number + 1) + " out of " + str(num_of_images) + " :")

                    # Step 1 : get the image
                    image_source_path = BEFORE_IMAGES_PATH + img_name

                    # check if file is a valid image
                    if not self.quality.check_if_file_is_valid_image(img_name):
                        os.remove(image_source_path)
                        continue

                    number_of_detected_faces_for_current_image = self.face_operations.measure_statistics_of_dlib_face_detector(
                        image_source_path)

                    if number_of_detected_faces_for_current_image > 0:
                        print str(number_of_detected_faces_for_current_image) + ' faces detected for image ' + str(
                            current_image_number + 1)
                        number_of_images_with_detected_faces += 1
                    else:
                        print 'no faces detected for image ' + str(current_image_number + 1) + ' : ' + img_name

                except Exception as ex:
                    print ex
                    logging.error("Couldn't process image number " + str(current_image_number) + " : " + ex.message)

            print 'only ' + str(
                number_of_images_with_detected_faces) + ' images were detected with faces out of ' + str(
                num_of_images) + ' images with faces'
            print str(
                float(number_of_images_with_detected_faces) / float(num_of_images)) + '% dlib face detection accuracy'


    def process_images_for_quality_test(self):

        BEFORE_IMAGES_PATH = self.PATH + 'PythonServer/image_quality_test/'


        list_of_images_in_directory = os.listdir(BEFORE_IMAGES_PATH)
        num_of_images = len(list_of_images_in_directory)
        image_objects_list = []

        if num_of_images > 0:

            print "\n#############################   STARTING TO PROCESS IMAGES   ##############################"

            for current_image_number, img_name in enumerate(list_of_images_in_directory):

                try:

                    print "\n###########################################################################################"
                    print("\nProcessing image " + str(current_image_number + 1) + " out of " + str(
                        num_of_images) + " :\n")

                    """ STEP 1 : GET IMAGE SOURCE PATH """
                    image_source_path = BEFORE_IMAGES_PATH + img_name

                    """ CHECK IF THE FILE IS A VALID IMAGE FILE """
                    if not self.quality.check_if_file_is_valid_image(img_name):
                        os.remove(image_source_path)
                        continue

                    self.face_operations.face_detection_openface_for_quality_test(image_source_path)


                except Exception as ex:
                    print ex
                    logging.error("Couldn't process image number " + str(current_image_number) + " : " + ex.message)



    def process_images(self):

        BEFORE_IMAGES_PATH = self.PATH + 'PythonServer/before_images/'
        AFTER_IMAGES_PATH = self.PATH + 'PythonServer/after_images/'
        NO_FACE_DETECTED_IMAGES = self.PATH + 'PythonServer/no_face_detected_images/'
        TAGGED_IMAGES_PATH = self.PATH + 'PythonServer/tagged_images/'

        initial_num_of_tagged_images = len(os.listdir(TAGGED_IMAGES_PATH))

        """Filter IN BULK Bad Quality Imaged"""
        # self.drop_low_quality_images() -> WE DESCIDED NOT TO CHECK QUALITY OF ENTIRE IMAGE - ONLY FACES

        list_of_images_in_directory = os.listdir(BEFORE_IMAGES_PATH)
        num_of_images = len(list_of_images_in_directory)
        image_objects_list = []

        if num_of_images > 0:

            print "\n#############################   STARTING TO PROCESS IMAGES   ##############################"

            for current_image_number, img_name in enumerate(list_of_images_in_directory):

                try:

                    print "\n###########################################################################################"
                    print("\nProcessing image " + str(current_image_number + 1) + " out of " + str(
                        num_of_images) + " :\n")

                    """ STEP 1 : GET IMAGE SOURCE PATH """
                    image_source_path = BEFORE_IMAGES_PATH + img_name

                    """ CHECK IF THE FILE IS A VALID IMAGE FILE """
                    if not self.quality.check_if_file_is_valid_image(img_name):
                        os.remove(image_source_path)
                        continue

                    """ CHECK QUALITY OF ENTIRE IMAGE => DROP LOW QUALITY (WE DESCIDED NOT TO CHECK QUALITY OF ENTIRE IMAGE - ONLY FACES) """
                    # self.quality.filter_quality_of_single_image(image_source_path, current_image_number, img_name, 0.42)

                    """ STEP 2 : RETRIEVE ENCODED EXIF DATA FROM IMAGE """
                    dict_image_data = self.MyExif.get_image_EXIF_data(image_source_path)
                    print "\nRetrieving primary exif from image:\n\n" + str(dict_image_data) + "\n"

                    """ STEP 3 : CREATE IMAGE-OBJECT INSTANCE FROM REAL PHOTO """
                    image_object = ImageObject(img_name, image_source_path)

                    """ STEP 4 : APPLY ATTRIBUTES FROM INITIAL EXIF TO NEWLY CREATED IMAGE-OBJECT INSTANCE """
                    image_object.apply_attributes_from_exiv(dict_image_data)

                    # Step 5 : perform Face Detection, Importance and Characterization
                    #          Update attributes of ImageObject while running Face functions
                    """ STEP 5 : PERFORMING FACE FUNCTIONS
                     - DETECT FACES IN IMGAE
                     - CHECK SIZE AND QUALITY OG EACH DETECTED FACE IN CURRENT IMAGE
                     - DROP FACES THAT ARE TOO SMALL OR WITH BAD QUALITY
                     - SORT GOOD QUALITY FACES BY IMPORTANCE
                     - FACE CHARACTERIZATION => CREATE A REPRESENTATION OF EACH FACE-RECT
                    """
                    self.perform_face_functions(image_object)
                    # faces = image_object.get_faces()

                    """ CHECK IF FACES WERE DETECTED IN CURRENT IMAGE => IF YES ADD TO IMAGE-OBJECT LIST"""
                    if image_object.number_of_detected_faces > 0:
                        image_objects_list.append(image_object)
                    else:
                        self.MyExif.write_updated_data_to_image_exif(image_object, image_source_path)
                        copyfile(image_source_path, NO_FACE_DETECTED_IMAGES + img_name)
                        shutil.move(image_source_path, AFTER_IMAGES_PATH)
                        continue

                    """ STEP 6 : ENCODE NEW DATA FROM FACE FUNCTIONS INTO IMAGE EXIF """
                    self.MyExif.write_updated_data_to_image_exif(image_object, image_source_path)

                except Exception as ex:
                    print ex
                    logging.error("Couldn't process image number " + str(current_image_number) + " : " + ex.message)


            num_of_tagged_images = len(os.listdir(TAGGED_IMAGES_PATH))

            if num_of_tagged_images > initial_num_of_tagged_images:
                #train system
                pass

            """ 
            STEP 7 : RUN FACE RECOGNITION ON ALL FROCCESSED IMAGE-OBJECTS => 
            WHEN A FACE RECT IS CLASSIFIED (GIVEN TAG) WITH A CERTAIN CREDABILITY THAT 
            IS HIGHER THEN A DEFINED CREDABILITY THRESHOLD WE ADD THE TAG TO THE FACE 
            iF A FACE IS TAGGED => ENCODE INTO IMAGE EXIF

            [ WE DON'T PERFORM THIS STEP INSIDE THE FOR-LOOP BECAUSE WE NEED THE ENTIRE LIST OF IMAGES ]

            """
            self.face_operations.face_recognition(image_objects_list, self.currentKindergarten)

            """ 
            STEP 8 : 

            ONLY AFTER PERFORMING FACE DETECTION -> QUALITY -> IMPORTANCE -> CHARACTERIZATION -> CLASSIFICATION -> TAGGING 

            AND ENCODING ALL DATA INTO EXIF OF IMAGE THEY ARE READY TO BE EXPORTED FROM THE SERVER
            """
            for image_object in image_objects_list:
                image_source_path = image_object.path

                print "\nFinal image exif after prediction tagging:\n\n" + str(image_object.exif) + "\n\n"
                self.MyExif.write_updated_data_to_image_exif(image_object,image_source_path)
                # mark pace rectangles and tags
                # self.face_operations.mark_face_rects_and_tag_of_each_rect(image_object)

                """ STEP 7 : MOVE PROCESSED IMAGES WITH ENCODED EXIF INTO 'AFTER_IMAGES' FOLDER => READY TO EXPORT """  # Step 7 : move processed images to 'after_images' folder
                shutil.move(image_source_path, AFTER_IMAGES_PATH)

            # for img in image_objects_list:
            #    print "\n" + str(img.exif)

        # Do

        # Finalize data about image
        # Send data in json format to Ruslan's DB

    def save_logic_state(self):
        self.database.saveKindergarten(self.currentKindergarten)
        self.database.saveChildren(self.currentKindergarten)
        self.database.saveSchedule(self.currentKindergarten)
        self.database.saveAlbumes(self.currentKindergarten)
        self.database.closeConnection()

    def start(self):
        logging.basicConfig(filename='pythonserver.log', level=logging.DEBUG)
        # self.get_EXIF_quality_measurements()
        try:
            #self.process_images()
            # self.process_images_for_statistics_measures()
            self.process_images_for_quality_test()
            #self.MyExif.write_untagged_pseudo_image_exif_authors_to_all_images()
        except Exception as ex:
            print ex.message
        self.save_logic_state()
        return

