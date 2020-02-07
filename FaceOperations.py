from asynchat import fifo

import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError
from dropbox.sharing import AccessLevel
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
from MyEXIF import MyExif


class FaceOperations(object):

    def __init__(self, kindergarten, path, db, quality):
        self.currentKindergarten = kindergarten
        self.PATH = path
        self.database = db
        self.quality = quality


    """
      face_detection function gets an image object.
      This function is an implementaion of face detection algorithm using opencv library.
      The results of this implementaion were worse in comparison to openface implementaion.
      """


    def face_detection(self, image):
        """
        :type image: ImageObject
        """
        classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        tmp = cv2.imread(image.path)
        if tmp is not None:
            gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            faces = classifier.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            # write number of faces to xmp
            image.write_xmp("RectsNum", str(len(faces)))

            countRects = 1
            for (x, y, w, h) in faces:
                crop_img = tmp[y:y + h, x:x + w]
                image_path = self.get_path_without_extension(image.path) + "-" + str(countRects) + ".jpg"
                cv2.imwrite(image_path, crop_img)
                # add rect to image's faces
                sub_image = ImageObject(self.get_file_name(image.path) + "-" + str(countRects) + ".jpg", image_path)
                rect = Rectangle(x, y, h, w, 0, sub_image)
                area = self.area_func(rect, h, w)
                if area > 0.01:
                    image.add_face_to_image(rect)
                    # write data to image xmp
                    data = "Start:(" + str(rect.startX) + "," + str(rect.startY) + ") End:(" + str(
                        rect.startX + rect.width) + "," + str(
                        rect.startY + rect.height) + ")"

                    image.write_xmp("Rect" + str(countRects), data)
                    countRects += 1


    """
    face_detection_openface function gets an image object
    The function specify the faces in the image using dlib frontal detector. - Not working
    The function disqualify rects of faces by their height and width
    Afterwards, writing all the data to metadata using xmp
    """

    def measure_statistics_of_dlib_face_detector(self, path):
        detector = dlib.get_frontal_face_detector()
        opencv_img_obj = cv2.imread(path)
        faces = detector(opencv_img_obj, 1)
        number_of_faces = len(faces)
        return number_of_faces

    def face_detection_openface_for_quality_test(self,image_source_path):
        detector = dlib.get_frontal_face_detector()
        # maps image to matrix with values
        opencv_img_obj = cv2.imread(image_source_path)
        height,width = opencv_img_obj.shape[:2]
        faces = detector(opencv_img_obj, 1)
        number_of_faces = len(faces)


        if number_of_faces == 0:  # if len(faces) == 0:
            print "no face detected for image " + image_source_path
            return

        countRelevantRects = 0
        for i, d in enumerate(faces):
            countRects = i + 1
            # get rect dimentions
            x = d.left()
            y = d.top()
            w = d.right() - x
            h = d.bottom() - y

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if w > 190 and h > 190:
                countRelevantRects += 1
                relative_size_of_face_to_image = round(float(w * h) / float(width * height), 5)
                current_rect_image_name = self.get_file_name(image_source_path) + "-" + str(countRects) + ".jpg"
                current_rect_image_path = self.PATH + "PythonServer/detected_faces_for_quality/" + current_rect_image_name
                # add rect to image's faces
                sub_image = ImageObject(self.get_file_name(image_source_path) + "-" + str(countRects) + ".jpg",
                                        current_rect_image_path)
                rect = Rectangle(x, y, h, w, 0, sub_image, image_source_path)
                rect.id = countRelevantRects
                # move relevant face rect to 'res' directory:
                self.save_sub_image_quality_test(image_source_path, rect, countRelevantRects)
                # measure quality of current rect
                current_face_quality = self.quality.filter_quality_of_single_face(current_rect_image_path,
                                                                                  countRelevantRects,
                                                                                  current_rect_image_name, 0)

        return

    def face_detection_openface(self, image):
        """
        if self.check_if_file_is_valid_image(image) is False: # 0 - valid; 1 - not valid
            print("Image has wrong extension!")
            return
            """

        # dlib.get_frontal_face_detector()(cv2.imread(image.path),1)

        detector = dlib.get_frontal_face_detector()
        # maps image to matrix with values
        opencv_img_obj = cv2.imread(image.path)
        image.height, image.width = opencv_img_obj.shape[:2]
        faces = detector(opencv_img_obj, 1)
        number_of_faces = len(faces)

        detected_faces = {'num_of_detected_faces': 0, 'faces_details': []}

        if number_of_faces == 0:  # if len(faces) == 0:
            """
            try:
                image.image_tag = self.get_images_tags(image.path, image.name)
            except Exception as ex:
                print(ex)
                #image_tag is unicode
            image.write_exif(image.image_tag)
            """
            return detected_faces

        # Tag extraction
        """
        try:
            image.image_tag = self.get_images_tags(image.path, image.name)
        except Exception:
            print("Could not get tags")
        """
        # print "tag " + str(image.image_tag)

        # write number of faces to xmp
        # image.write_xmp("RectsNum", str(len(faces)))
        try:
            image.write_xmp("RectsNum", str(number_of_faces))
        except:
            image.write_xmp_vika('RectsNum', str(number_of_faces))
        # xmp_data = image.get_xmp('RectsNum')
        file_path = image.path
        xmpfile = XMPFiles(file_path=image.path, open_forupdate=True)
        # xmp = xmpfile.get_xmp()
        xmp = file_to_dict(file_path)

        # get image dimentions
        height = image.height
        width = image.width
        countRelevantRects = 0
        for i, d in enumerate(faces):
            countRects = i+1
            # get rect dimentions
            x = d.left()
            y = d.top()
            w = d.right() - x
            h = d.bottom() - y

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if w > 190 and h > 190:
                countRelevantRects += 1
                relative_size_of_face_to_image = round(float(w*h)/float(width*height),5)
                current_rect_image_name = self.get_file_name(image.path) + "-" + str(countRects) + ".jpg"
                current_rect_image_path = self.PATH + "PythonServer/detected_faces_folder/" + current_rect_image_name
                # add rect to image's faces
                sub_image = ImageObject(self.get_file_name(image.path) + "-" + str(countRects) + ".jpg", current_rect_image_path)
                rect = Rectangle(x, y, h, w, 0, sub_image, image.path)
                rect.id = countRelevantRects
                # move relevant face rect to 'res' directory:
                self.save_sub_image(image, rect, countRelevantRects)
                # measure quality of current rect
                current_face_quality = self.quality.filter_quality_of_single_face(current_rect_image_path,
                                                                                   countRelevantRects,
                                                                                   current_rect_image_name, 0)
                detected_faces['faces_details'].append(
                    {'id': countRelevantRects, 'relative_size_of_face_to_image': relative_size_of_face_to_image,
                     'face_quality': current_face_quality, 'is_face_tagged': 0, 'tag_number': 0})
                image.add_face_to_image(rect)
                # write data to image xmp
                data = "Start:(" + str(rect.startX) + "," + str(rect.startY) + ") End:(" + str(
                    rect.startX + rect.width) + "," + str(
                    rect.startY + rect.height) + ")"
                rect.id = countRelevantRects
                image.write_xmp("Rect" + str(countRelevantRects), data)
                image.rects['Rect' + str(countRelevantRects)] = data
        detected_faces['num_of_detected_faces'] = countRelevantRects
        return detected_faces



    """
    save_sub_image function gets an image object, rect object and index.
    The function take the specific area from the original image and
    save it as a new image with the index.
    The index symbolize the importance of the face.
    """


    def save_sub_image(self, image, rect, index):
        img = cv2.imread(image.path)
        crop_img = img[rect.startY:rect.startY + rect.height, rect.startX:rect.startX + rect.width]
        image_path = self.get_file_name(image.path) + "-" + str(index) + ".jpg"
        cv2.imwrite(self.PATH + "PythonServer/detected_faces_folder/" + image_path, crop_img)


    def save_sub_image_quality_test(self, image_source_path, rect, index):
        img = cv2.imread(image_source_path)
        crop_img = img[rect.startY:rect.startY + rect.height, rect.startX:rect.startX + rect.width]
        image_path = self.get_file_name(image_source_path) + "-" + str(index) + ".jpg"
        cv2.imwrite(self.PATH + "PythonServer/detected_faces_for_quality/" + image_path, crop_img)


    """
    face_importance function gets an image object.
    The function calculates each rect (face) importance according to
    combination of the distance from the center of the image to the center of the 
    rect and the rect area. 
    I calculate the credibility of the image according to the number of the rects and their importance.
    I sort the rects according to their importance and write all the data to the
    metadata using xmp.
    """


    def face_importance(self, image,detected_faces):
        if len(image.faces) == 0:
            print "no face identified"
            return
        dist_weight = 0.65
        area_weight = 0.35
        rects = list(image.faces)
        # read image height and width from exif
        width = image.width
        height = image.height

        print image.path
        for rect in rects:
            dist = self.dist_func(rect, height, width)
            area = self.area_func(rect, height, width)
            rect.importance = dist_weight * dist + area_weight * area
            rect.area = rect.height * rect.width
        sorted_rects = sorted(rects, key=lambda x: x.importance, reverse=True)
        sum_all_rects_importance = self.calculate_importance_sum(sorted_rects)
        count_rects = 1
        for rect in sorted_rects:
            data = "Start:(" + str(rect.startX) + "," + str(rect.startY) + ") End:(" + str(
                rect.startX + rect.width) + "," + str(
                rect.startY + rect.height) + ")"
            #self.save_sub_image(image, rect, count_rects)
            rect.normalized_importance = float(rect.importance / sum_all_rects_importance)
            lst = detected_faces['faces_details']
            for l in lst:
                if l['id'] is (rect.id):
                    l['face_importance'] = count_rects
                    l['relative_position_of_face_to_image'] = round(rect.normalized_importance,5)
            detected_faces['faces_details'] = lst
            if count_rects == 1:
                image.write_xmp("MostImportant", data)
                # in case most important face was not recognized we tag it as the child of the author
                if image.image_tag == 0:
                    image.image_tag = image.child_of_author_tag
            image.write_xmp("Importance" + str(count_rects), data)
            count_rects += 1
        image.faces = sorted_rects[:]
        image.detected_faces = detected_faces
        image.word_details.credibility = sorted_rects[0].normalized_importance
        # image.write_exif(image.image_tag)
        self.update_histogram(image, self.currentKindergarten)
        self.update_child_recognition(self.currentKindergarten)
        return detected_faces


    """
    update_child_recognition function gets a kindergarten object
    The function update the recognition field which reflect the familiarity
    of the histogram with the specific child tag
    """

    def update_child_recognition(self, kindergarten):
        for child in self.currentKindergarten.children:
            if self.currentKindergarten.images_no == 0:
                child.recognition = 0
            else:
                child.recognition = float(
                    self.currentKindergarten.histogram[int(child.tag)] / self.currentKindergarten.images_no)


    """
    calculate_importance_sum function gets list of rects sorted by their importance.
    The function calculates the sum of importance of each rect in the image.
    Return the sum of importance
    """


    def calculate_importance_sum(self, sorted_rects):
        sum_all_rects_importance = 0
        for my_rect in sorted_rects:
            sum_all_rects_importance += my_rect.importance
        return sum_all_rects_importance


    """
    dist_func function gets an rect object, rect height and width.
    The function calculates the distance between the image center and the rect center.
    Return normalized distance (value between 0 to 1)
    """


    def dist_func(self, rect, height, width):
        start_x_rect = rect.startX
        start_y_rect = rect.startY

        middle_x = start_x_rect + float(rect.width) / 2
        middle_y = start_y_rect + float(rect.height) / 2

        middle_image_x = float(width) / 2
        middle_image_y = float(height) / 2
        dist_from_center = math.sqrt(
            (middle_x - middle_image_x) * (middle_x - middle_image_x) + (middle_y - middle_image_y) * (
                    middle_y - middle_image_y))

        max_dist = float(math.sqrt(height * height + width * width)) / 2

        # distance normalization
        res = 1 - (float(dist_from_center) / float(max_dist))
        return res


    """
    area_func function gets an rect object, rect height and width.
    The function calculates the area of the rect.
    Return normalized area (value between 0 to 1)
    """


    def area_func(self, rect, height, width):
        area_rect = rect.width * rect.height
        max_area = height * width
        # area normalization
        res = float(area_rect) / float(max_area)
        return res


    """
    face_characterization function gets an image object and neural network model.
    The function using openface library function in order to align the rect(face) 
    and resize it to 96x96, preparing it to the neural network.
    Afterwards, using Torch library function I calculates the 128 feature vector
    representation of the face and save the changes we made.
    """


    def face_characterization(self, image, network_model):
        try:
            # for face registration
            file_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.join(file_dir, '..', 'models')
            dlib_model_dir = os.path.join(model_dir, 'dlib')
            open_face_model_dir = os.path.join(model_dir, 'openface')
            landmark_map = {'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE}
            landmark_indices = landmark_map['outerEyesAndNose']
            align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
            # for face reduction

            network_model_vika = self.PATH + 'PythonServer/openface.nn4.small2.v1.t7'
            net = openface.TorchNeuralNet(network_model_vika, 96, False)
            # net = openface.TorchNeuralNet(network_model, 96, False)
            if not image.faces is None:
                for face in image.get_faces():
                    face_img = cv2.imread(face.faceImage.path)
                    if not face_img is None:
                        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        if not rgb_img is None:
                            out_rgb = align.align(96, rgb_img, landmarkIndices=landmark_indices,
                                                  skipMulti="Skip images with more than one face.")
                            if not out_rgb is None:
                                img_rep = net.forward(out_rgb)
                                face.update_face_rep(img_rep)
                                bgr_img = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(face.faceImage.path, bgr_img)
                            else:  # registration failed
                                rgb_img = cv2.resize(rgb_img, (96, 96))
                                img_rep = net.forward(rgb_img)
                                face.update_face_rep(img_rep)
                                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(face.faceImage.path, bgr_img)
        except:
            print('face_characterization exception')


    """
    update_histogram function gets an image object and kindergarten object.
    The function append each 1 to the appropriate cell that match the image tag 
    only if the credibility of the image is 1. 
    """


    def update_histogram(self, image, kindergarten):
        try:
            tag_number = int(image.image_tag)
        except:
            child_full_name = str(image.image_tag)
            first_name = child_full_name.split(' ')[0]
            last_name = child_full_name.split(' ')[1]
            tag_number = self.database.getChildTagByName(self.currentKindergarten.id, first_name, last_name)
            image.image_tag = tag_number
        if image.word_details.credibility == 1:
            kindergarten.histogram[tag_number] += 1
            kindergarten.images_no += 1


    """
    face_recognition function gets a list of images and kindergarten object.
    The function append each image to the kindergarten images list and build 
    svm model according to this list. 
    Afterwards, I get predict tag for each image in the images list that i got (not the kindergarten list).
    """


    # def face_recognition(self, images, kindergarten, train):
    def face_recognition(self, images, kindergarten):
        for img in images:
            if len(img.faces) != 0:
                #for rect in img.faces:
                rect = img.faces[0]
                if not rect.faceRep == []:
                    img.treshold = 0.2
                    if img.word_details.credibility >= img.treshold:
                        kindergarten.images.append(list(rect.get_rep()))
                        kindergarten.labels.append(int(img.image_tag))
        clf = SVC(C=1, kernel='rbf', gamma=2, probability=True)
        try:
            i = kindergarten.images
            l = kindergarten.labels

            x = i
            y = (len(l),)
            if (len(x) == 1):
                return
            #clf.fit(x, y)
            clf.fit(kindergarten.images, kindergarten.labels)
            kindergarten.clf = clf
            for img in images:
                # self.predict_faces_in_image(img, kindergarten, train)
                self.predict_faces_in_image(img, kindergarten)

        except Exception as ex:
            print(ex)
            print "Only one tag"


    """
    get_features function gets 128 feature vector and the amount of features to take.
    The function creates new vector that contains number of cells like the amount we got.
    Return this new vector of features.
    """


    def get_features(self, rep, amount):
        new_rep = []
        for i in range(amount):
            new_rep.append(rep[i])
        return new_rep


    def addImageToChildAlbum(self, childTag, imageName):
        return self.currentKindergarten.addImageToAlbum('tag ' + str(childTag), imageName)

    """
      get_path_without_extension function gets image path
      Return the all the image path without extension
      """

    def get_path_without_extension(self, path):
        file_path = path.split(".")[0]
        return file_path

    """
    get_file_name function gets image path 
    Return only the name of the file
    """

    def get_file_name(self, path):
        file_path = path.split(".")[0]
        file_arr = file_path.split("/")
        file_name = file_arr[len(file_arr) - 1]
        return file_name


    def mark_face_rects_and_tag_of_each_rect(self, image_object):

        from matplotlib import pyplot as plt
        img = cv2.imread(image_object.path)

        dict = image_object.exif

        for rect in image_object.faces:
            bla = rect
            try:
                cv2.rectangle(img,(rect.startX,rect.startY),(rect.startX+rect.width,rect.startY+rect.height),(255,255,224),35)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,'CHILD NAME',(rect.startX + (rect.width/2),rect.startY+rect.height),font,(255,255,255),3)

            except Exception as ex:
                logging.warning(ex.message)
        return


####################################################################################3

###  Karin image process functions ################################

    """
    get_images_tags function gets image path.
    The function read the ChildTag label from the exif metadata.
    Return the tag according to the field
    """
    def get_images_tags(self,path,name):
        metadata = pyexiv2.ImageMetadata(path)
        metadata.read()
        #raises error here
        dirname = '/'.join(path.split('/')[:-1]) + '/'
        #with open(os.path.join(dirname,name)) as fd:
            #json_data = json.load(fd)
        data = metadata['Exif.Photo.UserComment'].value
        try:
            userdata = json.loads(metadata["Exif.Photo.UserComment"].value)
        except Exception as ex:
            print(ex)
            return data

        #userdata = json.loads(metadata['Exif.Photo.UserComment'].value, encoding='ASCII')
        return userdata['ChildTag']


######################################################################################################
    """
    predict_faces_in_image function gets an image object and kindergarten object.
    The function predict for each image the tag which symbolized the child.
    Also we calculate the confidence value of the prediction and normalized it 
    by the sum of the confidences.
    Only if the confidence is bigger then 0.2 we append the image to the all_rects_histogram object.
    """
    def predict_faces_in_image(self,image, kindergarten,train = False):
        pred_images = []
        sum_conf = 0
        for rect in image.faces:
            if not rect.faceRep == []:
                pred_images.append(list(rect.get_rep()))
                kindergarten.iteration_images.append(list(rect.get_rep()))
                rect.predict_tag = kindergarten.clf.predict(pred_images)
                predictions_prob = kindergarten.clf.predict_proba(pred_images).ravel()
                for pro in predictions_prob:
                    sum_conf += pro
                max_value_index = np.argmax(predictions_prob)
                confidence = predictions_prob[max_value_index]
                rect.normalized_prob = float(confidence / sum_conf)
                if rect.normalized_prob > 0.2:
                    lst = image.exif['detected_faces']['faces_details']
                    for l in lst:
                        if l['id'] is rect.id:
                            l['is_face_tagged'] = 1
                            l['tag_number'] = rect.predict_tag[0]
                    image.exif['detected_faces']['faces_details'] = lst
                    kindergarten.all_rects_histograms[rect.predict_tag] += 1
                    childAlbum = self.addImageToChildAlbum(rect.predict_tag, rect.faceImage.path.split('/')[-1].split('-')[0])
                pred_images = []

    # functions I don't use anymore ########################################################

    """
    feature_extraction function gets an image object.
    I used this function when we thought to make the identification by calculating 
    distance from average feature vector.
    """
    def feature_extraction(self,image):
        # get parent name from exif
        child_first_name = "karin"
        child_last_name = "levi"
        for child in self.currentKindergarten.getChildList():
            if (child.first_name).lower() == (child_first_name).lower() and (child.last_name).lower() == (
            child_last_name).lower():
                if not image.faces == None and not len(image.faces[0].faceRep) == 0:
                    child.add_to_feature_vector(image.faces[0].faceRep)
                    self.update_average_vector(child)
                    self.update_variance_and_std(child)

    """
    update_average_vector function gets a child object.
    I used this function to append values to average feature vector and update it.
    """
    def update_average_vector(self,child):
        # calculate the avarage feature vector of each child
        n = len(child.feature_vector)
        if n == 1:
            child.avg_feature_vector = [child.feature_vector[0][elem] for elem in range(len(child.feature_vector[0]))]
        else:
            prev_feature_vector = [child.avg_feature_vector[elem] for elem in range(len(child.avg_feature_vector))]
            for i in range(len(child.feature_vector[n - 1])):  # from 0 to 127
                child.avg_feature_vector[i] = (float((n - 1)) / n) * prev_feature_vector[i] + (1 / float(n)) * \
                                              child.feature_vector[n - 1][i]

    """
    update_variance_and_std function gets a child object.
    I used this function to calculate the vectors of variance and std of each child
    when the average feature vector is changing.
    """
    def update_variance_and_std(self,child):
        n = len(child.feature_vector)
        if n == 1:
            child.variance = [0 for elem in range(len(child.feature_vector[0]))]
            child.std = [0 for elem in range(len(child.feature_vector[0]))]
        else:
            prev_varianace = [child.variance[elem] for elem in range(len(child.variance))]
            for i in range(len(child.avg_feature_vector)):
                child.variance[i] = (float((n - 1)) / n) * prev_varianace[i] + (1 / float(n)) * (
                        child.feature_vector[n - 1][i] - child.avg_feature_vector[i]) * (
                                            child.feature_vector[n - 1][i] - child.avg_feature_vector[i])
                child.std[i] = math.sqrt(child.variance[i])
                # print "variance = %.9f" % child.variance[i]
                # print "std = %.9f" % child.std[i]

    """
    remove_outlayer function gets a child object.
    I used this function to remove values that aren't match the defined formula.
    """
    def remove_outlayer(self,child):
        for i in range(len(child.feature_vector)):
            if (1 / 128) * self.get_dist_vector(child.feature_vector[i], child.avg_feature_vector, child.std) >= 9:
                child.feature_vector.remove(child.feature_vector[i])

    """
    get_dist_vector function gets child feature vector, the average feature vector
    and the std.
    The function calculate the distance between the two vectors and accumulate it in sum.
    """
    def get_dist_vector(self,one_feature_vector, avg_feature_vector, std):
        sum = 0
        for i in range(len(avg_feature_vector)):
            sum += (one_feature_vector[i] - avg_feature_vector[i]) / (float(std[i]))
        return sum

    """
    calculate_dist_test function gets a kindergarten object and image object.
    I used this function in order to build distances list of foreach child in the kindergarten.
    """
    def calculate_dist_test(self,kindergarten, image):
        distances = []
        for child in kindergarten.children:
            for i in range(len(child.avg_feature_vector)):
                if not image.faces[0].faceRep == []:
                    print child.avg_feature_vector[i]
                    print image.faces[0].faceRep[i]
                    distances.append(child.avg_feature_vector[i] - image.faces[0].faceRep[i])
        print distances


    # QA functions
    def test_svm(self,path):
        dict = {}
        labels = []

        print "read file..."
        dict = self.read_from_file(path)
        print "compute train set..."
        reps_train = self.create_from_list(dict, 18, labels)
        print len(reps_train)
        print "compute test set..."
        reps_test = self.create_test_set(dict, 18)
        print len(reps_test)
        print "predict..."
        self.check_svm(reps_train, labels, reps_test)

    def check_svm(self,reps_train, labels, reps_test):
        pred_images = []
        clf = SVC(C=1, kernel='rbf', gamma=2, probability=True)
        clf.fit(reps_train, labels)
        for rep in reps_test:
            pred_images.append(list(rep))
        pred = clf.predict(pred_images)

        print pred

    def create_from_list(self,dict, num, labels):
        reps_arr = []
        histogram = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0]  # array in size 31 - indexes 0-30
        for key in dict:
            if histogram[dict[key]] != num:
                reps_arr.append(self.face_reps(key))
                histogram[dict[key]] += 1
                labels.append(dict[key])
        return reps_arr

    def create_test_set(self,dict, num):
        reps_arr = []
        histogram = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0]  # array in size 31 - indexes 0-30
        for key in dict:
            if histogram[dict[key]] < (num + 1):
                histogram[dict[key]] += 1
            else:
                if histogram[dict[key]] == (num + 1):
                    print key
                    print dict[key]
                    reps_arr.append(self.face_reps(key))
                    histogram[dict[key]] += 1
        return reps_arr

    def read_from_file(self,path):
        my_dict = {}
        folder_path = self.PATH + "PythonServer/res3/"
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                res = line.split(" ");
                my_dict[folder_path + res[0]] = int(res[1])
            f.close()
        return my_dict

    def face_reps(self,name):
        img_rep = []
        img = ImageObject(name, name)
        # for face registration
        file_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.join(file_dir, '..', 'models')
        dlib_model_dir = os.path.join(model_dir, 'dlib')
        open_face_model_dir = os.path.join(model_dir, 'openface')
        landmark_map = {'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE}
        landmark_indices = landmark_map['outerEyesAndNose']
        align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
        # for face reduction
        net = openface.TorchNeuralNet(self.network_model, 96, False)
        face_img = cv2.imread(img.path)
        if not face_img is None:
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            if not rgb_img is None:
                out_rgb = align.align(96, rgb_img, landmarkIndices=landmark_indices,
                                      skipMulti="Skip images with more than one face.")
                if not out_rgb is None:
                    img_rep = net.forward(out_rgb)
                else:  # registration failed
                    rgb_img = cv2.resize(rgb_img, (96, 96))
                    img_rep = net.forward(rgb_img)

        return img_rep

