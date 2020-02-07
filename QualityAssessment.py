import imutils as imutils
import numpy as np
import math
import cv2
import os
from scipy.signal import convolve2d
import shutil
import logging
from math import e
from scipy import misc
from skimage.restoration import estimate_sigma
from numba import njit


class Quality(object):

    def __init__(self, path):
        self.PATH = path
        self.BAD_QUALITY_IMAGES_PATH = self.PATH + 'PythonServer/bad_quality_images/'

    ##########################################################################################################
    ########################################### HELPER FUNCTIONS   ###########################################

    """
        what: check if file is an image 
        how: checking if file extention is one of [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        file is an image :return True
        file is not an image :return False
        """

    def check_if_file_is_valid_image(self, image):
        valid_image_extension = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        valid_image_extension = [item.lower() for item in valid_image_extension]
        try:
            file_extension = "." + image.path.split(".")[1]
        except:
            file_extension = '.' + image.split('.')[1]
        if file_extension.lower() not in valid_image_extension:
            return False
        else:
            return True

    def read_image_with_opencv(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is not None:
                # print "sucessfully read image with opencv"
                return img
            else:
                print "failed to read image with opencv"
        except Exception as ex:
            print ex.message

    def normalize_value_in_range_0_and_1(self, x):
        """
        :param x: float (initial numeric value)
        :return: float

        Rescaling a numeric value to range [0,1]
        Using Logistic Transform:

        normalized_value = 1 / ( 1 + exponent(-x) )

        where exp(-x) is same as e^(-x)
        e is Euler's Number
        """

        # we would normalize a value only if it's greater than 1
        if x > 1:
            dominator = e ** (-x)
            normalized_value = 1 / (1 + e ** (-x))
            return normalized_value
        else:
            return x

    ##########################################################################################################
    ############################## FUNCTIONS TO MEASURE SEPARATE QUALITY FACTORS  ############################

    """
    This function is part of 'Blur' measurement 
    compute the Laplacian of the image and then return the focus
    measure, which is simply the variance of the Laplacian"""

    def variance_of_laplacian(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def get_image_pixles(self, image):

        saturation_sum = 0

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):

                # try:

                # getting original pixel value
                pixel = image[i, j]
                if 0 in pixel:
                    saturation_sum += 1
                elif 1 in pixel:
                    saturation_sum += 1
                elif 225 in pixel:
                    saturation_sum += 1

                # appending pixel to pixels list
                # pixels.append(pixel)

                # except Exception as ex:
                #    print ex.message

        # return pixels
        return saturation_sum

    def get_smoothed_image_brightness_of_pixel(self, pixel):

        if 0 in pixel:
            Ib = 0
        elif 1 in pixel:
            Ib = 1
        elif 225 in pixel:
            Ib = 255
        else:
            Ib = pixel[0]

        return Ib

    def calculate_pixels_saturation_of_image(self, pixels):
        saturation_sum = 0

        for pixel in pixels:
            try:
                Ib = self.get_smoothed_image_brightness_of_pixel(pixel)
            except Exception as ex:
                Ib = -1
                print ex.message

            if Ib > 1 and Ib < 255:
                # saturation_sum += 0
                continue
            elif Ib == 0 or Ib == 1 or Ib == 255:
                saturation_sum += 1
            else:
                print "Couldn't get saturation from pixel"

        return saturation_sum

    def image_colorfulness(self, image):
        # split the image into its respective RGB components
        (B, G, R) = cv2.split(image.astype("float"))

        # compute rg = R - G
        rg = np.absolute(R - G)

        # compute yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B)

        # compute the mean and standard deviation of both `rg` and `yb`
        (rbMean, rbStd) = (np.mean(rg), np.std(rg))
        (ybMean, ybStd) = (np.mean(yb), np.std(yb))

        # combine the mean and standard deviations
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

        # derive the "colorfulness" metric and return it
        return stdRoot + (0.3 * meanRoot)

    """Fast Noise Variance Estimation"""

    def estimate_noise(self, img_path):

        img = self.read_image_with_opencv(img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = img_gray.shape
        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        sigma = np.sum(np.sum(np.absolute(convolve2d(img_gray, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))
        return sigma

    def image_exposure(self, image):

        """
        :param image:
        :return: Qe

        Calculate image exposure using Saturation for each pixel

        Where:

        Qe	- Exposure Quality (dark & bright saturation)
        Si	- Saturation in pixel i
        Ib	- Smoothed image brightness in pixel i
        N   - Number of pixels in an image


        Saturation for each pixel will be measured by:

        If        0 < Ib < 255   =>  Si = 0
        Else if   Ib = 0 or ib = 1 or Ib = 255  =>  Si = 1


        Exposure Quality Formula:

        Qe = 1 - ( Summation(Si) / N)

        """

        width, height = image.shape[:2]
        N = width * height

        saturation_sum = self.get_image_pixles(image)
        frac = saturation_sum / N
        Qe = 1 - (saturation_sum / N)
        return Qe

        # try:
        # pixels = self.get_image_pixles(image)
        # saturation_sum = self.calculate_pixels_saturation_of_image(pixels)
        # try:
        # saturation_sum = self.get_image_pixles(image)
        # except Exception as ex:
        #    print ex.message
        # frac = saturation_sum / N
        # Qe = 1 - (saturation_sum / N)
        # return Qe
        # except Exception as ex:
        # print ex.message

    """
    To copute the 'Blur' of an image:
    load the image, convert it to grayscale, and compute the focus 
    measure of the image using the Variance of Laplacian method """

    def measure_image_blur(self, img_path):
        image = self.read_image_with_opencv(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = self.variance_of_laplacian(gray)
        return fm

    def measure_image_colorfulness(self, image_path):
        image = self.read_image_with_opencv(image_path)
        image = imutils.resize(image, width=250)
        C = self.image_colorfulness(image)
        return C

    def measure_image_exposure(self, image_path):
        image = self.read_image_with_opencv(image_path)
        # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = misc.imread(image_path)
        E = self.image_exposure(image)
        return E

    ##########################################################################################################
    ################################## FUNCTIONS TO MEASURE TOTAL QUALITY   ##################################

    def measure_image_quality_version1_Vika(self, image_source_path):

        try:
            # low noise => noisy image (less 0.5)
            noise = self.estimate_noise(image_source_path)
            print "noise = " + str(noise)
            logging.info("noise = " + str(noise))
        except Exception as ex:
            noise = 0
            logging.error("couldn't measure noise of image")

        try:
            # low blur => blurry image (less 0.20 or 0.25)
            blur = self.measure_image_blur(image_source_path)
            print "blur = " + str(blur)
            logging.info("blur = " + str(blur))
        except Exception as ex:
            blur = 0
            logging.error("couldn't measure blurriness of image")

        try:
            # low colorfulness => least colorful (less 0.10), high colorfulness => most colorful (more 0.80)
            colorfulness = self.measure_image_colorfulness(image_source_path)
            print "colorfulness = " + str(colorfulness)
            logging.info("colorfulness = " + str(colorfulness))
        except Exception as ex:
            colorfulness = 0
            logging.error("couldn't measure colorfulness of image")

        if noise > 0.5 and blur > 0.2 and colorfulness > 0.3:
            print "good quality"
        else:
            print "bad quality"

        # noise_weight
        w1 = 0.25

        # blur_weight
        w2 = 0.5

        # colorfulness_weight
        w3 = 0.25

        quality_measure = noise * w1 + blur * w2 + colorfulness * w3

        print "quality = " + str(quality_measure)
        logging.info("quality = " + str(quality_measure))

        return quality_measure

    def measure_image_quality_version2_Stekel(self, image_source_path):

        try:
            ########## GETTING NOISE ##############
            # low noise => noisy image (less 0.5)
            noise = self.estimate_noise(image_source_path)

            logging.info("noise = " + str(noise))

            ##### NORMALIZING NOISE BETWEEN [0,1] #####
            Qn = self.normalize_value_in_range_0_and_1(noise)
            print "normalized noise = " + str(Qn)

        except Exception as ex:
            Qn = 0
            logging.error("couldn't measure noise of image")

        try:
            ########## GETTING BLUR ##############
            # low blur => blurry image (less 0.20 or 0.25)
            blur = self.measure_image_blur(image_source_path)
            #print "blur = " + str(blur)
            logging.info("blur = " + str(blur))

            ##### NORMALIZING BLUR BETWEEN [0,1] #####
            Qb = self.normalize_value_in_range_0_and_1(blur)
            print "normalized blur = " + str(Qb)

        except Exception as ex:
            Qb = 0
            logging.error("couldn't measure blurriness of image")

        try:
            ########## GETTING EXPOSURE ##############
            # low exposure => least colorful (less 0.10), high exposure => most colorful (more 0.80)
            exposure = self.measure_image_exposure(image_source_path)
            #print "exposure = " + str(exposure)
            logging.info("exposure = " + str(exposure))

            ##### NORMALIZING EXPOSURE BETWEEN [0,1] #####
            Qe = self.normalize_value_in_range_0_and_1(exposure)
            print "normalized exposure = " + str(Qe)

        except Exception as ex:
            Qe = 0
            logging.error("couldn't measure exposure of image")

        """
        Definition of quality measure:

        Parametrs:
        Q	- Quality of image
        Qe	- Exposure Quality (dark & bright saturation)
        Qb	- Blur Quality
        Qn	- Noise Quality

        *** All Measures - Blur, Noise, Exposure are in range [0,1] ***


        Formula of Initial Quality:
        Q = ([(C + Pe*Qe)]^We) * ([(C + Pb*Qb)]^Wb) * ([(C + Pn*Qn)]^Wn)

        where Wn are weights and C,Pn are Parameters

        Choosing values of weigths and parameters to match between Computer Asessment and Human Assesment
        Wn = 1
        Pn = 0.7
        C = 0.3

        Adding Normalized Features to Initial Formula:

        Final Quality Asessmant  Formula:

        Q = [(C+Pe*Qe)*(C+Pb*Qb)*(C+Pn*Qn) - 0.27] / 0.973

        """

        # definition of weights and parameters
        Pe = Pb = Pn = 0.7
        C = 0.3

        try:
            # definition of quality measure by formula:
            Q = ((C + Pe * Qe) * (C + Pb * Qb) * (C + Pn * Qn) - 0.27) / 0.973
        except Exception as ex:
            Q = 0
            print ex.message

        print "quality = " + str(Q)
        logging.info("quality = " + str(Q))

        return Q

    def filter_quality_of_single_face(self, image_source_path, current_face_number, img_name, threshold):
        quality_measure = self.measure_image_quality_version2_Stekel(image_source_path)
        print "\nQuality of face " + str(current_face_number) + " in image " + img_name + " is : " + str(
            quality_measure)
        if quality_measure < threshold:
            shutil.copy2(image_source_path, self.BAD_QUALITY_IMAGES_PATH + img_name)
            # shutil.move(image_source_path, BAD_QUALITY_IMAGES_PATH + img_name)
        return round(quality_measure, 5)

    def filter_quality_of_single_image(self, image_source_path, current_image_number, img_name, threshold):
        quality_measure = self.measure_image_quality_version2_Stekel(image_source_path)
        print "\nQuality of image " + str(current_image_number + 1) + " is : " + str(quality_measure)
        if quality_measure < threshold:
            shutil.copy2(image_source_path, self.BAD_QUALITY_IMAGES_PATH + img_name)
            # shutil.move(image_source_path, BAD_QUALITY_IMAGES_PATH + img_name)
        return round(quality_measure, 5)

    def filter_low_quality_images(self):

        BEFORE_IMAGES_PATH = self.PATH + 'PythonServer/quality_test/'

        list_of_images_in_directory = os.listdir(BEFORE_IMAGES_PATH)
        num_of_images = len(list_of_images_in_directory)

        if num_of_images > 0:

            logging.info("================== ESTIMATING IMAGE QUALITY =====================")
            print "Estimate image quality:"

            for current_image_number, img_name in enumerate(list_of_images_in_directory):

                image_source_path = BEFORE_IMAGES_PATH + img_name

                logging.info("\n================================================================================="
                             "\nimage " + str(current_image_number + 1) + " out of " + str(num_of_images) +
                             "\nimage name " + img_name)

                print "\n================================================================================="
                print "\nimage " + str(current_image_number + 1) + " out of " + str(num_of_images) + "\n"
                print "image name " + img_name

                # check if the file is a valid image
                # if not -> remove file from directory
                if self.check_if_file_is_valid_image(image_source_path) is not True:
                    # add log warning about invalid image in folder
                    os.remove(image_source_path)
                    continue

                try:

                    self.filter_quality_of_single_image(image_source_path, current_image_number, img_name, 0.42)

                except Exception as ex:
                    print "Couldent access quality of image " + str(current_image_number + 1) + "\n" + ex.message


    def test_image_quality_assessment_algorithm(self):

        BEFORE_IMAGES_PATH = self.PATH + 'PythonServer/image_quality_test/'

        list_of_images_in_directory = os.listdir(BEFORE_IMAGES_PATH)
        num_of_images = len(list_of_images_in_directory)

        if num_of_images > 0:

            print "================== ESTIMATING IMAGE QUALITY ====================="
            print "Estimate image quality:"

            for current_image_number, img_name in enumerate(list_of_images_in_directory):

                image_source_path = BEFORE_IMAGES_PATH + img_name

                print "\n================================================================================="
                print "\nimage " + str(current_image_number + 1) + " out of " + str(num_of_images) + "\n"
                print "image name " + img_name

                # check if the file is a valid image
                # if not -> remove file from directory
                if self.check_if_file_is_valid_image(image_source_path) is not True:
                    # add log warning about invalid image in folder
                    os.remove(image_source_path)
                    continue

                try:

                    self.filter_quality_of_single_image(image_source_path, current_image_number, img_name, 0)

                except Exception as ex:
                    print "Couldent access quality of image " + str(current_image_number + 1) + "\n" + ex.message

