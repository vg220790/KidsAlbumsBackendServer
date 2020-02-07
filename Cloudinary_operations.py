import cloudinary.api
from cloudinary.api import delete_resources_by_tag, resources_by_tag
from cloudinary.uploader import upload
import logging
import os
import piexif
import piexif.helper
from cloudinary.utils import cloudinary_url

class Cloudinary_operations():

    def __init__(self, path, mongo):
        self.PATH = path
        self.mongodb = mongo
        self.URI = 'https://api.cloudinary.com/v1_1/kids-albums/images_for_albums/'
        self.public_id = 'kids-albums/images_for_albums/'
        self.cloud_name = "kids-albums"
        self.api_key = "246534377616525"
        self.api_secret = "dQDpTTAqZKmi33lQ6BSGLGJgT_U"
        # 'https://res.cloudinary.com/kids-albums/image/upload/'

    def upload_to_cloudinary(self):

        # local source folder for ready processed images
        READY_IMAGES_PATH = self.PATH + 'PythonServer/after_images/'

        logging.info('############Attempting push images to Dropbox:')
        logging.info('source path: ' + READY_IMAGES_PATH)
        logging.info('destination path: ' + self.URI)

        failed_upload_images = []
        images_data = []

        list_of_files_in_directory = os.listdir(READY_IMAGES_PATH)
        num_of_files = len(list_of_files_in_directory)

        # before uploading we check if the "after_images" directory is empty
        if num_of_files > 0:

            logging.info('######## Starting upload of ' + str(num_of_files) + ' files :')
            file_counter = 0

            for image_name in list_of_files_in_directory:

                source_path = READY_IMAGES_PATH + image_name
                image_public_id = self.public_id + image_name

                file_counter = file_counter + 1
                print("\nUploading " + str(file_counter) + " image out of " + str(len(list_of_files_in_directory)) + " to Cloudinary")
                # get exif from image before uploading
                info = piexif.load(source_path)
                usercmt = info['Exif'][37510]
                str_data = '{' + usercmt.split("{", 1)[1]
                dict_data = eval(str_data)

                # upload image and check if image uploaded successfuly (if image exists in dropbox folder)
                try:
                    # upload to cloudinary
                    response = cloudinary.uploader.upload(source_path,
                                               public_id=image_public_id,
                                               cloud_name=self.cloud_name,
                                               api_key=self.api_key,
                                               api_secret=self.api_secret)

                    link_to_image = response['url']
                    secure_link_to_image = response['secure_url']
                    dict_data['link'] = link_to_image
                    dict_data['secure link'] = secure_link_to_image
                    images_data.append(dict_data)
                    logging.info('Successfuly uploaded image ' + str(file_counter) + ' out of ' + str(num_of_files))
                    # os.remove(source_path)
                except Exception as ex:
                    logging.warning('Could not upload image no. ' + str(file_counter) + ', image name: ' + image_name)
                    logging.exception(ex.message)
                    failed_upload_images.append(image_name)

            self.mongodb.send_images_data_to_MongoDB(images_data)
        else:
            logging.info('There are no files to upload to Dropbox')

