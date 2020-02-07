
import os, os.path
import logging
import dropbox
import piexif
import piexif.helper


class Dropbox_operations():

    def __init__(self, path, mongo):
        self.PATH = path
        self.access_token = '9WHpN5kNjbAAAAAAAAAAEeKeHGVoW5-n6PGNroaWEnrmxD9udZCMnIVNc4AwKjrZ'
        self.dbx = dropbox.Dropbox(self.access_token)
        self.mongodb = mongo


    def pull_images_from_dropbox(self):

        # dropbox source folder for images that are recieved from clients (later via app)
        DROPBOX_SOURCE_PATH = '/Photos_Before_Processing/'

        # locat dest folder for raw images that will be processed on this machine
        RAW_IMAGES_PATH = self.PATH + 'PythonServer/before_images/'

        # creating a dropbox instance
        dbx = self.dbx

        failed_download_images = []
        entries_of_images_to_delete_from_dropbox = []

        if dbx is not None:

            logging.info('############  Attempting pull images from Dropbox:')
            logging.info('source path: ' + DROPBOX_SOURCE_PATH)
            logging.info('destination path: ' + RAW_IMAGES_PATH)

            entries = dbx.files_list_folder(DROPBOX_SOURCE_PATH).entries
            num_of_files_in_dropbox = len(entries)

            if num_of_files_in_dropbox > 0:

                logging.info('######## Starting download of ' + str(num_of_files_in_dropbox) + ' files :')
                print '######## Starting download of ' + str(num_of_files_in_dropbox) + ' files :'
                current_file_number = 0

                # downloading images from dropbox to local folder
                for entry in entries:
                    #if(current_file_number==11):
                    #    break
                    image_name = entry.name
                    source = DROPBOX_SOURCE_PATH + image_name
                    dest = RAW_IMAGES_PATH + image_name

                    # download image from source to dest
                    dbx.files_download_to_file(dest, source)
                    current_file_number = current_file_number + 1

                    # check if file exists in local folder (successful download)
                    if not os.path.exists(dest):
                        logging.warnings(
                            'Could not download image no. ' + str(current_file_number) + ', image name: ' + image_name)
                        print 'Could not download image no. ' + str(current_file_number) + ', image name: ' + image_name
                        failed_download_images.append(image_name)
                    # if download successful remove file from dropbox
                    else:
                        logging.info('Successfuly downloaded image ' + str(current_file_number) + ' out of ' + str(
                            num_of_files_in_dropbox))
                        print 'Successfuly downloaded image ' + str(current_file_number) + ' out of ' + str(
                            num_of_files_in_dropbox)
                        dbx.files_delete_v2(source)

            else:
                logging.info('There are no files to download from Dropbox')

    def push_images_to_dropbox(self):

        # local source folder for ready processed images
        READY_IMAGES_PATH = self.PATH + 'PythonServer/dropbox_images_to_add/'
        # READY_IMAGES_PATH = '/home/victoria/original_project/PythonServer/trainSet/'

        # dropbox dest folder for images that are ready for client side (creating albums)
        #DROPBOX_DEST_PATH = '/Photos_After_Processing_For_Client/'
        DROPBOX_DEST_PATH = '/Photos_Before_Processing/'

        # creating a dropbox instance
        dbx = self.dbx

        logging.info('############Attempting push images to Dropbox:')
        logging.info('source path: ' + READY_IMAGES_PATH)
        logging.info('destination path: ' + DROPBOX_DEST_PATH)

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
                dest_path = DROPBOX_DEST_PATH + image_name

                file_counter = file_counter + 1

                # get exif from image before uploading
                info = piexif.load(source_path)
                usercmt = info['Exif'][37510]
                str_data = '{' + usercmt.split("{", 1)[1]
                dict_data = eval(str_data)

                # upload image and check if image uploaded successfuly (if image exists in dropbox folder)
                try:
                    # uploading the image
                    with open(source_path, 'rb') as f:
                        dbx.files_upload(f.read(), dest_path)

                    metadata = dbx.files_get_metadata(dest_path)

                    # if file exists (successfuly uploaded) we can delete it from locat folder
                    if metadata is not None:
                        result = dbx.files_get_temporary_link(dest_path)
                        link_to_image = result.link
                        #dict_data['link'] = link_to_image
                        print dict_data
                        images_data.append(dict_data)
                        logging.info('Successfuly uploaded image ' + str(file_counter) + ' out of ' + str(num_of_files))
                        # os.remove(source_path)
                except Exception as ex:
                    logging.warning('Could not upload image no. ' + str(file_counter) + ', image name: ' + image_name)
                    failed_upload_images.append(image_name)

            #self.mongodb.send_images_data_to_MongoDB(images_data)
        else:
            logging.info('There are no files to upload to Dropbox')

    def upload_batch_to_dropbox(self):

        # local source folder for ready processed images
        READY_IMAGES_PATH = self.PATH + 'PythonServer/after_images/'

        # dropbox dest folder for images that are ready for client side (creating albums)
        DROPBOX_DEST_PATH = '/Photos_After_Processing_For_Client/'

        # creating a dropbox instance
        dbx = self.dbx

        files_to_upload = []

        for image_name in os.listdir(READY_IMAGES_PATH):
            files_to_upload.append(READY_IMAGES_PATH + image_name)

        dbx.files_upload_session_start(files_to_upload[0])

        for f in files_to_upload[1:]:
            current_cursor = dbx.files_list_folder_get_latest_cursor(f)
            dbx.files_upload_session_append_v2(f, current_cursor)

        dbx.files_upload_session_finish_batch(dbx.files_list_folder(DROPBOX_DEST_PATH).entries)

