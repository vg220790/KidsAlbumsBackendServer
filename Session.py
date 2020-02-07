from apscheduler.schedulers.blocking import BlockingScheduler
from Dropbox_operations import Dropbox_operations
from Cloudinary_operations import Cloudinary_operations
from MongoDB_operations import MongoDB_operations

from datetime import datetime
from Logic import Logic
import logging
import sys


class Session():

    def __init__(self, path, sql_pswd):
        self.LOCAL_PATH = path
        self.sql_pswd = sql_pswd
        self.schedualer = BlockingScheduler()
        self.mongodb = MongoDB_operations()
        self.cloudinary = Cloudinary_operations(self.LOCAL_PATH, self.mongodb)
        self.dbx = Dropbox_operations(self.LOCAL_PATH, self.mongodb)


    def run_session_with_schedualer(self):
        schedualer = self.schedualer
        # Schedules job_function to be run everyday at 01:00 and 13:00
        schedualer.add_job(self.run_session, 'cron', hour='20,13', minute='38')
        schedualer.start()

    def run_session_manualy(self):
        self.run_session()

    def run_session(self):

        logging.basicConfig(filename='pythonserver.log', level=logging.DEBUG)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info('\n\n################################################################\n')
        logging.info('######################## Session started at ' + str(current_time))

        # creating data in dropbox
        #self.dbx.push_images_to_dropbox()

        # step 1
        #self.dbx.pull_images_from_dropbox()
        # step 2
        self.run_engine()
        # step 3
        #self.cloudinary.upload_to_cloudinary()

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logging.info('######################## Session finished at ' + str(current_time))
        logging.info('\n################################################################\n\n')

    def run_engine(self):
        print("\nRunning engine\ninitializing Logic")
        logic = Logic(self.LOCAL_PATH, self.dbx, self.sql_pswd)

        # Load all the information about the kindergarten and it's kids from the database
        logic.loadKindergartenFromDatabase()
        logic.start()


