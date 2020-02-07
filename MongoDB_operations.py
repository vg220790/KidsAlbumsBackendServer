from pymongo import MongoClient
from collections import OrderedDict

"""
We are using a remote MongoDB on 'mLab' platform ("MongoDB Hosting: Database-as-a-Service by mLab")
https://mlab.com/databases/kids_albums_db
user: Kids_Albums
password: R1v2a3_afeka
"""

class MongoDB_operations():

    def __init__(self):
        self.MONGODB_URI = 'mongodb://admin:R1v2a3_afeka@ds145299.mlab.com:45299/kids_albums_db'
        self.host = 45299
        self.port = 'ds145299.mlab.com'
        self.client = None
        self.start_connection_with_client()
        self.db = self.client.kids_albums_db

    def start_connection_with_client(self):

        URI = self.MONGODB_URI
        # MONGODB_URI = 'mongodb://Kids_Albums:R1v2a3_afeka@ds145299.mlab.com:45299/kids_albums_db'

        try:
            self.client = MongoClient(URI, port=45299)
            self.client.PORT = self.port
            self.client.HOST = self.host
            print("Connected to remote MongoDB successfully!!!")
        except Exception as ex:
            print("Could not connect to remote MongoDB")


    def send_images_data_to_MongoDB(self, images_data):

        # database and collection
        try:
            collection = self.db.images_data
            # test_item = collection.find_one()
        except Exception as ex:
            print ex.message

        for dict in images_data:
            try:
                # json_data = json.dumps(dict)
                rec_id = collection.insert_one(dict)
                dict['mongodb record id'] = rec_id
                print dict
            except Exception as ex:
                print ex.message


    def add_users(self):

        users = []
        #children = self.create_children()

        first_names = ['YONATAN', 'AYA', 'GAYA', 'TUVI', 'ALMA', 'YUVAL', 'ASI', 'AVITAL', 'AVIV', 'DANA', 'DANI',
                       'DOR',
                       'EVYATAR', 'MAYA', 'NERI', 'NOA', 'NOAM', 'NOGA', 'SAGIV', 'SHAKED', 'SHANI', 'SOPHIE', 'TZADOK',
                       'YASMIN', 'DIMA', 'DAVIDA', 'KAI', 'YOAV', 'IDAN', 'DIKLA']

        last_names = ['LEVI', 'FRIDMAN', 'FARAJ', 'BIMBA', 'COHEN', 'DVIRI', 'AMSALEM', 'MARJI', 'BUHMAN', 'MEIRI',
                      'ENDELMAN', 'SHARONI', 'FISHMAN', 'GEVA', 'TAVOR', 'BENI', 'FISHMAN', 'YOVEL', 'BARUH', 'NAHMANI',
                      'NIMNI', 'ALTMAN', 'MAMO', 'BEN-SHIMON', 'GRAFMAN', 'YOSEFI', 'KIPER', 'SEGULA', 'ASHKENAZI',
                      'GRANATI']

        for i in range(1, 31):
            if (i < 10):
                aba_cell = '050-000000' + str(i)
                ima_cell = '054-000000' + str(i)
            else:
                aba_cell = '050-00000' + str(i)
                ima_cell = '054-00000' + str(i)

            child_first_name = str(first_names[i-1].lower())
            f_email = 'aba.' + child_first_name + '@gmail.com'
            f_password = child_first_name + "_01_afeka"
            f_name = "ABA_"+first_names[i-1] + " " + last_names[i-1]
            m_email = 'ima.' + child_first_name + '@gmail.com'
            m_password = child_first_name + "_02_afeka"
            m_name = "IMA_" + first_names[i - 1] + " " + last_names[i - 1]
            father = OrderedDict([('email', f_email), ('password', f_password), ('username',f_name), ('phone number', aba_cell),
                                  ('kindergarten id', 1), ('child first name', first_names[i-1]), ('child last name', last_names[i-1]), ('tag',i)])
            mother = OrderedDict([('email', m_email), ('password', m_password), ('username',m_name), ('phone number', ima_cell),
                                  ('kindergarten id', 1), ('child first name', first_names[i-1]), ('child last name', last_names[i-1]), ('tag',i)])

            users.append(father)
            users.append(mother)

        try:
            users_collection = self.db.users

            for user in users:
                try:
                    # json_data = json.dumps(dict)
                    rec_id = users_collection.insert_one(user)
                    #user['mongodb record id'] = rec_id
                    print user
                except Exception as ex:
                    print ex.message

        except Exception as ex:
            print ex.message

    def add_kindergarten_to_MongoDB(self):

        kindergarten_teachers = [OrderedDict([('name', "Shula"),('phone', '058-6270218')]),OrderedDict([('name', "Tova"),('phone', '054-4404838')])]

        children = self.create_children()

        kindergarten = OrderedDict([('id',1),('name', "Afeka"),('address',"Sirkin 22, Tel Aviv"),
                                    ('teachers', kindergarten_teachers),('children', children)])

        try:
            kindergartens_collection = self.db.kindergartens
            rec_id = kindergartens_collection.insert_one(kindergarten)
            print rec_id
        except Exception as ex:
            print ex.message


    def create_children(self):

        children = []

        ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
               "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]

        tags = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15",
                "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"]


        first_names = ['YONATAN', 'AYA', 'GAYA', 'TUVI', 'ALMA', 'YUVAL', 'ASI', 'AVITAL', 'AVIV', 'DANA', 'DANI',
                       'DOR',
                       'EVYATAR', 'MAYA', 'NERI', 'NOA', 'NOAM', 'NOGA', 'SAGIV', 'SHAKED', 'SHANI', 'SOPHIE', 'TZADOK',
                       'YASMIN', 'DIMA', 'DAVIDA', 'KAI', 'YOAV', 'IDAN', 'DIKLA']
        last_names = ['LEVI', 'FRIDMAN', 'FARAJ', 'BIMBA', 'COHEN', 'DVIRI', 'AMSALEM', 'MARJI', 'BUHMAN', 'MEIRI',
                      'ENDELMAN', 'SHARONI', 'FISHMAN', 'GEVA', 'TAVOR', 'BENI', 'FISHMAN', 'YOVEL', 'BARUH', 'NAHMANI',
                      'NIMNI', 'ALTMAN', 'MAMO', 'BEN-SHIMON', 'GRAFMAN', 'YOSEFI', 'KIPER', 'SEGULA', 'ASHKENAZI',
                      'GRANATI']

        birthdays = ['2013-01-15', '2013-02-15', '2013-03-15', '2013-04-15', '2013-05-15', '2013-06-15', '2013-07-15',
                     '2013-08-15',
                     '2013-09-15', '2013-10-15', '2013-11-15', '2013-12-15', '2013-01-17', '2013-02-17', '2013-03-17',
                     '2013-04-17',
                     '2013-05-17', '2013-06-17', '2013-07-17', '2013-08-17', '2013-09-17', '2013-10-17', '2013-11-17',
                     '2013-12-17',
                     '2013-01-22', '2013-02-22', '2013-03-22', '2013-04-22', '2013-05-22', '2013-06-22']

        for i in range(0, 30):

            if (i < 10):
                aba_cell = '050-000000' + str(i + 1)
                ima_cell = '054-000000' + str(i + 1)
            else:
                aba_cell = '050-00000' + str(i + 1)
                ima_cell = '054-00000' + str(i + 1)

            f_email = 'aba.' + str(first_names[i].lower()) + '@gmail.com'
            m_email = 'ima.' + str(first_names[i].lower()) + '@gmail.com'

            child = OrderedDict([('id', ids[i]), ('first name', first_names[i]), ('last name', last_names[i]), ('birthday', birthdays[i]),
                                 ('mother name', "IMA_"+first_names[i]),('mother phone', ima_cell),('mother email', m_email),
                                 ('father name', "ABA_"+first_names[i]),('father phone', aba_cell),('father email', f_email),
                                 ('tag',tags[i])])

            children.append(child)

        return children

