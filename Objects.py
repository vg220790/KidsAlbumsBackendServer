from LinkedList import LinkedList
from libxmp import exempi
from libxmp.consts import XMP_NS_Photoshop as NS_PHOTOSHOP
from libxmp.consts import XMP_OPEN_FORUPDATE, XMP_OPEN_READ
from libxmp.consts import XMP_CLOSE_SAFEUPDATE, XMP_CLOSE_NOOPTION
import os,sys
from PIL import Image
from PIL.ExifTags import TAGS
from libxmp.files import XMPFiles
from libxmp.utils import file_to_dict
from libxmp import consts
import numpy
from libxmp.utils import file_to_dict
from geopy.geocoders import Nominatim
from datetime import datetime
import exifread
import pyexiv2
import json

class Kindergarten():
    def __init__(self, name, email, phone, address, teacherPhone, teacher = 'admin', teacherAssistant = 'team', assistance = 'team1', children_no = 30):
        self.id = -1
        self.name = name
        self.email = email
        self.phone = phone
        self.address = address
        #self.address_longtitude_latitude = self.getGpsLocation(address)
        if teacher == "":
            self.teacher = "admin"
        else:
            self.teacher = teacher
        self.teacherPhone = teacherPhone
        if teacherAssistant == "":
            self.teacherAssistance = "team"
        else:
            self.teacherAssistance = teacherAssistant
        if assistance == "":
            self.assistance = "team1"
        else:
            self.assistance = assistance
        self.children = []
        self.schedule = {}
        self.albums = []
        self.images = []
        self.iteration_images = []
        self.labels = []
        self.children_no = children_no
        self.images_no = 0
        self.histogram = numpy.zeros(self.children_no + 1)
        self.all_rects_histograms = numpy.zeros(self.children_no + 1)
        self.clf = None

    def setId(self,id):
        self.id = id

    def setAddress(self,addressInStr):
        if addressInStr != None:
            self.address = eval(addressInStr)

    # Get coordinates of the address
    def getGpsLocation(self,address):
        geolocator = Nominatim()
        location = geolocator.geocode(address)
        if location != None:
            return (location.latitude,location.longitude)

    # Find the child with that tag and returns the email of his parents
    def getEmailOfChildTag(self,tag):
        parentsEmails = []
        if self.children != None and len(self.children) > 0:
            for ch in self.children:
                if ch.tag == tag:
                    if ch.father_email != None:
                        parentsEmails.append(ch.father_email)
                    if ch.mother_email != None:
                        parentsEmails.append(ch.mother_email)
                    return parentsEmails
            return None

    def getChildByFirstName(self, child_first_name):
        for child in self.children:
            if (child.first_name == child_first_name):
                return child

    # Adding new child to the kindergarten
    def add_new_child(self, child):
        self.children.append(child)

    # Create new instance of child and add it to kindergarten
    def addChild(self,name,last,birthDayDate,fatherName,fatherEmail,fatherCell,motherName,motherEmail,motherCell,webpage,child_tag,imagePath):
        for child in self.children:
            if name == child.first_name and last == child.last_name:
                return False
        ch = Child(name,last,birthDayDate,fatherName,fatherEmail,fatherCell,motherName,motherEmail,motherCell,webpage,child_tag,imagePath)
        self.children.append(ch)
        self.addToSchedule("Birthday of " + name + " " + last,birthDayDate,birthDayDate)
        albumOfChild = Album(child = ch)
        albumOfChild.setChildTag(ch.tag)
        self.albums.append(albumOfChild)
        return True

    # Add the image to the album in kindergarten which named like albumName
    def addImageToAlbum(self,albumName,imageName):
        for album in self.albums:
            if album.name == albumName:
                imagePath = '/Users/admin/PycharmProjects/Objects/trainSet/' + imageName + '.JPG'
                if imagePath not in album.selectedImages:
                    album.selectedImages.append(imagePath)
                    return album

    # Dictionary of event's names and dates
    def addToSchedule(self,eventName,startDate,endDate):
        try:
            if startDate == None and endDate == None:
                raise Exception("no events dates")
            if eventName not in self.schedule:
                newEvent = Event(eventName,startDate,endDate)
                self.schedule[eventName] = newEvent
            else:
                raise Exception("event exist!")
        except Exception as ex:
            print ex.message

    # Get date of an image and checks if there is any event in the kindergarten schedule
    # returns None if no event or the event itself
    def eventDetector(self,date):
        for key in self.schedule:
            if self.schedule[key].start.month == date.month and self.schedule[key].start.day == date.day:
                return self.schedule[key]
            elif self.schedule[key].end != None:
                if date > self.schedule[key].start and date <= self.schedule[key].end:
                    return self.schedule[key]
            else:
                return None

    # Get the children list of the kindergarten
    def getChildList(self):
        return self.children

    # Get the album list of the kindergarten
    def getAlbumsList(self):
        return self.albums

    # Get the child which the given mail is the same as his parents
    def getChildFromImage(self, emailOfParent):
        for child in self.children:
            if child.father_email == emailOfParent:
                return child
            elif child.mother_email == emailOfParent:
                return child
        return None

    def __repr__(self):
        res = "Kindergarten: " + self.name + "\n"
        res += "Email: " + str(self.email) + "\n"
        res += "Phone: " + str(self.phone) + "\n"
        res += "Address: " + str(self.address) + "\n"
        #not must
        res += "Teacher: " + str(self.teacher) + "\n"
        res += "Teacher Phone: " + str(self.teacherPhone) + "\n"
        res += "Assistance #1: " + str(self.teacherAssistance) + "\n"
        res += "Assistance #2: " + str(self.assistance) + "\n"
        res += "Children Count: " + str(len(self.children)) + "\n"
        res += "Albums Count: " + str(len(self.albums))

        return res


class Child():
    #todo: uri of child site??
    def __init__(self, first_name, last_name, birthDayDate, fatherName, fatherEmail, fatherCell, motherName,
                 motherEmail, motherCell,webpage,child_tag,imagePath=None):
        self.id = -1
        self.kindergartenId = -1
        self.first_name = first_name
        self.last_name = last_name
        self.tag = child_tag
        self.birthDayDate = birthDayDate
        self.father_name = fatherName
        self.father_email = fatherEmail
        self.father_cell = fatherCell
        self.mother_name = motherName
        self.mother_email = motherEmail
        self.mother_cell = motherCell
        self.uri = webpage
        self.image_path = imagePath
        self.images_of_child = []#LinkedList()
        self.avg_feature_vector = []  # avarage vector that contains the avarage of all the vectors
        self.feature_vector = []  # vector that contains all the words that we think belongs to child
        self.std = []
        self.variance = []
        self.recognition = 0

    def setId(self,id):
        self.id = id

    def setKindergartenId(self,kindergartenId):
        self.kindergartenId = kindergartenId

    def add_to_feature_vector(self , new_vec):
        self.feature_vector.append(new_vec)

    def getImagesOfChild(self):
        return self.images_of_child

    def __repr__(self):
        res = "Child: " + self.first_name + " " + self.last_name + "\n"
        res += "Date Of Birth: " + datetime.strftime(self.birthDayDate,"%Y-%m-%d") + "\n"
        res += "Father: " + self.father_name + "\t" + self.father_cell + "\t" + self.father_email + "\n"
        res += "Mother: " + self.mother_name + "\t" + self.mother_cell + "\t" + self.mother_email + "\n"
        return res


class Event():
    def __init__(self,name = None,startDateTime = None,endDateTime = None,loadDB = False):
        try:
            self.id = -1
            self.kindergartenId = -1
            if loadDB:
                if name != None:
                    self.name = name
                else:
                    raise Exception("event doesn't contain name")
            else:
                # if event doesn't contains name, the date is the name
                if startDateTime != None :
                    if name != None:
                        self.name = datetime.strftime(startDateTime,"%Y-%m-%d") + "\t" + name
                    else:
                        self.name = datetime.strftime(startDateTime,"%Y-%m-%d")
                else:
                    raise Exception("event doesn't contain name")
            self.start = startDateTime
            self.end = endDateTime
            self.eventImages = []
            self.selectedImages = []
        except Exception as ex:
            print ex.message

    def setId(self,id):
        self.id = id

    def setKindergartenId(self,id):
        self.kindergartenId = id

    def addImages(self,images):
        existImages = set(self.eventImages)
        for image in images:
            existImages.add(image)
        self.eventImages = list(existImages)

    def getImages(self):
        #convert all images into list of strings -- talk with karin
        return self.eventImages

    def getSelectedImages(self):
        return self.selectedImages

    def getEventImages(self):
        return self.eventImages

    def getSelectedImages(self):
        return self.selectedImages


    def __repr__(self):
        res = "Event: " + str(self.name) + "\n"
        res += "Start Date Time: " + str(datetime.strftime(self.start,"%Y-%m-%d")) + "\n"
        if self.end != None:
            res += "End Date Time: " + str(datetime.strftime(self.end,"%Y-%m-%d"))
        return res


class Album():
    def __init__(self,name = None,child = None):
        self.id = -1
        self.kindergarten = -1
        self.tag = -1
        self.name = self.makeAlbumName(name,child)
        self.selectedImages = []

    def setId(self,id):
        self.id = id

    def setKindergartenId(self,kindergartenId):
        self.kindergarten = kindergartenId

    def setChildTag(self,tag):
        self.tag = tag

    # if it is private album of the child then its name is the child's name, else the album name is the given name
    def makeAlbumName(self,name,child):
        nameStr = ""
        try:
            if name == None and child == None:
                raise Exception("Album must have name or child")

            if name != None:
                if child != None:
                    nameStr += child.first_name + " " + child.last_name + " -> " + name
                else:
                    nameStr += name
            else:
                nameStr += child.first_name + " " + child.last_name

            return nameStr
        except Exception as ex:
            print ex.message

    def addImageToSelectedImages(self,image):
        self.selectedImages.append(image)

    def getSelectedImages(self):
        return self.selectedImages

    def __repr__(self):
        res = "Album: " + self.name + "\n"
        res += "Images Count: " + str(len(self.selectedImages))
        return res


class ImageObject():
    def __init__(self,name,path):
        self.name = name
        self.path = path
        self.timestamp = None
        self.author = {}
        self.author_email = ""
        self.child_of_author_name = ""
        self.child_of_author_tag = 0
        self.are_there_detected_faces = False
        self.detected_faces = {}
        self.number_of_detected_faces = 0
        self.faces = []
        self.exif = {}
        self.json_exif = None
        self.word_details = WordDetails()
        self.image_tag = 0
        self.face_rect_to_image_size_threshold = 0.0475
        self.treshold = 0.75
        self.height = 0
        self.width = 0
        self.rects = {}

    def apply_attributes_from_exiv(self, dictionary):
        self.exif = dictionary
        self.timestamp = dictionary['timestamp']
        self.author = dictionary['author']
        self.author_email = self.author['author_email']
        child = self.author['child_of_author']
        self.child_of_author_name = child['name']
        self.child_of_author_tag = child['tag_number']

    def get_faces(self):
        return self.faces

    def update_faces(self,faces):
        self.faces = faces

    def add_face_to_image(self,face):
        self.faces.append(face)
        self.number_of_detected_faces += 1
        if self.are_there_detected_faces is False:
            self.are_there_detected_faces = True

    def write_exif(self,value):
        meta = pyexiv2.ImageMetadata(self.path)
        meta.read()
        childTag = {'ChildTag': value}
        meta['Exif.Photo.UserComment'] = json.dumps(childTag)
        meta.write()

    def get_all_exif(self):
        f = open(self.path, 'rb')
        self.exif = exifread.process_file(f)
        for tag in self.exif.keys():
            if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
                print "Key: %s, value %s" % (tag, self.exif[tag])
        return self.exif

    def get_tag_exif(self,tag):
        f = open(self.path, 'rb')
        self.exif = exifread.process_file(f)
        for key in self.exif.keys():
            if key == tag:
                return self.exif[key]

    def write_xmp(self,tag,value):
        xfptr = exempi.files_open_new(self.path,XMP_OPEN_FORUPDATE)
        xmp = exempi.files_get_new_xmp(xfptr)
        if xmp is not None:
            exempi.set_property(xmp,NS_PHOTOSHOP,tag,value,0)
            exempi.files_put_xmp(xfptr,xmp)
            exempi.free(xmp)
        exempi.files_close(xfptr, XMP_CLOSE_SAFEUPDATE)
        exempi.files_free(xfptr)

    def write_xmp_vika(self, tag, value):
        path = self.path
        xmpfile = XMPFiles(file_path=path)
        xmp = xmpfile.get_xmp()
        xmp.append_array_item(consts.XMP_NS_DC, tag, value,
                              {'prop_array_is_ordered': True, 'prop_value_is_array': True})

    def get_xmp(self,tag):
        xfptr = exempi.files_open_new(self.path, XMP_OPEN_READ)
        xmp = exempi.files_get_new_xmp(xfptr)
        the_prop, _ = exempi.get_property(xmp, NS_PHOTOSHOP, tag)
        exempi.free(xmp)
        exempi.files_close(xfptr, XMP_CLOSE_NOOPTION)
        exempi.files_free(xfptr)

    def __repr__(self):
        res = "Image: " + self.name + "\n"
        res += "Path: " + str(self.path) + "\n"
        # add more relevant printings
        return res


class Rectangle():
    def __init__(self, startX, startY, height, width,importance,faceImage,parentImage):
        self.startX = startX
        self.startY = startY
        self.height = height
        self.width = width
        self.importance = importance
        self.parent_image = parentImage
        self.normalized_importance = 0
        self.faceImage = faceImage
        self.faceRep = []
        self.id = 0
        self.area = 0
        self.predict_tag = 0
        self.normalized_prob = 0

    def update_face_rep(self,rep):
        self.faceRep = rep
        #setattr(self,self.faceRep, rep)

    def get_rep(self):
        return self.faceRep

    def __repr__(self):
        res = "Starting point:" + str(self.startX) + "," + str(self.startY)+"\n"
        res += "Height: " + str(self.height) + " Width:" + str(self.width)+"\n"
        return res


class WordDetails():
    def __init__(self):
        self.feature_vector = []
        self.credibility = 0
        self.tag = ""

    def __repr__(self):
        res = "Tag: " + self.tag + "\n"
        res += "Credibility: "+ str(self.credibility) + "\n"
        res += "Feature vector : " + "\n"
        res += self.feature_vector
        return res