import wx
from wx.adv import DatePickerCtrl
import Logic
import datetime

# Represents UI page which contains edit boxes and buttons to get all user inputs.
# contains also buttons events to send all the information to the logic unit of the system
class WizardPage(wx.Panel):
    def __init__(self, parent,logic,title=None,pageType = -1):
        """Constructor"""
        wx.Panel.__init__(self, parent)
        self.logic = logic
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        if title:
            title = wx.StaticText(self, -1, title)
            title.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
            sizer.Add(title, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
            sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND | wx.ALL, 5)

        if pageType == 0:
            inputsSizer = wx.GridBagSizer(2, 9)
            self.lblname = wx.StaticText(self, label="Name: ")
            self.editname = wx.TextCtrl(self, size=(140, -1))
            self.lblemail = wx.StaticText(self, label="Email: ")
            self.editemail = wx.TextCtrl(self, size=(140, -1))
            self.lblphone = wx.StaticText(self, label="Phone: ")
            self.editphone = wx.TextCtrl(self, size=(140, -1))
            self.lbladdress = wx.StaticText(self, label="Address: ")
            self.editaddress = wx.TextCtrl(self, size=(140, -1))
            self.lblteacher = wx.StaticText(self, label="Teacher Name: ")
            self.editteacher = wx.TextCtrl(self, size=(140, -1))
            self.lblteacherphone = wx.StaticText(self, label="Teacher Phone: ")
            self.editteacherphone = wx.TextCtrl(self, size=(140, -1))
            self.lblassistance = wx.StaticText(self, label="Assistance Name: ")
            self.editassistance = wx.TextCtrl(self, size=(140, -1))
            self.okbutton = wx.Button(self, label="OK")
            inputsSizer.Add(self.lblname, (0, 0))
            inputsSizer.Add(self.editname, (0, 1))
            inputsSizer.Add(self.lblemail, (1, 0))
            inputsSizer.Add(self.editemail, (1, 1))
            inputsSizer.Add(self.lblphone, (2, 0))
            inputsSizer.Add(self.editphone, (2, 1))
            inputsSizer.Add(self.lbladdress, (3, 0))
            inputsSizer.Add(self.editaddress, (3, 1))
            inputsSizer.Add(self.lblteacher, (4, 0))
            inputsSizer.Add(self.editteacher, (4, 1))
            inputsSizer.Add(self.lblteacherphone, (5, 0))
            inputsSizer.Add(self.editteacherphone, (5, 1))
            inputsSizer.Add(self.lblassistance, (6, 0))
            inputsSizer.Add(self.editassistance, (6, 1))
            inputsSizer.Add(self.okbutton, (8, 0), (1, 2), flag=wx.EXPAND)
            sizer.Add(inputsSizer,wx.ALIGN_BOTTOM,5)

            self.okbutton.Bind(wx.EVT_BUTTON, self.OnKinderOkButton)
        elif pageType == 1:
            inputsSizer = wx.GridBagSizer(2, 13)
            self.lblfirstname = wx.StaticText(self, label="First Name: ")
            self.editfirstname = wx.TextCtrl(self, size=(140, -1))
            self.lbllastname = wx.StaticText(self, label="Last Name: ")
            self.editlastname = wx.TextCtrl(self, size=(140, -1))
            self.lblbirthday = wx.StaticText(self, label="Birthday: ")
            self.editbirthday = DatePickerCtrl(self,size=(140, -1))
            self.lblfathername = wx.StaticText(self, label="Father Name: ")
            self.editfathername = wx.TextCtrl(self, size=(140, -1))
            self.lblfatheremail = wx.StaticText(self, label="Father Email: ")
            self.editfatheremail = wx.TextCtrl(self, size=(140, -1))
            self.lblfathercell = wx.StaticText(self, label="Father Mobile: ")
            self.editfathercell = wx.TextCtrl(self, size=(140, -1))
            self.lblmothername = wx.StaticText(self, label="Mother Name: ")
            self.editmothername = wx.TextCtrl(self, size=(140, -1))
            self.lblmotheremail = wx.StaticText(self, label="Mother Email: ")
            self.editmotheremail = wx.TextCtrl(self, size=(140, -1))
            self.lblmothercell = wx.StaticText(self, label="Mother Mobile: ")
            self.editmothercell = wx.TextCtrl(self, size=(140, -1))
            self.lblwebpage = wx.StaticText(self, label="Web Page: ")
            self.editwebpage = wx.TextCtrl(self, size=(140, -1))
            self.lblimagepath = wx.StaticText(self, label="Image: ")
            self.editimagepath = wx.TextCtrl(self, size=(140, -1))

            self.okChildButton = wx.Button(self, label="Add To Kindergarten")
            inputsSizer.Add(self.lblfirstname, (0, 0))
            inputsSizer.Add(self.editfirstname, (0, 1))
            inputsSizer.Add(self.lbllastname, (1, 0))
            inputsSizer.Add(self.editlastname, (1, 1))
            inputsSizer.Add(self.lblbirthday, (2, 0))
            inputsSizer.Add(self.editbirthday, (2, 1))
            inputsSizer.Add(self.lblfathername, (3, 0))
            inputsSizer.Add(self.editfathername, (3, 1))
            inputsSizer.Add(self.lblfatheremail, (4, 0))
            inputsSizer.Add(self.editfatheremail, (4, 1))
            inputsSizer.Add(self.lblfathercell, (5, 0))
            inputsSizer.Add(self.editfathercell, (5, 1))
            inputsSizer.Add(self.lblmothername, (6, 0))
            inputsSizer.Add(self.editmothername, (6, 1))
            inputsSizer.Add(self.lblmotheremail, (7, 0))
            inputsSizer.Add(self.editmotheremail, (7, 1))
            inputsSizer.Add(self.lblmothercell, (8, 0))
            inputsSizer.Add(self.editmothercell, (8, 1))
            inputsSizer.Add(self.lblwebpage, (9, 0))
            inputsSizer.Add(self.editwebpage, (9, 1))
            inputsSizer.Add(self.lblimagepath, (10, 0))
            inputsSizer.Add(self.editimagepath, (10, 1))
            inputsSizer.Add(self.okChildButton, (12, 0), (1, 2), flag=wx.EXPAND)
            sizer.Add(inputsSizer, wx.ALIGN_BOTTOM, 5)

            self.okChildButton.Bind(wx.EVT_BUTTON, self.OnChildOkButton)
        elif pageType == 2:
            inputsSizer = wx.GridBagSizer(4, 4)
            self.lblschedule = wx.StaticText(self, label="Please enter planned events: (e.g. Purim)")
            self.lbleventname = wx.StaticText(self, label="Name: ")
            self.editeventname = wx.TextCtrl(self, size=(140, -1))
            self.lblstart = wx.StaticText(self, label="Start: ")
            self.editstart = DatePickerCtrl(self,size=(140, -1))
            self.lblend = wx.StaticText(self, label="End: ")
            self.editend = DatePickerCtrl(self, size=(140, -1))
            self.okEventButton = wx.Button(self,label="Add Event")

            inputsSizer.Add(self.lblschedule,(0,0),(1,4))
            inputsSizer.Add(self.lbleventname,(1,0))
            inputsSizer.Add(self.editeventname,(1,1))
            inputsSizer.Add(self.lblstart,(2,0))
            inputsSizer.Add(self.editstart, (2, 1))
            inputsSizer.Add(self.lblend, (2, 2))
            inputsSizer.Add(self.editend, (2, 3))
            inputsSizer.Add(self.okEventButton,(3,1),(1,2))
            sizer.Add(inputsSizer, wx.ALIGN_CENTER, 5)

            self.okEventButton.Bind(wx.EVT_BUTTON, self.OnEventOkButton)
        elif pageType == 3:
            inputsSizer = wx.GridBagSizer(1,3)
            self.lblselect = wx.StaticText(self,label="Select child from list")
            self.childsDic = self.logic.getChildsNames()
            if self.childsDic != None:
                self.childsNamesList = self.childsDic.values()
            else:
                self.childsNamesList = []
            self.combonames = wx.ListBox(self,-1,(20,20),(120,80),self.childsNamesList,wx.LB_SINGLE)
            self.deleteButton = wx.Button(self,label="Delete")
            inputsSizer.Add(self.lblselect,(0,0))
            inputsSizer.Add(self.combonames,(0,1))
            inputsSizer.Add(self.deleteButton,(0,2))
            sizer.Add(inputsSizer, wx.ALIGN_CENTER, 5)

            self.deleteButton.Bind(wx.EVT_BUTTON,self.OnSelectButton)
        else:
            print "type of page: " + str(pageType)

    def OnKinderOkButton(self,e):
        try:
            if self.editname.GetValue() != "" and self.editemail.GetValue() != "" and self.editphone.GetValue() != "" and self.editaddress.GetValue() != "":
                done = self.logic.createKindergarten(self.editname.GetValue(),self.editemail.GetValue(),self.editphone.GetValue(),self.editaddress.GetValue(),self.editteacher.GetValue(),self.editteacherphone.GetValue(),self.editassistance.GetValue())
                if done:
                    wx.MessageBox("Go to next step to insert children","Done",wx.OK | wx.ICON_INFORMATION)
                else:
                    wx.MessageBox("This kindergarten exist", "Done", wx.OK | wx.ICON_INFORMATION)
            else:
                raise Exception("some of inputs empty or incorrect")
        except Exception as ex:
            wx.MessageBox(ex.message, "Message", wx.OK | wx.ICON_ERROR)
            print ex.message

    def OnChildOkButton(self,e):
        try:
            if (self.editfirstname.GetValue() != "" and self.editlastname.GetValue() != "" and self.editbirthday.GetValue() != None and self.editfathername.GetValue() != ""
            and self.editfatheremail.GetValue() != "" and self.editfathercell != "" and self.editmothername.GetValue() != ""
            and self.editmotheremail.GetValue() != "" and self.editmothercell.GetValue() != "" and self.editwebpage.GetValue() != ""
            and self.editimagepath.GetValue() != ""):
                done = self.logic.addChildToKindergarten(self.editfirstname.GetValue(),
                                                         self.editlastname.GetValue(),
                                                         datetime.date(self.editbirthday.GetValue().GetYear(),self.editbirthday.GetValue().GetMonth() + 1,self.editbirthday.GetValue().GetDay()),
                                                         self.editfathername.GetValue(),self.editfatheremail.GetValue(),self.editfathercell.GetValue(),
                                                         self.editmothername.GetValue(),self.editmotheremail.GetValue(),self.editmothercell.GetValue(),self.editwebpage.GetValue(),-1,
                                                         self.editimagepath.GetValue())
                if done:
                    wx.MessageBox("Add more children or click next","Done",wx.OK | wx.ICON_INFORMATION)
                    self.editfirstname.Clear()
                    self.editlastname.Clear()
                    self.editfathername.Clear()
                    self.editfatheremail.Clear()
                    self.editfathercell.Clear()
                    self.editmothername.Clear()
                    self.editmotheremail.Clear()
                    self.editmothercell.Clear()
                    self.editwebpage.Clear()
                    self.editimagepath.Clear()
                else:
                    raise Exception("adding child failed")
            else:
                raise Exception("some of inputs empty or incorrect")
        except Exception as ex:
            wx.MessageBox(ex.message, "Message", wx.OK | wx.ICON_ERROR)
            print ex.message

    def OnEventOkButton(self,e):
        try:
            startDate = datetime.date(self.editstart.GetValue().GetYear(),self.editstart.GetValue().GetMonth() + 1,self.editstart.GetValue().GetDay())
            endDate = datetime.date(self.editend.GetValue().GetYear(), self.editend.GetValue().GetMonth() + 1,self.editend.GetValue().GetDay())
            done = self.logic.addEventToKindergartenSchedule(self.editeventname.GetValue(),startDate,endDate)
            if done == False:
                raise Exception("adding event failed")
        except Exception as ex:
            wx.MessageBox(ex.message, "Message", wx.OK | wx.ICON_ERROR)
            print ex.message

    def OnSelectButton(self,e):
        if self.combonames.GetSelection() == wx.NOT_FOUND:
            wx.MessageBox("No child selection", "Error", wx.OK | wx.ICON_ERROR)
        else:
            deletechildId = -1
            for item in self.childsDic.iteritems():
                if item[1] == self.childsNamesList[self.combonames.GetSelection()]:
                    deletechildId = item[0]
            if deletechildId != -1:
                if self.logic.deleteChild(deletechildId):
                    wx.MessageBox("", "Done", wx.OK | wx.ICON_INFORMATION)
                else:
                    wx.MessageBox("", "Failed", wx.OK | wx.ICON_INFORMATION)

# Represents container for the ui pages
# containing addpage function to add new pages and next/prev functions to control the pages
class WizardPanel(wx.Panel):
    def __init__(self, parent,logic):
        """Constructor"""
        wx.Panel.__init__(self, parent=parent)
        self.logic = logic
        self.pages = []
        self.page_num = 0

        self.mainSizer = wx.BoxSizer(wx.VERTICAL)
        self.panelSizer = wx.BoxSizer(wx.VERTICAL)
        btnSizer = wx.BoxSizer(wx.HORIZONTAL)


        # add prev/next buttons
        self.prevBtn = wx.Button(self, label="Previous")
        self.prevBtn.Bind(wx.EVT_BUTTON, self.onPrev)
        btnSizer.Add(self.prevBtn, 0, wx.ALL | wx.ALIGN_RIGHT, 5)

        self.nextBtn = wx.Button(self, label="Next")
        self.nextBtn.Bind(wx.EVT_BUTTON, self.onNext)
        btnSizer.Add(self.nextBtn, 0, wx.ALL | wx.ALIGN_RIGHT, 5)

        # finish layout
        self.mainSizer.Add(self.panelSizer, 1, wx.EXPAND)
        self.mainSizer.Add(btnSizer, 0, wx.ALIGN_RIGHT)
        self.SetSizer(self.mainSizer)


    def addPage(self, title=None,pageType = -1):
        panel = WizardPage(self,self.logic,title,pageType)
        self.panelSizer.Add(panel, 2, wx.EXPAND)
        self.pages.append(panel)
        if len(self.pages) > 1:
            # hide all panels after the first one
            panel.Hide()
            self.Layout()

    def onNext(self, event):
        pageCount = len(self.pages)
        if pageCount - 1 != self.page_num:
            self.pages[self.page_num].Hide()
            self.page_num += 1
            self.pages[self.page_num].Show()
            self.panelSizer.Layout()
        else:
            print "End of pages!"

        if self.nextBtn.GetLabel() == "Finish":
            # close the app
            self.GetParent().Close()

        if pageCount == self.page_num + 1:
            # change label
            self.nextBtn.SetLabel("Finish")

    def onPrev(self, event):
        pageCount = len(self.pages)
        if self.page_num - 1 != -1:
            self.pages[self.page_num].Hide()
            self.page_num -= 1
            self.nextBtn.SetLabel("Next")
            self.pages[self.page_num].Show()
            self.panelSizer.Layout()
        else:
            print "You're already on the first page!"

# UI wizard containing 4 pages for getting user inputs(kindergarten,child,event details, children management)
# get the instance of logic in order to call to functions in our system
class MainFrame(wx.Frame):
    def __init__(self,logic):
        wx.Frame.__init__(self, None, title="Kindergarten Creation", size=(600,450))
        self.logic = logic
        self.panel = WizardPanel(self,self.logic)
        self.panel.addPage("Please enter all the details",0)
        self.panel.addPage("Add child's details",1)
        self.panel.addPage("Dates",2)
        self.panel.addPage("Configure",3)
        self.Show()
