class Node:
    def __init__(self,init):
        self.item = init
        self.next = None

    def getItem(self):
        return self.item

    def getNext(self):
        return self.next

    def setItem(self,newItem):
        self.item = newItem

    def setNext(self,newnext):
        self.next = newnext


class LinkedList:

    def __init__(self,name = None):
        self.head = None
        self.name = name

    def isEmpty(self):
        return self.head == None

    def add(self,item):
        temp = Node(item)
        temp.setNext(self.head)
        self.head = temp

    def size(self):
        current = self.head
        count = 0
        while current != None:
            count = count + 1
            current = current.getNext()

        return count

    def find(self,item):
        current = self.head
        found = False
        while current != None and not found:
            if current.getItem() == item:
                found = True
            else:
                current = current.getNext()

        return found

    def remove(self,item):
        current = self.head
        previous = None
        found = False
        while not found:
            if current.getItem() == item:
                found = True
            else:
                previous = current
                current = current.getNext()

        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())

