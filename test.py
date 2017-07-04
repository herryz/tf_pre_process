import sys
from time import sleep

adict = {}

if __name__ == '__main__':
    print len(sys.argv)

    adict["test"] = sys.argv[1]
    while 1:
        print("now num is %s" %(str(adict["test"])))
        sleep(2)