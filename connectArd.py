import serial
import time
from watchdog.observers import Observer  
from watchdog.events import FileSystemEventHandler
from loadNN import NN
import glob
import os.path 

old=0
class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global old
        if event.is_directory:
            None
        elif event.event_type == 'created':
            print("created")
        elif event.event_type == 'modified':
            print("modified")
            files = os.listdir("/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/photos")
            paths = [os.path.join("/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/photos", basename) for basename in files]
            newest_file = max(paths, key=os.path.getctime)
            statbuf = os.stat(newest_file)
            new = statbuf.st_mtime
            if (new - old) > 0.5:
                result = NN(newest_file)
                print(result)

                ser = serial.Serial('com9',9600)
                time.sleep(2)
                if (result==0):
                    ser.write(b'0')
                elif (result==1):
                    ser.write(b'1')
                elif (result==2):
                    ser.write(b'2')
                elif (result==3):
                    ser.write(b'3')
            old = new      


if __name__ == "__main__":
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path="/Users/durstido/PycharmProjects/IEEEQP_WI19/data/pills/photos", recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
