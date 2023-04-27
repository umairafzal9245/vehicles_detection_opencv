import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from object_detection import ObjectDetection

def type_of_vehicle(class_id):
    if class_id == 1:
        return "Bicycle"
    elif class_id == 2:
        return "Car"
    elif class_id == 3:
        return "Motorcycle"
    elif class_id == 5:
        return "Bus"
    elif class_id == 7:
        return "Truck"
    else :
        return "Unknown"


def speed_of_vehicle(dist, time):
    return dist//time

def color_of_vehicle():
    pass

def size_of_vehicle():
    pass

def flow_rate_of_vehicle():
    pass

def density_of_vehicle():
    pass

def find_centroid(x,y,w,h):
    x1 = x
    y1 = y
    x2 = x+w
    y2 = y+h
    cx = (x1+x2)//2
    cy = (y1+y2)//2
    return cx,cy


class Application(tk.Frame):
    
    def __init__(self, master=None):
        
        super().__init__(master)
        self.master = master
        self.pack()

        self.canvas_height = 500
        self.canvas_width = 800

        self.count_line_position = self.canvas_height - 150

        self.totalvehicles = 0
        self.vehicle_density = 0
        self.vehicle_flow_rate = 0

        self.trackvehicles = {}

        self.od = ObjectDetection()

        self.cap = cv2.VideoCapture('./los_angeles.mp4')

        self.create_widgets()

        self.update_Video()


    def create_widgets(self):

        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()

        self.quit_button = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy)
        self.quit_button.pack(side="bottom")


        self.vehicle_count_text = tk.StringVar()
        self.vehicle_flow_rate_text = tk.StringVar()
        self.vehicle_density_text = tk.StringVar()

        self.vehicle_count_text.set("Vehicle Count: 0")
        self.vehicle_flow_rate_text.set("Vehicle Flow Rate: 0")
        self.vehicle_density_text.set("Vehicle Density: 0")

        self.vehicle_count_label = tk.Label(self, textvariable=self.vehicle_count_text)
        self.vehicle_count_label.pack(side="bottom")

        self.vehicle_flow_rate_label = tk.Label(self, textvariable=self.vehicle_flow_rate_text)
        self.vehicle_flow_rate_label.pack(side="bottom")

        self.vehicle_density_label = tk.Label(self, textvariable=self.vehicle_density_text)
        self.vehicle_density_label.pack(side="bottom")

    def update_Video(self):

        ret, frame = self.cap.read()

        if not ret:
            print("Video Ended")
            self.cap.release()
            self.master.destroy()
            return
        
        frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))

        (class_ids, confidences, boxes) = self.od.detect(frame)
        
        for i in range(len(boxes)):
            class_id = class_ids[i]
            x,y,w,h = boxes[i]
            
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, "Vehicle Type "+type_of_vehicle(class_id), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
            
            cx,cy = find_centroid(x,y,w,h)
            new_vehicle = True
            for id, vehicle in self.trackvehicles.items():

                center = vehicle['center']
                
                dist = np.sqrt((center[0]-cx)**2 + (center[1]-cy)**2)
                
                if dist < 30:
                    new_vehicle = False
                    self.trackvehicles[id]['center'] = (cx,cy)
                    cv2.putText(frame, str(id), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 2)
                    if cx > self.canvas_width - 50 or cy > self.canvas_height - 50:
                        del self.trackvehicles[id]
                        break
                
            if new_vehicle:
                self.totalvehicles += 1
                self.trackvehicles[self.totalvehicles] = {'center':(cx,cy),'time':0,'color':'blue','size':0}
            
            

        self.vehicle_count_text.set("Vehicle Count: " + str(self.totalvehicles))
        self.vehicle_flow_rate_text.set("Vehicle Flow Rate: " + str(self.vehicle_flow_rate))
        self.vehicle_density_text.set("Vehicle Density: " + str(self.vehicle_density))

        self.image = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(self.image)

        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.create_line(0,self.count_line_position, self.canvas_width, self.count_line_position,fill="red")

        self.master.after(1, self.update_Video)
            

if __name__ == '__main__':
   
    root = tk.Tk()
    root.title("Vehicle Detection")
    root.geometry("800x600")
    app = Application(master=root)
    app.mainloop()
