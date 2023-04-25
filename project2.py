import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from object_detection import ObjectDetection

#vehicle detection 
#vehicle tracking and counting
#vehicle type estimation (car, bus, truck, bike, etc.)
#vehicle speed estimation
#vehicle color estimation
#vehicle size estimation
#vehicle flow rate estimation
#traffic density estimation


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
        
        for box in boxes:
            x,y,w,h = box
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, "Vehicle Type", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
            cv2.putText(frame, "Speed: 50km/h", (x,y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
            cv2.putText(frame, "Color: Red", (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
            cv2.putText(frame, "Size: 5m", (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

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