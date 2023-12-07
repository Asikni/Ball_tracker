import cv2
import numpy as np
import csv
from datetime import datetime
import os

video_path = 'C:/Users/HP/pythonProject1/AI_assignement/AI Assignment video.mp4'  #replace this with your video path
cap = cv2.VideoCapture(video_path)

color_ranges = {    
    #trying to capture the ball colours
    'yellow': (np.array([22, 93, 0]), np.array([45, 255, 255])),
    'green': (np.array([40, 45, 45]), np.array([80, 255, 255])),
    'white': (np.array([0, 0, 225]), np.array([180, 25, 255])),
    'orange': (np.array([5, 100, 100]), np.array([15, 255, 255]))
}

quadrants = {
    #fit the quadrants
    3: (780, 0, 1250, 520),
    4: (1250, 0, 1750, 520),    
    2: (780, 520, 1250, 1040),
    1: (1250, 520, 1750, 1040)
}

ball_positions = {color: None for color in color_ranges}
ball_entry_times = {color: None for color in color_ranges}

def is_inside_quadrant(x, y, quadrant):
    x1, y1, x2, y2 = quadrants[quadrant]
    return x1 <= x <= x2 and y1 <= y <= y2

def write_event(time, quadrant, color, event_type):   #creating csv file to store the events.
    csv_file_path = 'C:/Users/HP/pythonProject1/AI_assignement/ball_tracking_events_nw.csv'   #replace here you want to store the events
    try:
        # Check if the file exists and is empty
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, 'a', newline='') as file:  # 'a' is for append mode
            writer = csv.writer(file)
            if not file_exists or os.stat(csv_file_path).st_size == 0:
                # Write the header if the file is empty
                writer.writerow(['Time in Seconds', 'Quadrant Number', 'Ball Color', 'Event Type'])
            # Write the event data
            writer.writerow([time, quadrant, color, event_type])
        print("Event data added to CSV file.")
    except Exception as e:
        print("Failed to write to CSV file:", e)

frame_rate = cap.get(cv2.CAP_PROP_FPS)
start_time = datetime.now()


while True:
 # Capture frame-by-frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
     # Loop through each quadrant defined in the 'quadrants' dictionary
    for quadrant_number, (x1, y1, x2, y2) in quadrants.items():
     # Draw a blue rectangle around the quadrant
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 
        cv2.putText(frame, f'Q{quadrant_number}', (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    
    # Calculate the time elapsed since the start of the program
    time_elapsed = (datetime.now() - start_time).total_seconds()

     # Convert the frame from BGR to HSV color spac
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

     # Loop through each color range defined in the 'color_ranges' dictionary
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)  #gaussian blur to reduce noise
        mask = cv2.erode(mask, None, iterations=2) #to reduce white spots
        mask = cv2.dilate(mask, None, iterations=2)

        #find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)  #find the largest contour
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            if color == 'green':   #making changes in the tracker
                radius -= 9
            elif color == 'white':
                radius += 20
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                # Check if the detected object is inside any quadrant
                for quadrant_number in quadrants:
                    if is_inside_quadrant(x, y, quadrant_number):
                        if ball_positions[color] != quadrant_number:
                            if ball_entry_times[color] is None:   ## If the object has just entered the quadrant
                                ball_entry_times[color] = time_elapsed
                            elif time_elapsed - ball_entry_times[color] >= 3:   #ball is considered in the quadrant if it stays for 3 or more than 3 seconds
                                write_event(time_elapsed, quadrant_number, color, 'Entry')
                                ball_positions[color] = quadrant_number
                                ball_entry_times[color] = None
                        else:
                            ball_positions[color] = quadrant_number
                    elif ball_positions[color] == quadrant_number:
                        write_event(time_elapsed, quadrant_number, color, 'Exit')  # Write an event to the log file
                        ball_positions[color] = None 
                        ball_entry_times[color] = None
                        
    cv2.imshow('Ball Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # If 'q' is pressed, exit the loop
        break

cap.release()
cv2.destroyAllWindows()
