import cv2
import os

# Define the input size of the model
input_size = (224, 224)

# Open the web cam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)

# Set the save directory
save_path = "/Users/jeikei/Documents/deep_summer_dataset/webcam_dataset"

# Make sub directories if not exists
os.makedirs("{}/on_mask".format(save_path), exist_ok=True)
os.makedirs("{}/off_mask".format(save_path), exist_ok=True)

# Counting the number of collected images for each class.
on_mask_index = 0
off_mask_index = 0

# Mask status variable
is_mask = False

while cap.isOpened():
    # Reading frames from the camera
    ret, original_frame = cap.read()
    if not ret:
        break

    # Copy the original frame
    frame_to_show = cv2.copyTo(original_frame, None)

    # Add Information on screen
    msg_mask = "Mask "
    msg_mask += "On" if is_mask else "Off"

    cv2.putText(frame_to_show, msg_mask, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
    cv2.putText(frame_to_show, "Mask On: {:03d}".format(on_mask_index), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
    cv2.putText(frame_to_show, "Mask Off: {:03d}".format(off_mask_index), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)

    # Show the frame
    cv2.imshow('Wear a face mask', frame_to_show)

    # Press Q on keyboard to  exit
    in_key = cv2.waitKey(25)
    if in_key & 0xFF == ord('q'):
        break
    elif in_key & 0xFF == ord('m'):
        # Changing the mask status
        is_mask = not is_mask
    elif in_key & 0xFF == ord('s'):
        # Save the current frame
        path = save_path
        path += "/on_mask" if is_mask else "/off_mask"

        if is_mask:
            path += "/{:03d}.jpg".format(on_mask_index)
            on_mask_index += 1
        else:
            path += "/{:03d}.jpg".format(off_mask_index)
            off_mask_index += 1

        cv2.imwrite(path, original_frame)





