import cv2

video_path = r"C:\Users\Kristila\Downloads\video.mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

coords = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{x},{y}", (x+8, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"Point {len(coords)}: [{x}, {y}]")
        cv2.imshow("Click zone corners - press Q when done", frame)

cv2.imshow("Click zone corners - press Q when done", frame)
cv2.setMouseCallback("Click zone corners - press Q when done", on_click)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()

print("\nCopy this into zones.json:")
print('"pixel_coords":', coords)
