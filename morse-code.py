import cv2
import mediapipe as mp
import numpy as np
import time
import collections
from scipy.spatial import distance as dist

MORSE_TO_CHAR = {
    ".-":    "A", "-...":  "B", "-.-.":  "C", "-..":   "D",
    ".":     "E", "..-.":  "F", "--.":   "G", "....":  "H",
    "..":    "I", ".---":  "J", "-.-":   "K", ".-..":  "L",
    "--":    "M", "-.":    "N", "---":   "O", ".--.":  "P",
    "--.-":  "Q", ".-.":   "R", "...":   "S", "-":     "T",
    "..-":   "U", "...-":  "V", ".--":   "W", "-..-":  "X",
    "-.--":  "Y", "--..":  "Z",
    # Digits
    "-----": "0", ".----": "1", "..---": "2", "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7",
    "---..": "8", "----.": "9",
    # Punctuation
    ".-.-.-": ".", "--..--": ",", "..--..": "?", ".----.": "'",
    "-.-.--": "!", "-..-.":  "/", "-.--.":  "(", "-.--.-": ")",
    ".-...":  "&", "---...": ":", "-.-.-.": ";", "-...-":  "=",
    ".-.-.":  "+", "-....-": "-", "..--.-": "_", ".-..-.": '"',
    "...-..-":"$", ".--.-.": "@", "...---...": "SOS",
}

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 387, 385, 263, 380, 373]

EAR_THRESHOLD     = 0.21   
DEBOUNCE_FRAMES   = 2      
EAR_SMOOTH_N      = 5      
DOT_MIN_MS        = 80
DOT_MAX_MS        = 500
DASH_MIN_MS       = 600
DASH_MAX_MS       = 1500
CHAR_BREAK_MS     = 1600   
WORD_BREAK_MS     = 3200   


def ear(landmarks, indices, w, h):
    """Compute Eye Aspect Ratio from 6 landmark indices."""
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)


def decode_morse(token):
    return MORSE_TO_CHAR.get(token, f"[{token}]")



def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

   
    ear_history = collections.deque(maxlen=EAR_SMOOTH_N)
    blink_start      = None    
    open_start       = None     
    last_state       = "OPEN"   
    confirm_closed   = 0
    confirm_open     = 0

    current_morse    = ""       
    decoded_text     = ""       
    last_signal_ms   = None    

    
    EAR_HISTORY_LEN  = 120
    ear_graph         = collections.deque([0.21] * EAR_HISTORY_LEN, maxlen=EAR_HISTORY_LEN)

   
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    COLOR_DOT  = (100, 220, 255)
    COLOR_DASH = (100, 100, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_ACC  = (0, 200, 120)
    COLOR_WARN = (60, 80, 220)
    BG_PANEL   = (20, 20, 28)

    def draw_panel(img, x, y, w, h, alpha=0.75):
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x+w, y+h), BG_PANEL, -1)
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    def ms():
        return time.time() * 1000

    status_msg   = ""
    status_color = COLOR_TEXT
    last_char    = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        now = ms()

       
        avg_ear = 0.21
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear  = ear(lm, LEFT_EYE,  w, h)
            right_ear = ear(lm, RIGHT_EYE, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0

        ear_history.append(avg_ear)
        smooth_ear = np.mean(ear_history)
        ear_graph.append(smooth_ear)

        eye_closed = smooth_ear < EAR_THRESHOLD

        
        if eye_closed:
            confirm_closed += 1
            confirm_open    = 0
        else:
            confirm_open   += 1
            confirm_closed  = 0

        
        if last_state == "OPEN" and confirm_closed >= DEBOUNCE_FRAMES:
            last_state   = "CLOSED"
            blink_start  = now
           
            if open_start and (now - open_start) >= WORD_BREAK_MS and current_morse:
                ch = decode_morse(current_morse)
                decoded_text += ch
                last_char     = ch
                current_morse = ""
                decoded_text += " "
                status_msg   = "WORD BREAK → space"
                status_color = (200, 180, 60)
            elif open_start and (now - open_start) >= CHAR_BREAK_MS and current_morse:
                ch = decode_morse(current_morse)
                decoded_text += ch
                last_char     = ch
                current_morse = ""
                status_msg   = f"CHAR → {ch}"
                status_color = COLOR_ACC

        
        if last_state == "CLOSED" and confirm_open >= DEBOUNCE_FRAMES:
            last_state = "OPEN"
            open_start = now
            if blink_start:
                duration = now - blink_start
                if DOT_MIN_MS <= duration <= DOT_MAX_MS:
                    current_morse    += "."
                    last_signal_ms    = now
                    status_msg        = f"DOT  ({duration:.0f}ms)"
                    status_color      = COLOR_DOT
                elif DASH_MIN_MS <= duration <= DASH_MAX_MS:
                    current_morse    += "-"
                    last_signal_ms    = now
                    status_msg        = f"DASH ({duration:.0f}ms)"
                    status_color      = COLOR_DASH
                elif duration > DASH_MAX_MS:
                    status_msg   = f"IGNORED ({duration:.0f}ms too long)"
                    status_color = COLOR_WARN
                blink_start = None

       
        if (last_state == "OPEN" and last_signal_ms and current_morse and
                (now - last_signal_ms) >= CHAR_BREAK_MS):
            ch = decode_morse(current_morse)
            decoded_text += ch
            last_char     = ch
            current_morse = ""
            last_signal_ms = None
            status_msg    = f"AUTO-DECODE → {ch}"
            status_color  = COLOR_ACC

      

       
        draw_panel(frame, 0, 0, 420, h)

       
        graph_x, graph_y = 20, 20
        graph_w, graph_h = 380, 90
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x+graph_w, graph_y+graph_h),
                      (40, 44, 55), -1)
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x+graph_w, graph_y+graph_h),
                      (80, 88, 100), 1)

        
        thresh_y = graph_y + graph_h - int((EAR_THRESHOLD / 0.5) * graph_h)
        cv2.line(frame, (graph_x, thresh_y), (graph_x+graph_w, thresh_y),
                 (80, 80, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "EAR threshold", (graph_x+4, thresh_y-4),
                    FONT, 0.36, (120, 120, 200), 1)

        pts = list(ear_graph)
        for i in range(1, len(pts)):
            x1 = graph_x + int((i-1) / EAR_HISTORY_LEN * graph_w)
            x2 = graph_x + int(i / EAR_HISTORY_LEN * graph_w)
            y1 = graph_y + graph_h - int(np.clip(pts[i-1]/0.5, 0, 1) * graph_h)
            y2 = graph_y + graph_h - int(np.clip(pts[i]  /0.5, 0, 1) * graph_h)
            color = (80, 160, 255) if pts[i] > EAR_THRESHOLD else (60, 80, 220)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"EAR: {smooth_ear:.3f}", (graph_x+4, graph_y+graph_h+18),
                    FONT, 0.55, COLOR_TEXT, 1)

        
        state_lbl  = "CLOSED" if eye_closed else "OPEN"
        state_col  = (60, 80, 220) if eye_closed else (60, 200, 120)
        cv2.rectangle(frame, (graph_x+230, graph_y+graph_h+4),
                      (graph_x+380, graph_y+graph_h+26), state_col, -1, cv2.LINE_AA)
        cv2.putText(frame, f"EYE: {state_lbl}", (graph_x+240, graph_y+graph_h+20),
                    FONT, 0.55, (255,255,255), 1)

        
        cv2.putText(frame, "CURRENT MORSE", (20, 155), FONT, 0.45, (140,140,160), 1)
        morse_display = current_morse if current_morse else "_ _ _"
        cv2.putText(frame, morse_display, (20, 195), FONT, 1.1,
                    COLOR_DOT if "." in morse_display else COLOR_DASH, 2, cv2.LINE_AA)

       
        cv2.putText(frame, "LAST CHAR", (20, 250), FONT, 0.45, (140,140,160), 1)
        cv2.putText(frame, last_char if last_char else "–", (20, 310),
                    cv2.FONT_HERSHEY_DUPLEX, 2.8, COLOR_ACC, 3, cv2.LINE_AA)

       
        cv2.putText(frame, status_msg, (20, 360), FONT, 0.5, status_color, 1)

        
        cv2.putText(frame, "DECODED TEXT", (20, 400), FONT, 0.45, (140,140,160), 1)
      
        words = decoded_text[-56:]
        line1 = words[:28]
        line2 = words[28:]
        cv2.putText(frame, line1, (20, 430), FONT, 0.75, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, line2, (20, 460), FONT, 0.75, COLOR_TEXT, 1, cv2.LINE_AA)

        
        cv2.putText(frame, "[C] Clear  [Q] Quit", (20, h-14),
                    FONT, 0.42, (100, 100, 120), 1)

        
        if eye_closed and blink_start:
            elapsed = now - blink_start
            ratio   = min(elapsed / DASH_MAX_MS, 1.0)
            ring_col = COLOR_DASH if elapsed > DASH_MIN_MS else COLOR_DOT
            cx, cy  = w - 80, h - 80
            cv2.circle(frame, (cx, cy), 50, (40,40,50), -1)
            cv2.ellipse(frame, (cx, cy), (50,50), -90,
                        0, int(360*ratio), ring_col, 4, cv2.LINE_AA)
            label = "DASH?" if elapsed > DASH_MIN_MS else "DOT?"
            cv2.putText(frame, label, (cx-22, cy+6), FONT, 0.45, ring_col, 1)

        cv2.imshow("Eye Morse Communicator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_morse = ""
            decoded_text  = ""
            last_char     = ""
            status_msg    = "Cleared"
            status_color  = COLOR_TEXT

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Final decoded text:", decoded_text)


if __name__ == "__main__":
    main()
