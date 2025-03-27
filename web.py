import streamlit as st
import numpy as np
import cv2
import torch
import keras
import tensorflow as tf
import mediapipe as mp
from ultralytics import YOLO
from streamlit_option_menu import option_menu

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with st.sidebar:
    menu = option_menu("Main Menu", ["Tic-Tac-Toe", "Object Detection", "Gesture Recognition"],
                       icons=["grid-3x3", "camera", "hand-index-thumb"],
                       menu_icon="cast", default_index=0)

# Tic-Tac-Toe Game
if menu == "Tic-Tac-Toe":
    st.title("Tic-Tac-Toe 🤖 vs ❌")
    
    HUMAN, AI, EMPTY = -1, 1, 0

    def check_winner(board):
        for row in board:
            if abs(sum(row)) == 3:
                return np.sign(sum(row))
        for col in board.T:
            if abs(sum(col)) == 3:
                return np.sign(sum(col))
        if abs(sum(board.diagonal())) == 3:
            return np.sign(sum(board.diagonal()))
        if abs(sum(np.fliplr(board).diagonal())) == 3:
            return np.sign(sum(np.fliplr(board).diagonal()))
        return 0 if 0 in board else 2

    def minimax(board, depth, is_maximizing):
        winner = check_winner(board)
        if winner != 0:
            return winner * (10 - depth) if winner != 2 else 0

        if is_maximizing:
            best_score = -np.inf
            for i in range(3):
                for j in range(3):
                    if board[i, j] == EMPTY:
                        board[i, j] = AI
                        score = minimax(board, depth + 1, False)
                        board[i, j] = EMPTY
                        best_score = max(best_score, score)
            return best_score
        else:
            best_score = np.inf
            for i in range(3):
                for j in range(3):
                    if board[i, j] == EMPTY:
                        board[i, j] = HUMAN
                        score = minimax(board, depth + 1, True)
                        board[i, j] = EMPTY
                        best_score = min(best_score, score)
            return best_score

    def best_move(board):
        best_score = -np.inf
        move = None
        for i in range(3):
            for j in range(3):
                if board[i, j] == EMPTY:
                    board[i, j] = AI
                    score = minimax(board, 0, False)
                    board[i, j] = EMPTY
                    if score > best_score:
                        best_score = score
                        move = (i, j)
        return move

    if "board" not in st.session_state:
        st.session_state.board = np.zeros((3, 3), dtype=int)
        st.session_state.turn = HUMAN
        st.session_state.game_over = False

    def render_board():
        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                symbol = "❌" if st.session_state.board[i, j] == HUMAN else "⭕" if st.session_state.board[i, j] == AI else " "
                button_disabled = st.session_state.board[i, j] != EMPTY or st.session_state.game_over
                if cols[j].button(symbol, key=f"{i}-{j}", disabled=button_disabled, use_container_width=True):
                    if st.session_state.turn == HUMAN:
                        st.session_state.board[i, j] = HUMAN
                        st.session_state.turn = AI
                        st.rerun()
    
    render_board()

    winner = check_winner(st.session_state.board)
    if winner:
        st.session_state.game_over = True
        st.success("🎉 Player (❌) Wins!" if winner == HUMAN else "🤖 AI (⭕) Wins!" if winner == AI else "😲 It's a Draw!")
    
    if st.session_state.turn == AI and not st.session_state.game_over:
        move = best_move(st.session_state.board)
        if move:
            st.session_state.board[move] = AI
            st.session_state.turn = HUMAN
            st.rerun()
    
    if st.button("🔄 Restart Game", use_container_width=True):
        st.session_state.board = np.zeros((3, 3), dtype=int)
        st.session_state.turn = HUMAN
        st.session_state.game_over = False
        st.rerun()

# Object Detection
if menu == "Object Detection":
    st.title("Real-time Object Detection using YOLOv8")
    st.write("Start webcam to detect objects in real-time.")
    
    model = YOLO("best.pt")  # Gunakan model YOLOv8 pre-trained
    
    if "object_detection_active" not in st.session_state:
        st.session_state.object_detection_active = False
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start Detection"):
            st.session_state.object_detection_active = True
    with col2:
        if st.button("Stop Detection"):
            st.session_state.object_detection_active = False
    
    stframe = st.empty()
    
    if st.session_state.object_detection_active:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            st.error("Failed to access webcam.")
        else:
            while st.session_state.object_detection_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame)
                annotated_frame = results[0].plot()  # Menampilkan bounding box
                stframe.image(annotated_frame, channels="BGR")
            cap.release()


# Gesture Recognition
if menu == "Gesture Recognition":
    st.title("Real-time Gesture Recognition")
    st.write("Start webcam to recognize gestures in real-time.")
    
    try:
        gesture_model = keras.models.load_model("keras_model.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        gesture_model = None
    
    class_labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]
    
    if "gesture_recognition_active" not in st.session_state:
        st.session_state.gesture_recognition_active = False
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start Gesture Recognition"):
            st.session_state.gesture_recognition_active = True
    with col2:
        if st.button("Stop Gesture Recognition"):
            st.session_state.gesture_recognition_active = False
    
    stframe = st.empty()
    
    if st.session_state.gesture_recognition_active:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
        
        if not cap.isOpened():
            st.error("Failed to access webcam.")
        else:
            while st.session_state.gesture_recognition_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_min, y_min = 640, 480
                        x_max, y_max = 0, 0
                        
                        for lm in hand_landmarks.landmark:
                            x, y = int(lm.x * 640), int(lm.y * 480)
                            x_min, y_min = min(x, x_min), min(y, y_min)
                            x_max, y_max = max(x, x_max), max(y, y_max)
                        
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                image_resized = cv2.resize(frame, (224, 224))
                image_normalized = image_resized / 255.0
                image_expanded = np.expand_dims(image_normalized, axis=0)
                
                prediction = gesture_model.predict(image_expanded)
                confidence = np.max(prediction)
                if 0.4 <= confidence <= 0.9:
                    predicted_class = class_labels[np.argmax(prediction)]
                    cv2.putText(frame, f"{predicted_class} ({confidence:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                stframe.image(frame, channels="BGR")
            
            cap.release()